import os
import sys
import time
import argparse
 
from dataset.dataset import DATASET_LIST, \
    TVA_SET_LIST, AV_SET_LIST, TV_SET_LIST, \
    get_num_classes, build_train_val_test_datasets

from dataset.pdataloader import ParallelLoaderPool, DataLoader

from models.fusion_modules import Fusion_List

from models.basic_model import  \
    M_TEXT_NAME, M_AUDIO_NAME, M_VISUAL_NAME, \
    KEY_HELPERS, KEY_ENCODERS, KEY_FUSION, \
    KEY_TEXT_TOKENS, KEY_TEXT_PADDING_MASK, \
    forward_encoders, forward_fusion, forward_helper, \
    gen_model
    
    
from utils import print_args, set_save_path, TeeOutput, \
    printDebugInfo, setup_seed

from metrics import performanceMetric
from prettytable import PrettyTable

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


MAIN_DEVICE_KEY = "MAIN_DEVICE"
def gen_model_gpu(args):
    """
        gen model run on a single GPU (main device)
    """
    device_map = {
        M_TEXT_NAME: 0,
        M_AUDIO_NAME: 0,
        M_VISUAL_NAME: 0,
        MAIN_DEVICE_KEY: 0
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    model = gen_model(args)
    model.to(device_map[MAIN_DEVICE_KEY])
    return model, device_map


def update_arg(parser, arg_name: str, **kwargs):
    to_remove = []
    for action in parser._actions:
        if arg_name in action.option_strings:
            to_remove.append(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)

    for action in to_remove:
        parser._actions.remove(action)

    parser.add_argument(arg_name, **kwargs)

class BasicTrainer:
    def __init__(self, args_str=None):
        self.parser = self.init_parser()
        self.args = self.init_parser().parse_args(args_str)
        self.init_logging()
        print_args(self.args)
        
        self.init_multi_gpu_env()
        self.init_env()
        self.model, self.device_map = self.init_model()
        self.modality_name_list = list(self.model[KEY_ENCODERS].keys())
        self.modality_name_list.sort()
        
        try:
            self.init_dataloader(self.args.using_ploader)
            self.init_optimizer_scheduler()
        except Exception as e:
            print(f"Error in init_dataloader: {e}")
            self.release()
            raise e
        
        self.softmax=nn.Softmax(dim=1)
        self.criterion=nn.CrossEntropyLoss()
        
        # measuring the time
        self.train_val_epoch_time_list = []
        self.train_time_list = []
        self.train_first_batch_time_list = []
        self.val_time_list = []
        self.val_first_batch_time_list = []
        
             
    def init_parser(self):
        # subclass can inherit this function and add more arguments
        # or override arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='CREMAD', type=str,
                            help="now supported: " + str(DATASET_LIST),
                            choices=DATASET_LIST)
    
        parser.add_argument('--fusion_method', default='concat', type=str,
                            choices=Fusion_List)

        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--epochs', default=100, type=int)

        parser.add_argument('--learning_rate', default=0.001,
                            type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_step', default=70,
                            type=int, help='where learning rate decays')
        parser.add_argument('--lr_decay_ratio', default=0.1,
                            type=float, help='decay coefficient')

        
        parser.add_argument('--random_seed', default=0, type=int)
        
        # for logging and tensorboard
        parser.add_argument('--save_path', default='../ckpt',
                            type=str, help='path to save results')
        
        parser.add_argument('--prefix', default='Naive', type=str,    
                            help='prefix for the save path')

        # for parallel training, but not used in this version
        parser.add_argument('--device_list', default="0", 
                            type=lambda s: list(map(int, s.split(','))),
                            help='list of devices to use, eg. 0,1 or 0,1,2,3')
        # only single is supported in this version
        parser.add_argument('--parallel_method', default='single', type=str,
                            choices=['dp', 'ddp', 'blp', 'single'],)
        # if you have more than 64G memory, you can use it, otherwise, --no_using_ploader
        # this is a optimization for dataloader
        parser.add_argument('--no_using_ploader', dest='using_ploader', action='store_false',
                            help='Disable parallel dataloader')
        parser.set_defaults(using_ploader=True)
        
        # run test after validation every epoch
        parser.add_argument('--no_test', dest='run_test', action='store_false',
                            help='Disable test')
        parser.set_defaults(run_test=True)
        
        ## using the default is ok, just change the config py file
        # These are for other projects, not used in this version
        # if you have problems, like your device does not support TF32,
        # you can use --no_tf32 to disable it
        parser.add_argument('--data_path', default='', type=str,
                            help='dataset path')
        parser.add_argument('--no_tf32', dest='with_tf32', action='store_false',
                            help='Disable test')
        parser.set_defaults(with_tf32=True)
        return parser

    def init_logging(self):
        if self.args.parallel_method != "ddp":
            self.save_path = set_save_path(self.args)
        else:
            self.rank = int(os.environ["LOCAL_RANK"])
            self.save_path = set_save_path(self.args, sub_dir=f"rank_{self.rank}")

        sys.stdout = TeeOutput(
            os.path.join(self.save_path, 'output.log'))
        self.tsb_writer = SummaryWriter(self.save_path)

    
    def init_multi_gpu_env(self):
        # this part is removed, it is about our another project
        # sorry for the inconvenience
        return 
        
    
    def init_env(self):
        setup_seed(self.args.random_seed)
        # For A100 speed up
        print("TF32 for A100/A40")
        print("matmul.allow_tf32 = " + str(torch.backends.cuda.matmul.allow_tf32))
        print("cudnn.allow_tf32 = " + str(torch.backends.cudnn.allow_tf32))
        
        if self.args.with_tf32:
            print("Using TF32 for A100/40")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("matmul & cudnn using default settings")
        print("matmul.allow_tf32 = " + str(torch.backends.cuda.matmul.allow_tf32))
        print("cudnn.allow_tf32 = " + str(torch.backends.cudnn.allow_tf32))
    
    def init_model(self):
        parallel_methods = {
            "single": gen_model_gpu
        }

        if self.args.parallel_method in parallel_methods:
            model, device_map = parallel_methods[self.args.parallel_method](self.args)
        else:
            raise NotImplementedError(f"Parallel method {self.args.parallel_method} not supported")
            
        # print the model structure
        print(model, 
              file=open(os.path.join(
                  self.save_path, 
                  'model.txt'), 'w'))
        return model, device_map
    
    def init_optimizer_scheduler(self):
        # assign an individual optimizer for each model part
        optimizer_map = {
            key: optim.SGD(
                self.model[key].parameters(),
                lr=self.args.learning_rate, 
                momentum=0.9, 
                weight_decay=1e-4
            ) for key in [KEY_ENCODERS, KEY_FUSION, KEY_HELPERS]
        }
        
        # assign an individual scheduler for each model part
        scheduler_map = {
            key: optim.lr_scheduler.StepLR(
                optimizer_map[key], 
                self.args.lr_decay_step,
                self.args.lr_decay_ratio
            ) for key in [KEY_ENCODERS, KEY_FUSION, KEY_HELPERS]
        }
        
        self.optimizer_map = optimizer_map
        self.scheduler_map = scheduler_map
        
    def init_dataset(self):
        self.train_dataset, self.val_dataset, self.test_dataset\
        = build_train_val_test_datasets(self.args)
        self.n_classes =  get_num_classes(self.args.dataset)
    
    def init_dataloader(self, using_ploader):
        self.init_dataset()
        
        self.dataloader_kwargs = {
            "batch_size": self.args.batch_size,
            "num_workers": 16,
            "pin_memory": True,
            # "persistent_workers": True,
            # "prefetch_factor": 2
        }
        
        # normal dataloader
        self.using_ploader = using_ploader
        print(f"Using Parallel Dataloader: {self.using_ploader}")
    
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            shuffle=True, 
            **self.dataloader_kwargs
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, 
            shuffle=False, 
            **self.dataloader_kwargs
        )

        self.test_dataloader = DataLoader(
            self.test_dataset, 
            shuffle=False, 
            **self.dataloader_kwargs
        )
        
        if using_ploader:
            self.val_loader_pool = ParallelLoaderPool(
                self.val_dataset,
                name= "val", 
                shuffle=False, 
                **self.dataloader_kwargs
            )
            self.test_loader_pool = ParallelLoaderPool(
                self.test_dataset,
                name = "test", 
                shuffle=False, 
                **self.dataloader_kwargs
            )
            if self.args.parallel_method == "ddp":
                world_size = dist.get_world_size()
                # for ddp, we need to use the same seed for each epoch
                print("Using DistributedSampler for train dataloader")
                train_dataloader_kwargs = {
                        "batch_size": self.args.batch_size//world_size,
                        "num_workers": 16//world_size,
                        "pin_memory": True,
                        "prefetch_factor": 2,
                        "persistent_workers": True,
                        "sampler": DistributedSampler(self.train_dataset, shuffle=True)
                }
                self.train_loader_pool = ParallelLoaderPool(
                    self.train_dataset, 
                    name= "train",
                    **train_dataloader_kwargs
                )
            else: 
                self.train_loader_pool = ParallelLoaderPool(
                    self.train_dataset, 
                    name= "train",
                    shuffle=True,
                    **self.dataloader_kwargs
                )
        
    # for compatibility with the parallel dataloader
    def get_trainloader(self, epoch):
        if self.using_ploader:
            return self.train_loader_pool.get_loader(
                epoch, epoch<self.args.epochs - 1)
        
        return self.train_dataloader
    
    def get_valloader(self,epoch):
        if self.using_ploader:
            return self.val_loader_pool.get_loader(
                epoch, epoch<self.args.epochs - 1)
        return self.val_dataloader
    
    def get_testloader(self,epoch):
        if self.using_ploader:
            return self.test_loader_pool.get_loader(
                epoch, epoch<self.args.epochs - 1)
        
        return self.test_dataloader

    # metric
    def reinitialize_metrics(self):
        """
        Reinitialize the metrics，
        these metrics are common for train, val and test
        and all methods.
        """ 
        # record the fusion results, and the results of each modality
        self.m_map = {"f": performanceMetric(self.n_classes, name="f")}

        for modality_name in self.modality_name_list:
            self.m_map[modality_name] = performanceMetric(
                self.n_classes, name=modality_name)
            
        # record the helper results of each modality
        self.m_h_map = {}
        for modality_name in self.modality_name_list:
            self.m_h_map[modality_name] = performanceMetric(
                self.n_classes, name=f"{modality_name}_h")
    
    def print_metrics(self, mode = "train"):
        dataloader_map = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader
        }
        if mode in dataloader_map:
            batches = len(dataloader_map[mode])
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'val', or 'test'.")
        
        print("Fusion, and each modality acc: ")
        print_loss_and_acc(self.epoch, 
                        self.m_map, batches,
                        self.tsb_writer,
                        mode)
        
        print("Helper acc: ")
        print_loss_and_acc(self.epoch,
                        self.m_h_map, batches,
                        self.tsb_writer, 
                        mode + "_h")
        
        # also save the metrics of each mode
        if mode == "train":
            self.train_m_map = self.m_map
            self.train_m_h_map = self.m_h_map
        elif mode == "val":
            self.val_m_map = self.m_map
            self.val_m_h_map = self.m_h_map
        elif mode == "test":
            self.test_m_map = self.m_map
            self.test_m_h_map = self.m_h_map
        
    # forward function
    def prepare_input_dict(self, dataset, data_packet):
        """
        Prepare the input dictionary for the model based on the dataset and data packet.
        Args:
            dataset (str): The name of the dataset.
            data_packet (tuple): The data packet containing the input features and labels.
            # device_map (dict): A dictionary mapping device names to device objects.
        Returns:
            input_dict (dict): A dictionary containing the input features for the model.
            labels (torch.Tensor): The labels for the input data.
            extra_infos (list): Additional information about the input data (id).
        """
        device_map = self.device_map

        if dataset in TVA_SET_LIST:
            tokenizers, padding_masks, images, audio_features, \
            labels, extra_infos = data_packet
            tokenizers = tokenizers.to(self.device_map[M_TEXT_NAME])
            padding_masks = padding_masks.to(self.device_map[M_TEXT_NAME])
            audio_features = audio_features.to(device_map[M_AUDIO_NAME]).unsqueeze(1).float()
            images = images.to(device_map[M_VISUAL_NAME]).float()
            input_dict = {
                M_TEXT_NAME: {KEY_TEXT_TOKENS: tokenizers, 
                            KEY_TEXT_PADDING_MASK: padding_masks},
                M_AUDIO_NAME: audio_features,
                M_VISUAL_NAME: images
            }
        elif dataset == 'TCGA':
            # patches: (B, N_patches, PATCH_DIM) → visual slot
            # omics  : (B, OMICS_DIM)            → audio slot
            patches, omics, labels, extra_infos = data_packet
            patches = patches.to(device_map[M_VISUAL_NAME]).float()
            omics   = omics.to(device_map[M_AUDIO_NAME]).float()
            input_dict = {
                M_VISUAL_NAME: patches,
                M_AUDIO_NAME:  omics,
            }
        elif dataset in AV_SET_LIST:
            audio_features, images, labels, extra_infos = data_packet
            audio_features = audio_features.to(device_map[M_AUDIO_NAME]).unsqueeze(1).float()
            images = images.to(device_map[M_VISUAL_NAME]).float()
       
            input_dict = {
                M_AUDIO_NAME: audio_features,
                M_VISUAL_NAME: images
            }
        elif dataset in TV_SET_LIST:
            tokenizers, padding_masks, images, labels, extra_infos = data_packet
            tokenizers = tokenizers.to(device_map[M_TEXT_NAME])
            padding_masks = padding_masks.to(device_map[M_TEXT_NAME])
            images = images.to(device_map[M_VISUAL_NAME]).float()
            input_dict = {
                M_TEXT_NAME: {KEY_TEXT_TOKENS: tokenizers, 
                            KEY_TEXT_PADDING_MASK: padding_masks},
                M_VISUAL_NAME: images
            }
        else:
            raise NotImplementedError(f"Dataset not supported: {dataset}")
        return input_dict, labels, extra_infos

    def forward(self, data_packet, model=None):
        """
        Forward pass for the model.
        forward the encoders, fusion layer and helper
        """
        if model is None:
            model = self.model
                
        device_map = self.device_map
        modality_name_list = self.modality_name_list
        softmax=self.softmax
        criterion=self.criterion
        m_map = self.m_map
        m_h_map = self.m_h_map
        
        with torch.profiler.record_function("prepare_input_dict"):
            input_dict, labels, infos = \
                self.prepare_input_dict(self.args.dataset, 
                                        data_packet)
                
            labels_device = labels.to(device_map[MAIN_DEVICE_KEY])
        
        with torch.profiler.record_function("forward_encoders"):
            embedding_dict = forward_encoders(model[KEY_ENCODERS], input_dict)
        
        local_embedding_dict = {}
        for modality_name in modality_name_list:
            # it is just for BL parallelism (move data to the main device)
            # for single GPU, DP parallelism (it is not needed, but for consistency,
            # all data is moved automatically to the main device)
            # for DDP parallelism, main device is same as the rank of the process
            local_embedding_dict[modality_name] = \
                embedding_dict[modality_name].detach().to(
                device_map[MAIN_DEVICE_KEY])
            
        def forward_fusion_step(embedding_dict):
            with torch.profiler.record_function("forward_fusion"), \
            torch.no_grad():
                out_f = forward_fusion(model[KEY_FUSION], embedding_dict)
                
                if m_map is not None and "f" in m_map:
                    out_f_pred = softmax(out_f)
                    out_f_loss = criterion(out_f, labels_device)
                    m_map["f"].update(out_f_pred, labels_device, loss=out_f_loss)
                
                for modality_name in modality_name_list:
                    if m_map is not None and modality_name in m_map:
                        out_x = model[KEY_FUSION].get_out_m(modality_name)
                        out_x_pred = softmax(out_x)
                        out_x_loss = criterion(out_x, labels_device)
                        m_map[modality_name].update(out_x_pred, labels_device, loss=out_x_loss)

        # Call the function
        forward_fusion_step(local_embedding_dict)
                    
        def forward_helper_step(embedding_dict):
            with torch.profiler.record_function("forward_helper"):
                helper_out_dict = forward_helper(model[KEY_HELPERS], embedding_dict)
                
                # helper metric
                for modality_name in modality_name_list:
                    if m_h_map is not None and modality_name in m_h_map:
                        out_h_x = helper_out_dict[modality_name]
                        out_h_x_pred = softmax(out_h_x)
                        out_h_x_loss = criterion(out_h_x, labels_device)
                        m_h_map[modality_name].update(
                            out_h_x_pred, labels_device, loss=out_h_x_loss)

            return helper_out_dict

        # Call the function
        helper_out_dict = forward_helper_step(local_embedding_dict)
        torch.cuda.synchronize()
        return embedding_dict, helper_out_dict, labels_device

    # train and valid function
    def after_forward_batch(self, embedding_dict, labels_device):
        """
        override this function to do something like 
        collecting the metrics after forwarding each batch
        """
        
        pass
    
    def before_train_epoch(self):
        pass 
    
    def after_train_epoch(self):
        pass
        
    def before_valid(self):
        pass
    
    def after_valid(self):
        pass
    
    def before_test(self):
        pass
    
    def after_test(self):
        pass

    def after_summary(self):
        pass

    def valid(self, dataloader): 
        self.reinitialize_metrics()         
        self.model.eval()
        
        with torch.no_grad():
            epoch_start_time = time.time()
            for step, data_packet in enumerate(dataloader):
                if step == 0:
                    first_batch_end_time = time.time()
                ### forward ###
                embeddings_map, helper_out_map, labels_d = \
                    self.forward(data_packet)
                self.after_forward_batch(embeddings_map, labels_d)
                
            # time measurement
            epoch_end_time = time.time()
            self.val_time_list.append(
                epoch_end_time - epoch_start_time)
            self.val_first_batch_time_list.append(
                first_batch_end_time - epoch_start_time)
            #
            
    def train_method(self, embedding_dict, labels_device):   
        out_f = forward_fusion(self.model[KEY_FUSION], embedding_dict)
        loss = self.criterion(out_f, labels_device)
        self.optimizer_map[KEY_FUSION].zero_grad()
        self.optimizer_map[KEY_ENCODERS].zero_grad()
        loss.backward()
        self.optimizer_map[KEY_FUSION].step()
        self.optimizer_map[KEY_ENCODERS].step()
 

    def train_epoch(self, dataloader): 
        
        self.reinitialize_metrics()        
        self.model.train() 
    
        epoch_start_time = time.time()
        for step, data_packet in enumerate(dataloader):
            if step == 0:
                first_batch_end_time = time.time()

            ### forward ###
            embedding_dict, helper_out_dict, labels_device = \
                self.forward(data_packet)
            self.after_forward_batch(embedding_dict, labels_device)
            
            ### backward ###
            # train each modality alternatively
            self.train_method(embedding_dict, labels_device)
                
            # backward helper, we don't update the backbone, just update the helper
            self.optimizer_map[KEY_HELPERS].zero_grad()
            for modality_name in self.modality_name_list:
                loss = self.criterion(helper_out_dict[modality_name], labels_device)
                loss.backward()
            self.optimizer_map[KEY_HELPERS].step()

            # time measurement
            epoch_end_time = time.time()
            self.train_time_list.append(
                epoch_end_time - epoch_start_time)
            self.train_first_batch_time_list.append(
                first_batch_end_time - epoch_start_time)
            #
    
        for sch in self.scheduler_map.values():
            sch.step()

    def init_best_model_metric(self):
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.best_epoch = 0
        
    def update_best_model(self):
        val_acc = self.val_m_map["f"].get_acc()
        if val_acc > self.best_val_acc:
            self.best_val_acc = self.val_m_map["f"].get_acc()
            if self.args.run_test:
                self.best_test_acc = self.test_m_map["f"].get_acc()
            self.best_epoch = self.epoch
            
            # save the best model
            torch.save({
                "model": self.model.state_dict(),
                "args": self.args,
                "best_val_acc": self.best_val_acc,
                "best_test_acc": self.best_test_acc,
                "epoch": self.epoch
            }, os.path.join(self.save_path, 'best_model.pth'))
        
    def print_best_model(self):
        print("Best Val Acc: {:.3f}".format(
            self.best_val_acc))
        if self.args.run_test:
            print("Best Test Acc: {:.3f}".format(
                self.best_test_acc))
        print("Best Epoch: {}".format(
            self.best_epoch))
    
    def need_run_test(self):
        if self.args.run_test:
            val_acc = self.val_m_map["f"].get_acc()
            if val_acc > self.best_val_acc:
                return True
        return False
        
    # whole training and validation process
    def train_validate(self):
        try:        
            train_validate_start_time = datetime.now()
            print("Training and Validation Start: {}".format(
                train_validate_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]))
                          
            self.init_best_model_metric()
            
            end_time = time.time()
            for epoch in range(self.args.epochs):
                self.epoch = epoch
                start_time = time.time()
                print("\n", "#" * 20, "Epoch: ", epoch, "#" * 20)
                print_separator("Training")
                
                train_dataloader = self.get_trainloader(epoch)
                
                self.before_train_epoch()
                self.train_epoch(train_dataloader)
                self.print_metrics("train")
                self.after_train_epoch()
                
                train_time = time.time()
                # validation
                
                print_separator("Validation")
                
                val_dataloader = self.get_valloader(epoch)
                self.before_valid()
                self.valid(val_dataloader)
                self.print_metrics("val")
                self.after_valid()
                
                val_time = time.time()
                
                # testing
                if self.need_run_test():
                    print_separator("Testing")
                    test_dataloader = self.get_testloader(epoch)
                    self.before_test()
                    self.valid(test_dataloader)
                    self.print_metrics("test")
                    self.after_test()
                    
                test_time = time.time()
                    

                end_time = time.time()
            
                self.update_best_model()
                
                self.train_val_epoch_time_list.append(
                    end_time - start_time)
                
                print_separator("Epoch Summary")
                
                self.print_best_model()
                
                print("Time      : {:.3f}".format(
                    end_time - start_time))
                print("Train Time: {:.3f}".format(
                    train_time - start_time))
                print("Val Time  : {:.3f}".format(
                    val_time - train_time))
                if self.args.run_test:
                    print("Test Time : {:.3f}".format(
                        test_time - val_time))
                print("Remaining Time(min): {:.3f}".format(
                    (self.args.epochs - epoch - 1) * (end_time - start_time)/60))
                    
                # self.print_best_model()
                
                sys.stdout.flush()
            
            print_separator("Final Summary")
            
            self.print_best_model()
            self.after_summary()
            
            print("Time INOF:")
                
            train_validate_end_time = datetime.now()
            print("Training and Validation Start: {}".format(
                train_validate_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]))
            print("Training and Validation End: {}".format(
                train_validate_end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]))

            # self.print_best_model()
            self.release()
        except Exception as e:
            print(f"Exception: {e}")
            self.release()
            raise e

    def release(self):
        if self.tsb_writer is not None:
            self.tsb_writer.close()
        if hasattr(self, "train_loader_pool"):
            self.train_loader_pool = None
        if hasattr(self, "val_loader_pool"):
            self.val_loader_pool = None
        if hasattr(self, "test_loader_pool"):
            self.test_loader_pool = None
        
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
            
def print_separator(msg):
    print("-" * 20, msg, "-" * 20)
    

def print_loss_and_acc(epoch, m_dict: dict, data_len, writer=None, prefix_name="train"):
    loss_key_val = {}
    acc_key_val = {}
    acc_prob_key_val = {}
    
    def update_metrics(metric:performanceMetric, prefix):
        if metric is not None:
            loss = metric.loss / data_len
            acc = metric.get_acc()
            loss_key_val[f"{prefix}"], acc_key_val[f"{prefix}"] = loss, acc
            acc_prob = metric.comput_class_avg_prob()
            acc_prob_key_val[f"{prefix}"] = acc_prob

    metric_name = list(m_dict.keys())
    
    for name in metric_name:
        update_metrics(m_dict[name], name)
   
    table = PrettyTable()
    table.field_names = ["Metric", "Loss", "Accuracy", "Pred Confidence"]

    for name in metric_name:
        table.add_row([name, 
                       f"{loss_key_val[name]:.3f}", 
                       f"{acc_key_val[name]:.3f}",
                       f"{acc_prob_key_val[name]:.3f}"
                       ])

    print(table)

    if writer is not None:
        writer.add_scalars("{}/loss".format(prefix_name), 
            {key: loss_key_val[key] for key in metric_name}, epoch)
        
        writer.add_scalars("{}/acc".format(prefix_name), 
            {key: acc_key_val[key] for key in metric_name}, epoch)

        writer.add_scalars("{}/acc_prob".format(prefix_name),
            {key: acc_prob_key_val[key] for key in metric_name}, epoch)