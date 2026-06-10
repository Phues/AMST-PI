# This file is the implementation of the dataset class used in this project.
# The very basic version is from the open-source code of "MLA, OGM-GE, PMR"

from .data_path_config_alvis import get_data_path_config

import os
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio.transforms as T

from PIL import Image


def printDebugInfo(*args):
    # print("[DEBUG] [%.3f]"%(time.time()), *args)
    pass

DATASET_LIST = ['CREMAD', 'AVE', 'MVSA', 'IEMOCAP', 'IEMOCAP3', 'URFUNNY', 'TCGA']
TVA_SET_LIST = ["URFUNNY", "IEMOCAP3"]
AV_SET_LIST  = ["CREMAD", "AVE", 'IEMOCAP', 'TCGA']
TV_SET_LIST  = ["MVSA"]

# datasets have 2 modalities, because some methods only support 2 modalities
M2DATASET_LIST = ["MVSA", "CREMAD", "AVE", "IEMOCAP", "TCGA"]
# datasets have 3 modalities
M3DATASET_LIST = ["IEMOCAP3", "URFUNNY"]

# TCGA uses the 'v' slot for CLAM patches and the 'a' slot for multi-omics.
DATASET_HAS_AUDIO_LIST  = ['AVE', 'CREMAD', 'IEMOCAP', 'IEMOCAP3', 'URFUNNY', 'TCGA']
DATASET_HAS_VISUAL_LIST = ['AVE', 'CREMAD', 'MVSA', 'IEMOCAP', 'IEMOCAP3', 'URFUNNY', 'TCGA']
DATASET_HAS_TEXT_LIST   = ['MVSA', 'IEMOCAP3', 'URFUNNY']


def get_num_classes(dataset):
    dataset_classes_map = {
        'MVSA': 3,
        'CREMAD': 6,
        'AVE': 28,
        'IEMOCAP': 5,
        'IEMOCAP3': 5,
        'URFUNNY': 2,
        'TCGA': 5,   # cancer subtypes: LumA, LumB, Her2, Basal, Normal
    }
    if dataset not in dataset_classes_map:
        raise NotImplementedError('Incorrect dataset name {}'.format(dataset))
    if dataset_classes_map[dataset] is None:
        raise NotImplementedError('Dataset {} not implemented yet'.format(dataset)) 
    return dataset_classes_map[dataset]

def build_train_val_test_datasets(args):
    from .tcga_dummy_dataset import TCGADataset   # local import avoids circular deps

    dataset_classes = {
        'MVSA':    TVDataset,
        'CREMAD':  AVDataset,
        'AVE':     AVDataset,
        'IEMOCAP': AVDataset,
        'IEMOCAP3': TVADataset,
        'URFUNNY': TVADataset,
        'TCGA':    TCGADataset,
    }

    if args.dataset not in dataset_classes:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    DatasetClass = dataset_classes[args.dataset]
    train_dataset = DatasetClass(args, mode='train')
    val_dataset   = DatasetClass(args, mode='val')
    test_dataset  = DatasetClass(args, mode='test')

    return train_dataset, val_dataset, test_dataset

sep_map = {
            "AVE": ".mp4",
            "CREMAD": ".flv",
            "IEMOCAP": ",",
            "IEMOCAP3": ",",
            "URFUNNY": ",",
            "MVSA": ".jpg ", # becareful there is a space 
        }

class DatasetBase(Dataset):
    """
    Base class for all datasets.
    This class is used to load the dataset and preprocess the data.
    It is also used to load the dataset and preprocess the data.
    The dataset is loaded from the path specified in the config file.
    """
    def __init__(self, args, mode='train', pick_num=3):
        self.mode = mode
        self.args = args
        self.pick_num = pick_num

        self._init_paths(args, mode)
        self._init_classes()
        self._init_transforms()
        self.data = []
        

    def _init_paths(self, args, mode):
        dataset_cfg = get_data_path_config(args, mode)
        printDebugInfo(os.path.basename(__file__), 
                       " -  Dataset {} {}".format(args.dataset, mode))
        printDebugInfo(os.path.basename(__file__), 
                       " -  Dataset config:\n", dataset_cfg.str)
        
        self.data_root = dataset_cfg.data_root
        if hasattr(dataset_cfg, "visual_feature_path"):
            self.visual_feature_path = dataset_cfg.visual_feature_path
        if hasattr(dataset_cfg, "audio_feature_path"):
            self.audio_feature_path = dataset_cfg.audio_feature_path
        if hasattr(dataset_cfg, "text_feature_path"):
            self.text_feature_path = dataset_cfg.text_feature_path
            
        self.stat_path = dataset_cfg.stat_path
        self.train_txt = dataset_cfg.train_txt
        self.val_txt = dataset_cfg.val_txt
        self.test_txt = dataset_cfg.test_txt

    def _init_classes(self):
        # Read class names from the stat file.
        # The stat file contains the class names, one per line.
        # check if the file exists
        if not os.path.exists(self.stat_path):
            raise FileNotFoundError("File not found: ", self.stat_path)
        with open(self.stat_path, "r") as f:
            self.classes = sorted([line.strip() for line in f])

    def _init_transforms(self):
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def idx2name(self, idx: int) -> str:
        return self.data[idx]
    
    def _load_visual_features(self, visual_path, allimages):
        """
        Load visual features with support for multi-frame sampling.

        Args:
            visual_path (str): Path to the folder containing image files.
            allimages (list): List of image file names.

        Returns:
            torch.Tensor: Tensor of sampled images with shape (C, pick_num, H, W).
        """
        
        """
        The original code uses the following logic to pick images:
            seg = int(file_num / pick_num)
            image_arr = []
            for i in range(pick_num):
                tmp_index = int(seg * i)
                
        when num >=pick_num, everything is ok.
        when num < pick_num, it will pick only the 
        first image for pick_num times.
        """
        file_num = len(allimages)
        # seg = max(1, int(file_num / self.pick_num))  # new version
        seg = int(file_num / self.pick_num)  # old version, just for consistency
        
        image_arr = []
        for i in range(self.pick_num):
            # tmp_index = min(int(seg * i), file_num - 1)  # new version
            tmp_index = int(seg * i)  # old version, just for consistency
            
            with Image.open(os.path.join(visual_path, allimages[tmp_index])) as img:
                image = self.transform(img.convert('RGB')).unsqueeze(1).float()
            image_arr.append(image)
        return torch.cat(image_arr, 1)

class AVDataset(DatasetBase):
    def __init__(self, args, mode='train'):
        printDebugInfo(os.path.basename(__file__), " -  AVDataset")
        super().__init__(args, mode, pick_num=3)
        self._init_data()

    def _init_data(self):
        mode_to_file = {'train': self.train_txt, 
                        'val': self.val_txt, 
                        'test': self.test_txt}
          
        self.data = []
        self.data2class = {}
        self.audio_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.sid_all_imgs_map = {}
        
        csv_file = mode_to_file[self.mode]
        sep = sep_map[self.args.dataset]
        with open(csv_file, "r") as f:
            for line in f:
                item = [i.strip() for i in line.strip().split(sep)]
                sid = item[0]   # sample id
                label = item[1] # label
                
                # pre-check if the audio and visual files exist
                audio_path = os.path.join(self.audio_feature_path, sid + '.npy')
                visual_path = os.path.join(self.visual_feature_path, sid)
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.data.append(sid)
                    self.data2class[sid] = label
                    self.audio_sid_path_map[sid] = audio_path
                    self.visual_sid_path_map[sid] = visual_path
                    # in old version, they did't sort, so every time 
                    # the order of images is guaranteed to be the same.
                    self.sid_all_imgs_map[sid] = \
                        os.listdir(visual_path) # old version
                else:
                    raise FileNotFoundError("File not found: ", 
                        audio_path if not os.path.exists(audio_path) else visual_path)

    def __getitem__(self, idx: int):
        sid = self.data[idx]
        # Audio
        audio_path = self.audio_sid_path_map[sid]
        audio_feature = torch.from_numpy(np.load(audio_path))

        # Visual
        visual_path = self.visual_sid_path_map[sid]
        allimages = self.sid_all_imgs_map[sid]
        image_n = self._load_visual_features(visual_path, allimages)

        label = self.classes.index(self.data2class[sid])
        return audio_feature, image_n, label, sid

class TVDataset(DatasetBase):
    
    def __init__(self, args, mode='train'):
        printDebugInfo(os.path.basename(__file__), " -  TVDataset")
        super().__init__(args, mode, pick_num=1)
        self._init_data()

    def _init_data(self):
        mode_to_file = {'train': self.train_txt, 
                        'val': self.val_txt, 
                        'test': self.test_txt}
    
        csv_file = mode_to_file[self.mode]
        
        self.data = []
        self.data2class = {}
        self.token_sid_path_map = {}
        self.pm_sid_path_map = {}
        self.visual_sid_path_map = {}
    
        sep = sep_map[self.args.dataset]
        with open(csv_file, "r") as f:
            for line in f:
                item = line.strip().split(sep)
                if len(item) < 2:
                    continue
                sid, label = item[0], item[1]

                token_path = os.path.join(
                    self.text_feature_path, sid + '_token.pt')
                pm_path = os.path.join(self.text_feature_path, sid + '_pm.pt')
                visual_path = os.path.join(self.visual_feature_path, sid + ".jpg")
                
                if os.path.exists(token_path) and \
                    os.path.exists(pm_path) and \
                    os.path.exists(visual_path):
                    
                    self.data.append(sid)
                    self.data2class[sid] = label
                    self.token_sid_path_map[sid] = token_path
                    self.pm_sid_path_map[sid] = pm_path
                    self.visual_sid_path_map[sid] = visual_path
                else:
                    raise FileNotFoundError("File not found: ", 
                        token_path if not os.path.exists(token_path) else visual_path)

    def __getitem__(self, idx: int):
        sid = self.data[idx]
        # Text
        token_path = self.token_sid_path_map[sid]
        pm_path = self.pm_sid_path_map[sid]
        tokenizer = torch.load(token_path)
        padding_mask = torch.load(pm_path)

        # Visual
        """
        Obviously, now it only supports MVSA dataset.
        which has only one image.
        """
        image_path = self.visual_sid_path_map[sid]
        with Image.open(image_path) as img:
            image = self.transform(img.convert('RGB')).unsqueeze(1).float()
        image_n = image
    
        label = self.classes.index(self.data2class[sid])
        
        tokenizer = tokenizer.clone().detach().squeeze(0)
        padding_mask = padding_mask.clone().detach().squeeze(0)
        return tokenizer, padding_mask, image_n, label, sid

class TVADataset(DatasetBase):
    
    def __init__(self, args, mode='train'):
        printDebugInfo(os.path.basename(__file__), " -  TVADataset")
        super().__init__(args, mode, pick_num=3)
        self._init_data()

    def _init_data(self):
        mode_to_file = {'train': self.train_txt, 
                        'val': self.val_txt, 
                        'test': self.test_txt}
        csv_file = mode_to_file[self.mode]
        self.data = []
        self.data2class = {}
        self.token_sid_path_map = {}
        self.pm_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.sid_all_imgs_map = {}
        self.audio_sid_path_map = {}
        
        sep = sep_map[self.args.dataset]
        with open(csv_file, "r") as f:
            for line in f:
                item = [i.strip() for i in line.strip().split(sep)]
                sid = item[0]
                label = item[1]
                audio_path = os.path.join(self.audio_feature_path, sid + '.npy')
                visual_path = os.path.join(self.visual_feature_path, sid)
                token_path = os.path.join(self.text_feature_path, sid + '_token.pt')
                pm_path = os.path.join(self.text_feature_path, sid + '_pm.pt')
                if all([
                    os.path.exists(audio_path),
                    os.path.exists(visual_path),
                    os.path.exists(token_path),
                    os.path.exists(pm_path)
                ]):
                    self.data.append(sid)
                    self.data2class[sid] = label
                    self.token_sid_path_map[sid] = token_path
                    self.pm_sid_path_map[sid] = pm_path
                    self.visual_sid_path_map[sid] = visual_path
                    self.audio_sid_path_map[sid] = audio_path
                    self.sid_all_imgs_map[sid] = \
                        os.listdir(visual_path)
                else:
                    raise FileNotFoundError("File not found: ",
                        audio_path if not os.path.exists(audio_path) else visual_path,
                        token_path if not os.path.exists(token_path) else pm_path)

    def __getitem__(self, idx: int):
        sid = self.data[idx]
        # Text
        token_path = self.token_sid_path_map[sid]
        pm_path = self.pm_sid_path_map[sid]
        tokenizer = torch.load(token_path)
        padding_mask = torch.load(pm_path)

        # Audio
        audio_path = self.audio_sid_path_map[sid]
        audio_feature = torch.from_numpy(np.load(audio_path))

        # Visual
        visual_path = self.visual_sid_path_map[sid]
        allimages = self.sid_all_imgs_map[sid]
        image_n = self._load_visual_features(visual_path, allimages)

        label = self.classes.index(self.data2class[sid])
        tokenizer = tokenizer.squeeze(0)
        padding_mask = padding_mask.squeeze(0)
        
        # size of tokenizer size:  torch.Size([128])
        # size of padding_mask size:  torch.Size([128])
    
        return tokenizer, padding_mask, image_n, audio_feature, label, sid