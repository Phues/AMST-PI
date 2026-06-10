# This file is used to configure the data path for different datasets.
# The data path is configured according to the dataset name.
# Why it looks a bit messy is because the historical reasons.
# We wrapped the data path in a class, and then we can use it in the code.

from types import SimpleNamespace
import os, json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"



CREMAD_ROOT = '/mimer/NOBACKUP/groups/multimodal_learning/code_refactoring/CREMA-D'
AVE_ROOT = '/mimer/NOBACKUP/groups/multimodal_learning/code_refactoring/AVE_Dataset'
MVSA_ROOT = '/home/phues/AMST-PI/MVSA_Single'
IEMOCAP_ROOT = '/mimer/NOBACKUP/groups/multimodal_learning/code_refactoring/IEMOCAP_PROCESSED' 
URFUNNY_ROOT = '/mimer/NOBACKUP/groups/multimodal_learning/code_refactoring/UR-FUNNY'


IEMOCAP_CONFIG = {
    # raw data
    "data_root": IEMOCAP_ROOT,
    "visual_feature_path": IEMOCAP_ROOT+ "/IMAGE_KEPT_2_PER_SEC",
    "audio_feature_path": IEMOCAP_ROOT + "/fbank/",
    "text_feature_path":  IEMOCAP_ROOT + "/text_token/",
    # data files
    "stat_path": PROJECT_ROOT + "data/IEMOCAP/stat_iemocap.txt",
    "train_txt": PROJECT_ROOT + "data/IEMOCAP/iemocap_train.txt",
    "test_txt":  PROJECT_ROOT+ "data/IEMOCAP/iemocap_test.txt",
    "val_txt":   PROJECT_ROOT+ "data/IEMOCAP/iemocap_valid.txt"
}

DATA_PATH_CONFIG = {
    "AVE": {
        # raw data
        "data_root": AVE_ROOT,
        "visual_feature_path": AVE_ROOT + '/IMAGE_KEPT_1_PER_SEC/',
        "audio_feature_path": AVE_ROOT + '/fbank/',
        # data files
        "stat_path": PROJECT_ROOT + "data/AVE/stat_ave.txt",
        "train_txt": PROJECT_ROOT + "data/AVE/my_train_ave.txt",
        "val_txt": PROJECT_ROOT + "data/AVE/my_val_ave.txt",
        "test_txt": PROJECT_ROOT + "data/AVE/my_test_ave.txt"
    },
    "CREMAD": {
        # raw data
        "data_root": CREMAD_ROOT,
        "visual_feature_path": CREMAD_ROOT + '/IMAGE_KEPT_1_PER_SEC/',
        "audio_feature_path": CREMAD_ROOT + '/fbank/',
        # data files
        "stat_path": PROJECT_ROOT + "data/CREMAD/stat_cre.txt",
        "train_txt": PROJECT_ROOT + "data/CREMAD/80_train_cre.txt",
        "test_txt":  PROJECT_ROOT+ "data/CREMAD/10_test_cre.txt",
        "val_txt":   PROJECT_ROOT+ "data/CREMAD/10_val_cre.txt"
    },
    # IEMOCAP3 is the same as IEMOCAP, but with text features, for compatibility
    "IEMOCAP": IEMOCAP_CONFIG,
    "IEMOCAP3": IEMOCAP_CONFIG,
    "MVSA": {
        "data_root": MVSA_ROOT,
        "visual_feature_path": MVSA_ROOT + '/visual/',
        "text_feature_path": MVSA_ROOT + '/text_token/roberta-base/',
        "stat_path": PROJECT_ROOT + "data/MVSA/stat_mvsa.txt",
        "train_txt": PROJECT_ROOT + "data/MVSA/my_train_mvsa.txt",
        "val_txt": PROJECT_ROOT + "data/MVSA/my_val_mvsa.txt",
        "test_txt": PROJECT_ROOT + "data/MVSA/my_test_mvsa.txt"
    },
    "URFUNNY": {
        "data_root": URFUNNY_ROOT,
        "visual_feature_path": URFUNNY_ROOT + '/IMAGE_KEPT_1_PER_SEC/',
        "audio_feature_path": URFUNNY_ROOT + '/fbank/',
        "text_feature_path": URFUNNY_ROOT + '/text_token/roberta-base/',
        "stat_path": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_stat.txt",
        "train_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_train.txt",
        "val_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_valid.txt",
        "test_txt": PROJECT_ROOT + "data/UR-FUNNY/ur_funny_test.txt"
    },
    # ----------------------------------------------------------------
    # TCGA: CLAM patch bags (visual slot) + multi-omics (audio slot)
    # Default root is /tmp/tcga_dummy; override with --data_path.
    # The dummy generator in tcga_dummy_dataset.py creates the files
    # automatically on first run, so no real TCGA data is needed.
    # ----------------------------------------------------------------
    "TCGA": {
        "data_root": '/tmp/tcga_dummy',
        "stat_path": '/tmp/tcga_dummy/stat_tcga.txt',
        "train_txt": '/tmp/tcga_dummy/train.txt',
        "val_txt":   '/tmp/tcga_dummy/val.txt',
        "test_txt":  '/tmp/tcga_dummy/test.txt',
    },
}

def get_data_path_config(args, mode):     
    if args.dataset not in DATA_PATH_CONFIG:
        raise ValueError("Invalid dataset: {},"\
                          " please choose from {}".format(
                              args.dataset, DATA_PATH_CONFIG.keys()))

    dataset_cfg = DATA_PATH_CONFIG[args.dataset]
    dataset_cfg = SimpleNamespace(**dataset_cfg)
    if hasattr(args, "data_path") and args.data_path != "":
        dataset_cfg.data_root = args.data_path
        print("Using data path from args: ", dataset_cfg.data_root)
        
    # special handling for Dataset if you want in experiment
    # This part is removed in this version
    
    dataset_cfg.str = json.dumps(dataset_cfg.__dict__, indent=4)
    return dataset_cfg
