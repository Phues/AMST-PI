# dataset/tcga_dummy_dataset.py
#
# Dummy TCGA dataset for testing AMST with CLAM patch bags and multi-omics.
#
# On first call with generate=True (default), this script creates fully
# synthetic data files under <root>/ that exactly match the on-disk layout
# expected by TCGADataset and the existing DatasetBase file-list convention.
#
# Generated layout
# ----------------
#   <root>/
#     patches/          one .npy file per sample  (N_patches, PATCH_DIM)
#     omics/            one .npy file per sample  (OMICS_DIM,)
#     stat_tcga.txt     one class name per line
#     train.txt         lines: "<sid>,<class>"
#     val.txt
#     test.txt
#
# Modality shapes that reach the encoders
# ----------------------------------------
#   patches :  FloatTensor (N_patches, PATCH_DIM)   e.g. (64, 1024)
#   omics   :  FloatTensor (OMICS_DIM,)             e.g. (1024,)
#
# Data packet returned by __getitem__
# ------------------------------------
#   patches, omics, label, sid
#   -- identical positional layout to AVDataset (audio, image, label, sid)
#      so prepare_input_dict's AV branch handles it with one extra key mapping.

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------
# Dataset constants  (change here to resize the dummy data)
# -----------------------------------------------------------------------
NUM_CLASSES     = 5          # e.g. BRCA subtypes: LumA LumB Her2 Basal Normal
CLASS_NAMES     = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']

N_TRAIN         = 400        # samples per split
N_VAL           = 100
N_TEST          = 100

N_PATCHES       = 64         # patches per slide (dummy: fixed bag size)
PATCH_DIM       = 1024       # CLAM ResNet-50 feature dim
OMICS_DIM       = 1024       # genomic feature vector dimension
                              # (RNA-seq + CNV + mutation concatenated & PCA'd)


# -----------------------------------------------------------------------
def generate_dummy_data(root: str, seed: int = 0) -> None:
    """
    Write synthetic .npy files and split .txt files to `root`.
    Safe to call repeatedly — skips generation if stat file exists.
    """
    stat_path = os.path.join(root, 'stat_tcga.txt')
    if os.path.exists(stat_path):
        print(f"[TCGADummy] Data already exists at {root}, skipping generation.")
        return

    print(f"[TCGADummy] Generating dummy TCGA data at {root} ...")
    rng = np.random.default_rng(seed)

    os.makedirs(os.path.join(root, 'patches'), exist_ok=True)
    os.makedirs(os.path.join(root, 'omics'),   exist_ok=True)

    # Write class stat file
    with open(stat_path, 'w') as f:
        for c in CLASS_NAMES:
            f.write(c + '\n')

    splits = {
        'train': N_TRAIN,
        'val':   N_VAL,
        'test':  N_TEST,
    }

    for split, n in splits.items():
        txt_path = os.path.join(root, f'{split}.txt')
        with open(txt_path, 'w') as txt_f:
            for i in range(n):
                sid   = f'{split}_{i:04d}'
                label = CLASS_NAMES[i % NUM_CLASSES]

                # CLAM patch bag  (N_patches, PATCH_DIM)  — unit-norm rows
                patches = rng.standard_normal((N_PATCHES, PATCH_DIM)).astype(np.float32)
                patches /= (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
                np.save(os.path.join(root, 'patches', sid + '.npy'), patches)

                # Multi-omics vector  (OMICS_DIM,)
                omics = rng.standard_normal(OMICS_DIM).astype(np.float32)
                np.save(os.path.join(root, 'omics', sid + '.npy'), omics)

                txt_f.write(f'{sid},{label}\n')

    print(f"[TCGADummy] Done. {N_TRAIN + N_VAL + N_TEST} samples written.")


# -----------------------------------------------------------------------
class TCGADataset(Dataset):
    """
    Loads the (dummy) TCGA dataset.

    __getitem__ returns:
        patches  : FloatTensor (N_patches, PATCH_DIM)
        omics    : FloatTensor (OMICS_DIM,)
        label    : int
        sid      : str

    This positional layout is intentionally identical to AVDataset so that
    prepare_input_dict can route it via the AV branch using the key mapping
        'a' -> omics,   'v' -> patches
    set in data_path_config and common.py.
    """

    def __init__(self, args, mode: str = 'train'):
        assert mode in ('train', 'val', 'test')
        self.mode = mode

        # Resolve root from args.data_path or the config default
        from .data_path_config_alvis import get_data_path_config
        cfg = get_data_path_config(args, mode)
        self.root = cfg.data_root

        # Auto-generate dummy data if needed
        generate_dummy_data(self.root)

        self.patches_dir = os.path.join(self.root, 'patches')
        self.omics_dir   = os.path.join(self.root, 'omics')

        # Load class names from stat file
        stat_path = os.path.join(self.root, 'stat_tcga.txt')
        with open(stat_path) as f:
            self.classes = sorted([l.strip() for l in f if l.strip()])

        # Load split file
        split_files = {'train': 'train.txt', 'val': 'val.txt', 'test': 'test.txt'}
        split_path = os.path.join(self.root, split_files[mode])

        self.data      = []   # list of sids
        self.data2class = {}  # sid -> class name

        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sid, label = line.split(',', 1)
                sid, label = sid.strip(), label.strip()
                patch_path = os.path.join(self.patches_dir, sid + '.npy')
                omics_path = os.path.join(self.omics_dir,   sid + '.npy')
                if os.path.exists(patch_path) and os.path.exists(omics_path):
                    self.data.append(sid)
                    self.data2class[sid] = label
                else:
                    raise FileNotFoundError(
                        f"Missing files for sample '{sid}' in {self.root}")

        print(f"[TCGADataset] {mode}: {len(self.data)} samples, "
              f"{len(self.classes)} classes")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sid = self.data[idx]

        patches = torch.from_numpy(
            np.load(os.path.join(self.patches_dir, sid + '.npy')))   # (N, D)
        omics = torch.from_numpy(
            np.load(os.path.join(self.omics_dir, sid + '.npy')))     # (OMICS_DIM,)

        label = self.classes.index(self.data2class[sid])

        return patches, omics, label, sid
