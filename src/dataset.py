"""PyTorch Dataset class for Titan SAR terrain patches."""

import json
from pathlib import Path

import numpy as np

from .utils import PROCESSED_DIR, SPLITS_DIR, NUM_CLASSES, GLOBAL_MEAN, GLOBAL_STD

# Conditional torch import — not needed for CPU-only notebooks
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  # stub so the class definition doesn't error


class TitanSARDataset(Dataset):
    """
    Reads pre-tiled SAR patches and label patches from disk.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.
    split_file : str or Path
        Path to split JSON (default: data/splits/split_v1.json).
    sar_dir : str or Path
        Directory containing SAR tile .npy files.
    label_dir : str or Path
        Directory containing label tile .npy files.
    transform : callable, optional
        Albumentations or custom transform applied to (image, mask) pairs.
    use_nldsar : bool
        If True, load from nldsar_tiles/ instead of sar_tiles/.
    global_normalize : bool
        If True (default), apply global mean/std normalization after loading.
    """

    def __init__(
        self,
        split="train",
        split_file=None,
        sar_dir=None,
        label_dir=None,
        transform=None,
        use_nldsar=False,
        global_normalize=True,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TitanSARDataset")

        self.split = split
        self.transform = transform
        self.global_normalize = global_normalize

        split_file = Path(split_file or SPLITS_DIR / "split_v1.json")
        with open(split_file) as f:
            split_map = json.load(f)

        # tile_id → split assignment
        self.tile_ids = [tid for tid, s in split_map.items() if s == split]
        self.tile_ids.sort()

        sar_subdir = "nldsar_tiles" if use_nldsar else "sar_tiles"
        self.sar_dir = Path(sar_dir or PROCESSED_DIR / sar_subdir)
        self.label_dir = Path(label_dir or PROCESSED_DIR / "label_tiles")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tid = self.tile_ids[idx]
        sar = np.load(self.sar_dir / f"{tid}.npy").astype(np.float32)
        label = np.load(self.label_dir / f"{tid}.npy").astype(np.int64)

        if self.global_normalize:
            sar = (sar - GLOBAL_MEAN) / GLOBAL_STD

        if self.transform is not None:
            transformed = self.transform(image=sar, mask=label)
            sar = transformed["image"]
            label = transformed["mask"]

        # Convert to tensors: SAR → (1, H, W), label → (H, W)
        if isinstance(sar, np.ndarray):
            sar = torch.from_numpy(sar)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        if sar.ndim == 2:
            sar = sar.unsqueeze(0)  # add channel dim

        return sar, label, tid
