"""Augmentation pipelines for SAR terrain patches."""

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms():
    """Training augmentations: flips, rotations, noise."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for augmentations")
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    ])


def get_val_transforms():
    """Validation/test: no augmentation."""
    if not HAS_ALBUMENTATIONS:
        return None
    return A.Compose([])
