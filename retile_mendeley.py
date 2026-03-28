"""Re-tile label map using improved Mendeley labels (Lopes 2020 shapefiles).

Reads label_map_mendeley.tif and re-generates label_tiles/*.npy + tile_metadata.csv.
Keeps the same tile grid and SAR tiles — only labels change.
Also regenerates the geographic split since class fractions change.
"""
import numpy as np
import pandas as pd
import json
import rasterio
from rasterio.windows import Window
from pathlib import Path
from collections import Counter

# Paths
PROCESSED_DIR = Path("data/processed")
SAR_PATH = Path("data/raw/Titan_SAR_HiSAR_Global_Mosaic_351m.tif")
LABEL_PATH = PROCESSED_DIR / "label_map_mendeley.tif"
LABEL_TILES_DIR = PROCESSED_DIR / "label_tiles"
SAR_TILES_DIR = PROCESSED_DIR / "sar_tiles"
SPLITS_DIR = Path("data/splits")

TILE_SIZE = 256
NODATA_THRESHOLD = 0.5
NUM_CLASSES = 6
BLOCK_DEG = 10
TITAN_DEG_M = 44934  # ~1 degree in metres on Titan

print("=== Re-tiling with Mendeley labels ===")

# Back up old labels
import shutil
old_backup = PROCESSED_DIR / "label_tiles_old"
if not old_backup.exists() and LABEL_TILES_DIR.exists():
    print("Backing up old label tiles...")
    shutil.copytree(LABEL_TILES_DIR, old_backup)
    print(f"  Backed up to {old_backup}")

# Open both rasters
with rasterio.open(SAR_PATH) as sar_src, rasterio.open(LABEL_PATH) as lbl_src:
    full_height = sar_src.height
    full_width = sar_src.width
    n_rows = full_height // TILE_SIZE
    n_cols = full_width // TILE_SIZE
    transform = sar_src.transform

    print(f"SAR: {full_width}x{full_height}, Label: {lbl_src.width}x{lbl_src.height}")
    print(f"Tile grid: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} potential tiles")

    # Check dimensions match
    assert lbl_src.height == full_height and lbl_src.width == full_width, \
        f"Label map dimensions {lbl_src.width}x{lbl_src.height} don't match SAR {full_width}x{full_height}"

    records = []
    kept = 0
    discarded = 0

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            tile_id = f"tile_{row_idx:04d}_{col_idx:04d}"
            window = Window(
                col_off=col_idx * TILE_SIZE,
                row_off=row_idx * TILE_SIZE,
                width=TILE_SIZE,
                height=TILE_SIZE,
            )

            # Check if SAR tile exists (use same filtering as original)
            sar_tile_path = SAR_TILES_DIR / f"{tile_id}.npy"
            if not sar_tile_path.exists():
                discarded += 1
                continue

            # Read label tile
            lbl_tile = lbl_src.read(1, window=window).astype(np.uint8)

            # Compute nodata fraction
            nodata_mask = (lbl_tile == 255)
            nodata_frac = nodata_mask.sum() / nodata_mask.size

            # Save label tile (overwrite old one)
            np.save(LABEL_TILES_DIR / f"{tile_id}.npy", lbl_tile)

            # Compute class fractions
            valid_pixels = lbl_tile[~nodata_mask]
            n_valid = len(valid_pixels)
            class_counts = Counter(valid_pixels.tolist())

            tile_bounds = rasterio.windows.bounds(window, transform)
            record = {
                'tile_id': tile_id,
                'row': row_idx,
                'col': col_idx,
                'lon_min': tile_bounds[0],
                'lat_min': tile_bounds[1],
                'lon_max': tile_bounds[2],
                'lat_max': tile_bounds[3],
                'nodata_frac': float(nodata_frac),
            }
            for cls_idx in range(NUM_CLASSES):
                record[f'class_{cls_idx}_frac'] = class_counts.get(cls_idx, 0) / n_valid if n_valid > 0 else 0.0

            records.append(record)
            kept += 1

        if (row_idx + 1) % 10 == 0:
            print(f"  Row {row_idx+1}/{n_rows} ({kept} tiles kept, {discarded} skipped)")

print(f"\nTotal: {kept} tiles kept, {discarded} skipped")

# Save metadata
tile_df = pd.DataFrame(records)
tile_df.to_csv(PROCESSED_DIR / 'tile_metadata.csv', index=False)
print(f"Saved tile_metadata.csv ({len(tile_df)} rows)")

# Class distribution
print("\nClass distribution (Mendeley labels):")
class_names = ["plains", "dunes", "hummocky", "lakes_seas", "labyrinth", "craters"]
for i, name in enumerate(class_names):
    mean_frac = tile_df[f'class_{i}_frac'].mean()
    print(f"  {name:>12s}: {mean_frac:.3f}")

# Regenerate geographic split (same algorithm, seed=42)
print("\n=== Regenerating geographic split ===")
block_size_m = BLOCK_DEG * TITAN_DEG_M
tile_df['center_lon'] = (tile_df['lon_min'] + tile_df['lon_max']) / 2
tile_df['center_lat'] = (tile_df['lat_min'] + tile_df['lat_max']) / 2
tile_df['block_lon'] = (tile_df['center_lon'] // block_size_m).astype(int)
tile_df['block_lat'] = (tile_df['center_lat'] // block_size_m).astype(int)
tile_df['block_id'] = tile_df['block_lon'].astype(str) + '_' + tile_df['block_lat'].astype(str)

blocks = tile_df['block_id'].unique()
rng = np.random.RandomState(42)
rng.shuffle(blocks)

n_train = int(len(blocks) * 0.70)
n_val = int(len(blocks) * 0.15)
train_blocks = set(blocks[:n_train])
val_blocks = set(blocks[n_train:n_train + n_val])
test_blocks = set(blocks[n_train + n_val:])

split_map = {}
for _, row in tile_df.iterrows():
    bid = row['block_id']
    if bid in train_blocks:
        split_map[row['tile_id']] = 'train'
    elif bid in val_blocks:
        split_map[row['tile_id']] = 'val'
    else:
        split_map[row['tile_id']] = 'test'

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
# Save as v2 (Mendeley labels)
with open(SPLITS_DIR / 'split_v2_mendeley.json', 'w') as f:
    json.dump(split_map, f)

# Also overwrite v1 so existing code picks it up
with open(SPLITS_DIR / 'split_v1.json', 'w') as f:
    json.dump(split_map, f)

train_count = sum(1 for v in split_map.values() if v == 'train')
val_count = sum(1 for v in split_map.values() if v == 'val')
test_count = sum(1 for v in split_map.values() if v == 'test')
print(f"Split: {train_count} train, {val_count} val, {test_count} test")

# Recompute class weights
print("\n=== Computing class weights ===")
all_labels = []
for tid in tile_df['tile_id']:
    lbl = np.load(LABEL_TILES_DIR / f"{tid}.npy")
    valid = lbl[lbl != 255]
    all_labels.append(valid)

all_labels = np.concatenate(all_labels)
counts = np.bincount(all_labels, minlength=NUM_CLASSES)
freqs = counts / counts.sum()
# Inverse frequency weights, capped
weights = 1.0 / (freqs + 1e-6)
weights = weights / weights.min()  # normalize so min weight = 1
weights = np.clip(weights, 1.0, 50.0)  # cap at 50x

print("Class weights:")
for i, name in enumerate(class_names):
    print(f"  {name:>12s}: count={counts[i]:>10d}  freq={freqs[i]:.4f}  weight={weights[i]:.2f}")

with open(PROCESSED_DIR / 'class_weights.json', 'w') as f:
    json.dump({
        'weights_list': weights.tolist(),
        'counts': counts.tolist(),
        'frequencies': freqs.tolist(),
    }, f, indent=2)

print("\n=== Done! Ready for training. ===")
