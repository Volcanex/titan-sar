"""Execute Notebook 02 — Preprocessing and Tiling."""
import sys, os, json
os.chdir('/home/gabriel/titan-sar')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import features as rio_features
import fiona
from shapely.geometry import shape
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from src.utils import (
    RAW_DIR, PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR,
    TERRAIN_CLASSES, CLASS_COLORS, NUM_CLASSES,
    write_geotiff, get_logger,
)

log = get_logger('02_preprocessing')

SAR_TILES_DIR = PROCESSED_DIR / 'sar_tiles'
LABEL_TILES_DIR = PROCESSED_DIR / 'label_tiles'
for d in [SAR_TILES_DIR, LABEL_TILES_DIR, SPLITS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 1. SAR Mosaic ────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading SAR Mosaic")
print("=" * 60)

sar_path = RAW_DIR / 'Titan_SAR_HiSAR_Global_Mosaic_351m.tif'
with rasterio.open(sar_path) as src:
    sar_profile = src.profile.copy()
    print(f'CRS: {src.crs}')
    print(f'Shape: {src.height} x {src.width}')
    print(f'Resolution: {src.res}')
    print(f'Dtype: {src.dtypes}, NoData: {src.nodata}')

# ── 2. Rasterise Geomorphological Map ────────────────────────────────
print("\n" + "=" * 60)
print("2. Rasterising Geomorphological Map")
print("=" * 60)

gdb_path = str(RAW_DIR / 'geomorphology' / 'titan_6unit_geomap' /
               'TITAN_2019-11_global_geomap_6unit' / 'Titan_Geodatabase_2019-11.gdb')

# Map the geodatabase class names to our indices
CLASS_MAP = {
    'Plains': 0,
    'Dunes': 1,
    'Mountains': 2,      # = hummocky/mountainous
    'Basins': 3,          # = lakes/seas
    'Labyrinth': 4,
    'Craters': 5,
}

LABEL_RASTER_PATH = PROCESSED_DIR / 'label_map_aligned.tif'

if LABEL_RASTER_PATH.exists():
    log.info(f"Label raster already exists: {LABEL_RASTER_PATH}")
else:
    # Read geologic units from GDB
    shapes_list = []
    with fiona.open(gdb_path, layer='TITAN_GeologicUnits') as src:
        gdb_crs = src.crs
        print(f'GDB CRS: {gdb_crs}')
        for feat in src:
            terrain_name = feat['properties']['Meta_Terra']
            cls_idx = CLASS_MAP.get(terrain_name)
            if cls_idx is not None:
                geom = shape(feat['geometry'])
                shapes_list.append((geom, cls_idx))
                print(f'  {terrain_name} -> class {cls_idx} ({TERRAIN_CLASSES[cls_idx]})')
            else:
                log.warning(f'Unmapped class: {terrain_name}')

    print(f'\nTotal polygons: {len(shapes_list)}')

    # Rasterise to match SAR mosaic grid
    with rasterio.open(sar_path) as sar_src:
        out_shape = (sar_src.height, sar_src.width)
        out_transform = sar_src.transform

        log.info(f'Rasterising {len(shapes_list)} polygons to {out_shape}...')
        label_raster = rio_features.rasterize(
            shapes_list,
            out_shape=out_shape,
            transform=out_transform,
            fill=255,  # nodata
            dtype='uint8',
        )

        # Save
        label_profile = sar_src.profile.copy()
        label_profile.update(dtype='uint8', count=1, nodata=255)
        write_geotiff(LABEL_RASTER_PATH, label_raster, label_profile, dtype='uint8')

    print(f'Label raster saved: {LABEL_RASTER_PATH}')
    unique, counts = np.unique(label_raster, return_counts=True)
    total = label_raster.size
    for u, c in zip(unique, counts):
        name = TERRAIN_CLASSES.get(u, 'nodata' if u == 255 else f'unknown_{u}')
        print(f'  Class {u} ({name:>12s}): {c:>12,} pixels ({100*c/total:.1f}%)')

# ── 3. Tile into 256x256 patches ─────────────────────────────────────
print("\n" + "=" * 60)
print("3. Tiling into 256x256 patches")
print("=" * 60)

TILE_SIZE = 256
NODATA_THRESHOLD = 0.5

tile_records = []

with rasterio.open(sar_path) as sar_src, rasterio.open(LABEL_RASTER_PATH) as lbl_src:
    n_rows = sar_src.height // TILE_SIZE
    n_cols = sar_src.width // TILE_SIZE
    total_tiles = n_rows * n_cols

    log.info(f'Tiling: {n_rows} rows x {n_cols} cols = {total_tiles} potential tiles')

    kept = 0
    discarded = 0

    for row_idx in tqdm(range(n_rows), desc='Tiling rows'):
        for col_idx in range(n_cols):
            window = Window(
                col_off=col_idx * TILE_SIZE,
                row_off=row_idx * TILE_SIZE,
                width=TILE_SIZE,
                height=TILE_SIZE,
            )

            sar_tile = sar_src.read(1, window=window).astype(np.float32)
            lbl_tile = lbl_src.read(1, window=window).astype(np.uint8)

            # Nodata: SAR=0 (nodata value), label=255
            sar_nodata_val = 0.0
            nodata_mask = (sar_tile == sar_nodata_val) | (lbl_tile == 255)
            nodata_frac = nodata_mask.sum() / nodata_mask.size

            if nodata_frac > NODATA_THRESHOLD:
                discarded += 1
                continue

            valid = sar_tile[~nodata_mask]
            if len(valid) == 0:
                discarded += 1
                continue

            # Normalise SAR to [0, 1] using per-tile 2nd-98th percentile
            p2, p98 = np.percentile(valid, [2, 98])
            sar_tile_norm = np.clip(sar_tile, p2, p98)
            if p98 > p2:
                sar_tile_norm = (sar_tile_norm - p2) / (p98 - p2)
            else:
                sar_tile_norm = np.zeros_like(sar_tile_norm)
            sar_tile_norm[nodata_mask] = 0

            tile_id = f'tile_{row_idx:04d}_{col_idx:04d}'

            np.save(SAR_TILES_DIR / f'{tile_id}.npy', sar_tile_norm)
            np.save(LABEL_TILES_DIR / f'{tile_id}.npy', lbl_tile)

            # Metadata
            tile_bounds = rasterio.windows.bounds(window, sar_src.transform)
            valid_lbl = lbl_tile[~nodata_mask]
            class_counts = Counter(valid_lbl.flatten().tolist())
            n_valid = int((~nodata_mask).sum())

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
                record[f'class_{cls_idx}_frac'] = class_counts.get(cls_idx, 0) / n_valid if n_valid > 0 else 0

            tile_records.append(record)
            kept += 1

    log.info(f'Tiling complete: {kept} kept, {discarded} discarded')

# Save metadata
tile_df = pd.DataFrame(tile_records)
tile_df.to_csv(PROCESSED_DIR / 'tile_metadata.csv', index=False)
print(f'Tile metadata saved: {len(tile_df)} tiles')

# ── 4. Geographic Train/Val/Test Split ───────────────────────────────
print("\n" + "=" * 60)
print("4. Creating Geographic Train/Val/Test Split")
print("=" * 60)

BLOCK_SIZE_DEG = 10.0

# Compute block coordinates — these are in projected metres, convert to approx degrees
# Titan radius ~2575 km, so 1 degree ~ 44.9 km ~ 44,934 m
TITAN_DEG_TO_M = 2_575_000 * np.pi / 180  # ~44,934 m per degree
block_size_m = BLOCK_SIZE_DEG * TITAN_DEG_TO_M

tile_df['center_lon'] = (tile_df['lon_min'] + tile_df['lon_max']) / 2
tile_df['center_lat'] = (tile_df['lat_min'] + tile_df['lat_max']) / 2
tile_df['block_lon'] = (tile_df['center_lon'] // block_size_m).astype(int)
tile_df['block_lat'] = (tile_df['center_lat'] // block_size_m).astype(int)
tile_df['block_id'] = tile_df['block_lon'].astype(str) + '_' + tile_df['block_lat'].astype(str)

rng = np.random.RandomState(42)
unique_blocks = sorted(tile_df['block_id'].unique())
rng.shuffle(unique_blocks)

n_blocks = len(unique_blocks)
n_train = int(0.70 * n_blocks)
n_val = int(0.15 * n_blocks)

block_split = {}
for i, block in enumerate(unique_blocks):
    if i < n_train:
        block_split[block] = 'train'
    elif i < n_train + n_val:
        block_split[block] = 'val'
    else:
        block_split[block] = 'test'

tile_df['split'] = tile_df['block_id'].map(block_split)

print('Split distribution:')
for split_name in ['train', 'val', 'test']:
    subset = tile_df[tile_df['split'] == split_name]
    n = len(subset)
    pct = 100 * n / len(tile_df)
    print(f'  {split_name:5s}: {n:5d} tiles ({pct:.1f}%)')

# Save
split_map = dict(zip(tile_df['tile_id'], tile_df['split']))
with open(SPLITS_DIR / 'split_v1.json', 'w') as f:
    json.dump(split_map, f, indent=2)

# Also save updated metadata
tile_df.to_csv(PROCESSED_DIR / 'tile_metadata.csv', index=False)

print(f'\nSplit saved to {SPLITS_DIR / "split_v1.json"}')
print(f'Total tiles: {len(tile_df)}')

# ── 5. Quick Verification ────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Verification")
print("=" * 60)

# Count tiles per class (by majority class)
for cls in range(NUM_CLASSES):
    n_tiles = (tile_df[f'class_{cls}_frac'] > 0.5).sum()
    print(f'  Tiles dominated by {TERRAIN_CLASSES[cls]:>12s}: {n_tiles:>5d}')

print(f'\nTotal tiles: {len(tile_df)}')
print(f'SAR tiles dir: {len(list(SAR_TILES_DIR.glob("*.npy")))} files')
print(f'Label tiles dir: {len(list(LABEL_TILES_DIR.glob("*.npy")))} files')
print(f'\nReady for Notebook 03.')
