"""Regenerate global map using DeepLabV3+ pixel-level predictions on CPU."""
import numpy as np
import json
import torch
import time
import shutil
import pandas as pd
import segmentation_models_pytorch as smp
import rasterio
from pathlib import Path
from src.utils import (PROCESSED_DIR, PREDICTIONS_DIR, MODELS_DIR, RAW_DIR,
                        FIGURES_DIR, TERRAIN_CLASSES, CLASS_COLORS, NUM_CLASSES,
                        write_geotiff)

print("Loading U-Net EfficientNet-B4 (R3 best)...")
model = smp.create_model("Unet", encoder_name="efficientnet-b4",
                         encoder_weights=None, in_channels=3, classes=6)
state = torch.load(MODELS_DIR / "r3_unet_effb4_dice_best.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded")

tile_df = pd.read_csv(PROCESSED_DIR / 'tile_metadata.csv')
print(f"Tiles to process: {len(tile_df)}")

SAR_TILES_DIR = PROCESSED_DIR / 'sar_tiles'
TILE_SIZE = 256

sar_path = RAW_DIR / 'Titan_SAR_HiSAR_Global_Mosaic_351m.tif'
with rasterio.open(sar_path) as src:
    full_height = src.height
    full_width = src.width
    profile = src.profile.copy()

global_map = np.full((full_height, full_width), 255, dtype=np.uint8)

t0 = time.time()
for idx, row in tile_df.iterrows():
    tid = row['tile_id']
    r = int(row['row'])
    c = int(row['col'])

    sar = np.load(SAR_TILES_DIR / f"{tid}.npy")

    with torch.no_grad():
        x = torch.from_numpy(sar).float().unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
        pred = model(x).argmax(dim=1).squeeze().numpy().astype(np.uint8)

    r_start = r * TILE_SIZE
    c_start = c * TILE_SIZE
    global_map[r_start:r_start+TILE_SIZE, c_start:c_start+TILE_SIZE] = pred

    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        remaining = (len(tile_df) - idx - 1) / rate
        print(f"  {idx+1}/{len(tile_df)} tiles ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

profile.update(dtype='uint8', count=1, nodata=255)
output_path = PREDICTIONS_DIR / 'global_segmentation_map_dl.tif'
write_geotiff(output_path, global_map, profile, dtype='uint8')
elapsed = time.time() - t0
print(f"\nDone! {len(tile_df)} tiles in {elapsed:.0f}s")
print(f"Saved: {output_path}")

shutil.copy(output_path, PREDICTIONS_DIR / 'global_segmentation_map.tif')
print("Updated global_segmentation_map.tif")

# Also regenerate the PNG figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

class_cmap = ListedColormap([CLASS_COLORS[i] for i in range(NUM_CLASSES)])

with rasterio.open(sar_path) as sar_src:
    ds = 6
    sar_full = sar_src.read(1, out_shape=(sar_src.height // ds, sar_src.width // ds))

pred_ds = global_map[::ds, ::ds]
nodata = sar_src.nodata if sar_src.nodata is not None else 0
valid = sar_full[sar_full != nodata]
vmin, vmax = np.percentile(valid, [2, 98]) if len(valid) > 0 else (0, 1)

fig, ax = plt.subplots(figsize=(24, 12))
ax.imshow(sar_full, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
pred_masked = np.ma.masked_where((pred_ds == 255) | (sar_full == nodata), pred_ds)
ax.imshow(pred_masked, cmap=class_cmap, vmin=0, vmax=NUM_CLASSES-1, alpha=0.5, interpolation='nearest')
legend_patches = [mpatches.Patch(color=CLASS_COLORS[i], label=TERRAIN_CLASSES[i]) for i in range(NUM_CLASSES)]
ax.legend(handles=legend_patches, loc='lower left', fontsize=12, framealpha=0.9)
ax.set_title('Titan Global Terrain Classification — U-Net EfficientNet-B4 (mIoU 0.455)', fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'global_map.png', dpi=200, bbox_inches='tight', facecolor='black')
plt.close()
print("Updated global_map.png")
