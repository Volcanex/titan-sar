"""Execute Notebook 03 — Exploratory Data Analysis."""
import sys, os, json
os.chdir('/home/gabriel/titan-sar')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from skimage.feature import graycomatrix, graycoprops

from src.utils import (
    PROCESSED_DIR, FIGURES_DIR, SPLITS_DIR,
    TERRAIN_CLASSES, CLASS_COLORS, NUM_CLASSES, get_logger,
)

log = get_logger('03_eda')

EDA_DIR = FIGURES_DIR / 'eda'
EDA_DIR.mkdir(parents=True, exist_ok=True)

SAR_TILES_DIR = PROCESSED_DIR / 'sar_tiles'
LABEL_TILES_DIR = PROCESSED_DIR / 'label_tiles'

tile_df = pd.read_csv(PROCESSED_DIR / 'tile_metadata.csv')
with open(SPLITS_DIR / 'split_v1.json') as f:
    split_map = json.load(f)
tile_df['split'] = tile_df['tile_id'].map(split_map)

print(f'Total tiles: {len(tile_df)}')
print(f'Splits: {tile_df["split"].value_counts().to_dict()}')

# ── 1. Class Distribution ────────────────────────────────────────────
print("\n" + "="*60)
print("1. Class Distribution")
print("="*60)

class_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
class_counts_by_split = {s: np.zeros(NUM_CLASSES, dtype=np.int64) for s in ['train', 'val', 'test']}

sample_size = min(500, len(tile_df))
sample_df = tile_df.sample(n=sample_size, random_state=42)

for _, row in tqdm(sample_df.iterrows(), total=sample_size, desc='Counting pixels'):
    lbl = np.load(LABEL_TILES_DIR / f"{row['tile_id']}.npy")
    for cls in range(NUM_CLASSES):
        count = int((lbl == cls).sum())
        class_pixel_counts[cls] += count
        if row['split'] in class_counts_by_split:
            class_counts_by_split[row['split']][cls] += count

total_pixels = class_pixel_counts.sum()
class_fracs = class_pixel_counts / total_pixels

print('\nGlobal class distribution (sampled):')
for i in range(NUM_CLASSES):
    print(f'  {TERRAIN_CLASSES[i]:>12s}: {class_pixel_counts[i]:>12,} pixels ({100*class_fracs[i]:.1f}%)')

# Bar chart
names = [TERRAIN_CLASSES[i] for i in range(NUM_CLASSES)]
colors = [CLASS_COLORS[i] for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(names, class_fracs * 100, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Percentage of pixels')
axes[0].set_title('Global Class Distribution')
axes[0].tick_params(axis='x', rotation=45)

x = np.arange(NUM_CLASSES)
width = 0.25
split_colors = {'train': '#2196F3', 'val': '#FF9800', 'test': '#4CAF50'}
for j, (split_name, split_counts) in enumerate(class_counts_by_split.items()):
    split_total = split_counts.sum()
    fracs = split_counts / split_total * 100 if split_total > 0 else np.zeros(NUM_CLASSES)
    axes[1].bar(x + j * width, fracs, width, label=split_name, color=split_colors[split_name])

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(names, rotation=45)
axes[1].set_ylabel('Percentage of pixels')
axes[1].set_title('Class Distribution per Split')
axes[1].legend()
plt.tight_layout()
plt.savefig(EDA_DIR / 'class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: class_distribution.png')

# Class weights
class_weights = np.where(class_fracs > 0, 1.0 / class_fracs, 0.0)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

weights_dict = {TERRAIN_CLASSES[i]: float(class_weights[i]) for i in range(NUM_CLASSES)}
print('\nClass weights:')
for name, w in weights_dict.items():
    print(f'  {name:>12s}: {w:.4f}')

with open(PROCESSED_DIR / 'class_weights.json', 'w') as f:
    json.dump({
        'weights': weights_dict,
        'weights_list': class_weights.tolist(),
        'class_fractions': {TERRAIN_CLASSES[i]: float(class_fracs[i]) for i in range(NUM_CLASSES)},
    }, f, indent=2)

# ── 2. SAR Backscatter Statistics per Class ──────────────────────────
print("\n" + "="*60)
print("2. SAR Backscatter Statistics per Class")
print("="*60)

n_sample = min(200, len(tile_df))
sample_tiles = tile_df.sample(n=n_sample, random_state=123)['tile_id'].tolist()
max_per_class = 100_000
class_values = {i: [] for i in range(NUM_CLASSES)}

for tid in tqdm(sample_tiles, desc='Collecting SAR stats'):
    sar = np.load(SAR_TILES_DIR / f"{tid}.npy")
    lbl = np.load(LABEL_TILES_DIR / f"{tid}.npy")
    for cls in range(NUM_CLASSES):
        mask = lbl == cls
        if mask.any():
            vals = sar[mask]
            if len(vals) > 5000:
                vals = np.random.choice(vals, 5000, replace=False)
            class_values[cls].extend(vals.tolist())

for cls in range(NUM_CLASSES):
    if len(class_values[cls]) > max_per_class:
        class_values[cls] = class_values[cls][:max_per_class]
    print(f'  {TERRAIN_CLASSES[cls]:>12s}: {len(class_values[cls]):>8,} values')

fig, ax = plt.subplots(figsize=(10, 6))
for cls in range(NUM_CLASSES):
    vals = class_values[cls]
    if len(vals) > 100:
        ax.hist(vals, bins=100, density=True, alpha=0.4,
                color=CLASS_COLORS[cls], label=TERRAIN_CLASSES[cls])
ax.set_xlabel('Normalised SAR Backscatter')
ax.set_ylabel('Density')
ax.set_title('SAR Backscatter Distribution per Terrain Class')
ax.legend()
plt.tight_layout()
plt.savefig(EDA_DIR / 'backscatter_per_class.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: backscatter_per_class.png')

print('\nBackscatter statistics:')
print(f'{"Class":>12s}  {"Mean":>8s}  {"Std":>8s}  {"Median":>8s}')
for cls in range(NUM_CLASSES):
    vals = np.array(class_values[cls])
    if len(vals) > 0:
        print(f'{TERRAIN_CLASSES[cls]:>12s}  {vals.mean():>8.4f}  {vals.std():>8.4f}  {np.median(vals):>8.4f}')

# ── 3. GLCM Texture Analysis ─────────────────────────────────────────
print("\n" + "="*60)
print("3. GLCM Texture Analysis")
print("="*60)

GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

n_per_class = 20
texture_records = []

for cls in range(NUM_CLASSES):
    cls_tiles = tile_df.nlargest(n_per_class, f'class_{cls}_frac')['tile_id'].tolist()
    for tid in tqdm(cls_tiles, desc=f'GLCM {TERRAIN_CLASSES[cls]}', leave=False):
        sar = np.load(SAR_TILES_DIR / f"{tid}.npy")
        sar_q = (np.clip(sar, 0, 1) * 63).astype(np.uint8)
        glcm = graycomatrix(sar_q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                            levels=64, symmetric=True, normed=True)
        record = {'tile_id': tid, 'class': cls, 'class_name': TERRAIN_CLASSES[cls]}
        for prop in GLCM_PROPS:
            vals = graycoprops(glcm, prop)
            for d_idx, d in enumerate(GLCM_DISTANCES):
                record[f'{prop}_d{d}'] = float(vals[d_idx].mean())
        texture_records.append(record)

texture_df = pd.DataFrame(texture_records)
print(f'Computed GLCM features for {len(texture_df)} tiles')

feature_cols = [c for c in texture_df.columns if c not in ['tile_id', 'class', 'class_name']]
n_features = len(feature_cols)
n_cols_fig = 4
n_rows_fig = (n_features + n_cols_fig - 1) // n_cols_fig

fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(4 * n_cols_fig, 3 * n_rows_fig))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    data_by_class = [texture_df[texture_df['class'] == c][col].values for c in range(NUM_CLASSES)]
    bp = axes[i].boxplot(data_by_class, labels=[TERRAIN_CLASSES[c][:6] for c in range(NUM_CLASSES)],
                         patch_artist=True)
    for j, box in enumerate(bp['boxes']):
        box.set_facecolor(CLASS_COLORS[j])
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(axis='x', rotation=45, labelsize=7)

for i in range(n_features, len(axes)):
    axes[i].axis('off')

plt.suptitle('GLCM Texture Features by Terrain Class', fontsize=14)
plt.tight_layout()
plt.savefig(EDA_DIR / 'glcm_boxplots.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: glcm_boxplots.png')

# ── 4. Example Gallery ───────────────────────────────────────────────
print("\n" + "="*60)
print("4. Example Gallery")
print("="*60)

class_cmap = ListedColormap([CLASS_COLORS[i] for i in range(NUM_CLASSES)])
n_examples = 4

fig, axes = plt.subplots(NUM_CLASSES, n_examples, figsize=(3 * n_examples, 3 * NUM_CLASSES))

for cls in range(NUM_CLASSES):
    cls_tiles = tile_df.nlargest(n_examples * 3, f'class_{cls}_frac')['tile_id'].tolist()
    shown = 0
    for tid in cls_tiles:
        if shown >= n_examples:
            break
        sar = np.load(SAR_TILES_DIR / f"{tid}.npy")
        lbl = np.load(LABEL_TILES_DIR / f"{tid}.npy")
        ax = axes[cls, shown]
        ax.imshow(sar, cmap='gray', vmin=0, vmax=1)
        lbl_masked = np.ma.masked_where(lbl == 255, lbl)
        ax.imshow(lbl_masked, cmap=class_cmap, vmin=0, vmax=NUM_CLASSES-1, alpha=0.35)
        if shown == 0:
            ax.set_ylabel(TERRAIN_CLASSES[cls], fontsize=11, fontweight='bold')
        ax.axis('off')
        shown += 1

plt.suptitle('Representative SAR Patches by Terrain Class', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(EDA_DIR / 'example_gallery.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: example_gallery.png')

print(f'\nAll EDA figures saved to: {EDA_DIR}')
print(f'Class weights saved to: {PROCESSED_DIR / "class_weights.json"}')
print('Ready for Notebook 04.')
