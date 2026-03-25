"""Execute Notebook 05 — Resolution Sanity Test."""
import sys, os, json
os.chdir('/home/gabriel/titan-sar')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

from src.utils import (
    PROCESSED_DIR, SPLITS_DIR, FIGURES_DIR,
    TERRAIN_CLASSES, NUM_CLASSES, get_logger,
)
from src.metrics import compute_all_metrics, print_metrics_table

log = get_logger('05_resolution')

SAR_TILES_DIR = PROCESSED_DIR / 'sar_tiles'
LABEL_TILES_DIR = PROCESSED_DIR / 'label_tiles'

with open(SPLITS_DIR / 'split_v1.json') as f:
    split_map = json.load(f)

# Use a subset for speed
train_ids = [k for k, v in split_map.items() if v == 'train'][:500]
val_ids = [k for k, v in split_map.items() if v == 'val'][:200]
test_ids = [k for k, v in split_map.items() if v == 'test'][:200]

print(f'Resolution test subset: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test')

# ── Helpers ───────────────────────────────────────────────────────────
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

def extract_features(sar_tile):
    features = {}
    valid = sar_tile[sar_tile > 0]
    if len(valid) == 0:
        valid = sar_tile.flatten()
    features['mean'] = float(valid.mean())
    features['std'] = float(valid.std())
    features['median'] = float(np.median(valid))
    features['skewness'] = float(skew(valid))
    features['kurtosis'] = float(kurtosis(valid))
    features['p10'] = float(np.percentile(valid, 10))
    features['p90'] = float(np.percentile(valid, 90))

    sar_q = (np.clip(sar_tile, 0, 1) * 63).astype(np.uint8)
    glcm = graycomatrix(sar_q, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=64, symmetric=True, normed=True)
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        for d_idx, d in enumerate(GLCM_DISTANCES):
            features[f'{prop}_d{d}'] = float(vals[d_idx].mean())
    return features

def get_majority_label(lbl_tile):
    valid = lbl_tile[lbl_tile != 255]
    if len(valid) == 0:
        return -1
    values, counts = np.unique(valid, return_counts=True)
    return int(values[np.argmax(counts)])

def downsample_tile(sar_tile, factor=2):
    h, w = sar_tile.shape
    new_h, new_w = h // factor, w // factor
    return sar_tile[:new_h*factor, :new_w*factor].reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))

def downsample_label(lbl_tile, factor=2):
    h, w = lbl_tile.shape
    new_h, new_w = h // factor, w // factor
    result = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            block = lbl_tile[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            valid = block[block != 255]
            if len(valid) > 0:
                vals, counts = np.unique(valid, return_counts=True)
                result[i, j] = vals[np.argmax(counts)]
            else:
                result[i, j] = 255
    return result

# ── Extract features at each resolution ──────────────────────────────
RESOLUTIONS = {
    '351m (native)': 1,
    '700m (2x down)': 2,
}

print("\n" + "="*60)
print("Extracting features at multiple resolutions")
print("="*60)

results_by_res = {}

for res_name, factor in RESOLUTIONS.items():
    log.info(f'Processing {res_name}...')
    datasets = {}
    for split_name, tile_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        records = []
        for tid in tqdm(tile_ids, desc=f'{res_name} {split_name}', leave=False):
            sar = np.load(SAR_TILES_DIR / f"{tid}.npy")
            lbl = np.load(LABEL_TILES_DIR / f"{tid}.npy")

            if factor > 1:
                sar = downsample_tile(sar, factor)
                lbl = downsample_label(lbl, factor)

            label = get_majority_label(lbl)
            if label < 0:
                continue

            feats = extract_features(sar)
            feats['label'] = label
            records.append(feats)

        datasets[split_name] = pd.DataFrame(records)
    results_by_res[res_name] = datasets
    for s, df in datasets.items():
        print(f'  {res_name} {s}: {len(df)} tiles')

# ── Train and evaluate RF at each resolution ─────────────────────────
print("\n" + "="*60)
print("Training RF at each resolution")
print("="*60)

resolution_metrics = {}
trained_models = {}

for res_name, datasets in results_by_res.items():
    train_data = datasets['train']
    test_data = datasets['test']
    feature_cols = [c for c in train_data.columns if c != 'label']

    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42
    )
    rf.fit(train_data[feature_cols].values, train_data['label'].values)
    trained_models[res_name] = (rf, feature_cols)

    y_pred = rf.predict(test_data[feature_cols].values)
    y_test = test_data['label'].values
    metrics = compute_all_metrics(y_test, y_pred)
    resolution_metrics[res_name] = metrics

    print(f'\n{res_name}:')
    print_metrics_table(metrics)

# ── Cross-resolution transfer ─────────────────────────────────────────
print("\n" + "="*60)
print("Cross-Resolution Transfer")
print("="*60)

res_names = list(results_by_res.keys())
transfer_matrix = pd.DataFrame(index=res_names, columns=res_names, dtype=float)

for train_res in res_names:
    rf, feature_cols = trained_models[train_res]
    for test_res in res_names:
        test_data = results_by_res[test_res]['test']
        if len(test_data) == 0:
            continue
        y_pred = rf.predict(test_data[feature_cols].values)
        y_test = test_data['label'].values
        acc = compute_all_metrics(y_test, y_pred)['overall_accuracy']
        transfer_matrix.loc[train_res, test_res] = acc

print('Transfer Matrix (accuracy):')
print('Rows = train, Cols = test')
print(transfer_matrix.to_string(float_format='%.3f'))

# ── Summary Figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

res_labels = list(resolution_metrics.keys())
accs = [resolution_metrics[r]['overall_accuracy'] for r in res_labels]
mious = [resolution_metrics[r]['mean_iou'] for r in res_labels]

x = np.arange(len(res_labels))
width = 0.35
axes[0].bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
axes[0].bar(x + width/2, mious, width, label='Mean IoU', color='darkorange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(res_labels)
axes[0].set_ylabel('Score')
axes[0].set_title('RF Performance by Resolution')
axes[0].legend()
axes[0].set_ylim(0, 1)

tm = transfer_matrix.astype(float).values
im = axes[1].imshow(tm, cmap='YlOrRd', vmin=0, vmax=1)
axes[1].set_xticks(range(len(res_labels)))
axes[1].set_xticklabels([r.split(' ')[0] for r in res_labels])
axes[1].set_yticks(range(len(res_labels)))
axes[1].set_yticklabels([r.split(' ')[0] for r in res_labels])
axes[1].set_xlabel('Test Resolution')
axes[1].set_ylabel('Train Resolution')
axes[1].set_title('Cross-Resolution Transfer (Accuracy)')
for i in range(len(res_labels)):
    for j in range(len(res_labels)):
        if not np.isnan(tm[i, j]):
            axes[1].text(j, i, f'{tm[i,j]:.2f}', ha='center', va='center', fontsize=14)
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'resolution_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('\nSaved: resolution_comparison.png')

print('\nInterpretation:')
diff = abs(accs[0] - accs[1]) if len(accs) >= 2 else 0
if diff < 0.05:
    print('  Features are SCALE-ROBUST (< 5% accuracy difference)')
elif diff < 0.15:
    print('  Graceful degradation — model learns terrain morphology')
else:
    print('  Significant drop — model may overfit to resolution-specific noise')

print('\nReady for Notebook 06 (deep learning).')
