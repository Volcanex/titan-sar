"""Execute Notebook 04 — Traditional ML Baseline (Random Forest)."""
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
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

from src.utils import (
    PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR,
    TERRAIN_CLASSES, NUM_CLASSES, get_logger,
)
from src.metrics import compute_all_metrics, print_metrics_table

log = get_logger('04_rf_baseline')

SAR_TILES_DIR = PROCESSED_DIR / 'sar_tiles'
LABEL_TILES_DIR = PROCESSED_DIR / 'label_tiles'
BASELINE_FIG_DIR = FIGURES_DIR / 'baselines'
BASELINE_FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(SPLITS_DIR / 'split_v1.json') as f:
    split_map = json.load(f)

train_ids = [k for k, v in split_map.items() if v == 'train']
val_ids = [k for k, v in split_map.items() if v == 'val']
test_ids = [k for k, v in split_map.items() if v == 'test']

print(f'Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}')

# ── Feature Extraction ───────────────────────────────────────────────
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

def extract_split_features(tile_ids, split_name):
    records = []
    for tid in tqdm(tile_ids, desc=f'Features ({split_name})'):
        sar = np.load(SAR_TILES_DIR / f"{tid}.npy")
        lbl = np.load(LABEL_TILES_DIR / f"{tid}.npy")
        label = get_majority_label(lbl)
        if label < 0:
            continue
        feats = extract_features(sar)
        feats['tile_id'] = tid
        feats['label'] = label
        records.append(feats)
    return pd.DataFrame(records)

print("\n" + "="*60)
print("1. Feature Extraction")
print("="*60)

train_df = extract_split_features(train_ids, 'train')
val_df = extract_split_features(val_ids, 'val')
test_df = extract_split_features(test_ids, 'test')

print(f'Train: {len(train_df)} tiles, Val: {len(val_df)}, Test: {len(test_df)}')

feature_cols = [c for c in train_df.columns if c not in ['tile_id', 'label']]
X_train = train_df[feature_cols].values
y_train = train_df['label'].values
X_val = val_df[feature_cols].values
y_val = val_df['label'].values
X_test = test_df[feature_cols].values
y_test = test_df['label'].values

print(f'Features ({len(feature_cols)}): {feature_cols}')
print(f'Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}')

# ── Train ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. Training Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=0,
)

log.info('Training Random Forest (500 trees)...')
rf.fit(X_train, y_train)
log.info('Training complete.')

model_path = MODELS_DIR / 'rf_baseline_v1.joblib'
joblib.dump(rf, model_path)
print(f'Model saved to {model_path}')

# ── Evaluate ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. Evaluation")
print("="*60)

y_val_pred = rf.predict(X_val)
y_test_pred = rf.predict(X_test)

np.save(PREDICTIONS_DIR / 'rf_baseline_test_predictions.npy', y_test_pred)

print("\nVALIDATION SET")
print("-"*50)
val_metrics = compute_all_metrics(y_val, y_val_pred)
print_metrics_table(val_metrics)

print("\nTEST SET")
print("-"*50)
test_metrics = compute_all_metrics(y_test, y_test_pred)
print_metrics_table(test_metrics)

# Save metrics
all_metrics = {
    'model': 'RandomForest_baseline_v1',
    'n_estimators': 500,
    'features': feature_cols,
    'validation': val_metrics,
    'test': test_metrics,
}
with open(MODELS_DIR / 'rf_baseline_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ── Confusion Matrix ──────────────────────────────────────────────────
print("\n" + "="*60)
print("4. Confusion Matrix & Feature Importance")
print("="*60)

class_names = [TERRAIN_CLASSES[i] for i in range(NUM_CLASSES)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (y_true, y_pred, title) in zip(axes, [
    (y_val, y_val_pred, 'Validation'),
    (y_test, y_test_pred, 'Test'),
]):
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        normalize='true',
        ax=ax,
        cmap='Blues',
        values_format='.2f',
    )
    ax.set_title(f'RF Baseline — {title}')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(BASELINE_FIG_DIR / 'rf_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: rf_confusion_matrix.png')

# Feature importance
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(feature_cols)), importances[sorted_idx[::-1]], align='center')
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels([feature_cols[i] for i in sorted_idx[::-1]], fontsize=8)
ax.set_xlabel('Feature Importance (Gini)')
ax.set_title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig(BASELINE_FIG_DIR / 'rf_feature_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved: rf_feature_importance.png')

print('\nTop 10 features:')
for i in sorted_idx[:10]:
    print(f'  {feature_cols[i]:>25s}: {importances[i]:.4f}')

# Sanity check
acc = test_metrics['overall_accuracy']
miou = test_metrics['mean_iou']
print(f'\nTest Accuracy: {acc:.3f}')
print(f'Test mIoU:     {miou:.3f}')

if acc < 0.30:
    print('WARNING: Accuracy < 30%. Something may be wrong.')
elif acc > 0.85:
    print('NOTE: Accuracy > 85%. RF baseline is already strong.')
else:
    print(f'Baseline in expected range. Deep learning should improve on this.')

print('\nReady for Notebook 05.')
