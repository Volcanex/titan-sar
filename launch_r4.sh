#!/bin/bash
set -eo pipefail
cd /workspace

# Only extract if not already done
if [ ! -d "data/processed/sar_tiles" ]; then
    echo "[1] Extracting dataset..."
    tar xzf titan_sar_dataset_r4.tar.gz -C data/ 2>/dev/null || {
        # Fallback: extract and fix structure
        tar xzf titan_sar_dataset_r4.tar.gz
        if [ -d "processed" ] && [ ! -d "data/processed" ]; then
            mkdir -p data
            mv processed data/processed
            mv splits data/splits 2>/dev/null || true
        fi
    }
else
    echo "[1] Dataset already extracted"
fi
echo "Tiles: $(ls data/processed/sar_tiles/*.npy | wc -l) SAR, $(ls data/processed/nldsar_tiles/*.npy 2>/dev/null | wc -l) NLDSAR, $(ls data/processed/label_tiles/*.npy | wc -l) labels"

# Install deps (numpy<2 for torch compatibility)
echo "[2] Installing deps..."
pip install -q 'numpy<2' segmentation_models_pytorch albumentations timm tensorboard 2>&1 | tail -2

echo "[GPU] $(python3 -c 'import torch; g=torch.cuda.get_device_properties(0); print(f"{g.name}, {g.total_memory//1024**2}MB")')"

# Run 1 — NLDSAR + SAR combined (checkpoint auto-resumes)
echo "============================================"
echo "Run 1: U-Net EfficientNet-B4 + Dice + NLDSAR (150ep)"
echo "============================================"
python3 train.py --config configs/unet_efficientb4_dice_150ep.yaml --run-name r4_unet_effb4_nldsar --data-dir data --nldsar

# Run 2 — SAR only with proper normalization
echo "============================================"
echo "Run 2: U-Net EfficientNet-B4 + Dice + SAR normalized (150ep)"
echo "============================================"
python3 train.py --config configs/unet_efficientb4_dice_150ep.yaml --run-name r4_unet_effb4_norm --data-dir data

echo "=== ALL DONE ==="
