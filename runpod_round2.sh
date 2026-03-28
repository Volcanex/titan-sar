#!/bin/bash
set -eo pipefail

echo "============================================"
echo "  Titan SAR — Round 2 GPU Training"
echo "  Mendeley Labels + Dice Loss + 100 epochs"
echo "============================================"

cd /workspace

# 1. Extract dataset
if [ ! -d "data/processed/sar_tiles" ]; then
    echo "[1/4] Extracting dataset..."
    mkdir -p data
    tar xzf titan_sar_dataset.tar.gz -C data/
    echo "  SAR tiles: $(ls data/processed/sar_tiles/*.npy | wc -l)"
    echo "  Label tiles: $(ls data/processed/label_tiles/*.npy | wc -l)"
else
    echo "[1/4] Dataset already extracted."
fi

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install -q segmentation_models_pytorch albumentations tensorboard pyyaml scikit-learn 2>&1 | tail -2

# 3. Verify GPU
echo "[3/4] GPU check:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

mkdir -p models data/predictions runs

# ============================================
# Run 1: U-Net + ResNet34 + Dice + 100ep
# ============================================
echo ""
echo "============================================"
echo "[4a/4] U-Net ResNet34 + Dice Loss + 100ep"
echo "============================================"
python3 train.py \
    --config configs/unet_r34_dice_100ep.yaml \
    --run-name r2_unet_r34_dice \
    --data-dir data

# ============================================
# Run 2: DeepLabV3+ ResNet50 + Dice + 100ep
# ============================================
echo ""
echo "============================================"
echo "[4b/4] DeepLabV3+ ResNet50 + Dice Loss + 100ep"
echo "============================================"
python3 train.py \
    --config configs/dlv3_r50_dice_100ep.yaml \
    --run-name r2_dlv3_r50_dice \
    --data-dir data

echo ""
echo "============================================"
echo "  ROUND 2 TRAINING COMPLETE"
echo "============================================"
echo "Results:"
for f in models/r2_*_metrics.json; do
    if [ -f "$f" ]; then
        name=$(python3 -c "import json; m=json.load(open('$f')); print(f\"{m['run_name']:30s} val_iou={m['best_val_iou']:.4f}  test_iou={m['test_iou']:.4f}\")")
        echo "  $name"
    fi
done
echo ""
echo "Ready for download. Pod can be terminated."
