#!/bin/bash
set -e

echo "============================================"
echo "  Titan SAR — GPU Training Pipeline"
echo "  RunPod RTX A5000"
echo "============================================"

cd /workspace

# 1. Extract dataset
if [ ! -d "data/processed/sar_tiles" ]; then
    echo "[1/7] Extracting dataset..."
    mkdir -p data
    tar xzf titan_sar_dataset.tar.gz -C data/
    echo "  Tiles: $(ls data/processed/sar_tiles/*.npy | wc -l) SAR, $(ls data/processed/label_tiles/*.npy | wc -l) labels"
else
    echo "[1/7] Dataset already extracted."
fi

# 2. Install dependencies
echo "[2/7] Installing dependencies..."
pip install -q segmentation_models_pytorch albumentations tensorboard pyyaml scikit-learn 2>&1 | tail -2

# 3. Verify GPU
echo "[3/7] GPU check:"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"

mkdir -p models data/predictions runs

# ============================================
# Run 1: U-Net + ResNet34 (primary result)
# ============================================
echo ""
echo "============================================"
echo "[4/7] Run 1: U-Net + ResNet34 + Focal Loss"
echo "============================================"
python3 train.py \
    --config configs/unet_resnet34.yaml \
    --run-name unet_r34_focal \
    --data-dir data

# ============================================
# Run 2: DeepLabV3+ + ResNet50
# ============================================
echo ""
echo "============================================"
echo "[5/7] Run 2: DeepLabV3+ + ResNet50 + Focal Loss"
echo "============================================"
python3 train.py \
    --config configs/deeplabv3_resnet50.yaml \
    --run-name dlv3_r50_focal \
    --data-dir data

# ============================================
# Run 3: U-Net + ResNet34 with NLDSAR (if available)
# ============================================
if [ -d "data/processed/nldsar_tiles" ] && [ "$(ls data/processed/nldsar_tiles/*.npy 2>/dev/null | wc -l)" -gt 0 ]; then
    echo ""
    echo "============================================"
    echo "[5.5/7] Run 3: U-Net + ResNet34 on NLDSAR"
    echo "============================================"
    python3 train.py \
        --config configs/unet_resnet34.yaml \
        --run-name unet_r34_nldsar \
        --data-dir data \
        --sar-dir-name nldsar_tiles
else
    echo ""
    echo "[5.5/7] Skipping NLDSAR run — no NLDSAR tiles available."
fi

# ============================================
# Run 4: Domain Gap — Random Init
# ============================================
echo ""
echo "============================================"
echo "[6/7] Run 4: Domain Gap — Random Initialisation"
echo "============================================"
python3 train.py \
    --config configs/unet_resnet34.yaml \
    --run-name domain_gap_random \
    --data-dir data \
    --encoder-weights None

# ============================================
# Run 5: Domain Gap — Frozen Encoder (Linear Probing)
# ============================================
echo ""
echo "============================================"
echo "[7/7] Run 5: Domain Gap — Frozen Encoder"
echo "============================================"
python3 train.py \
    --config configs/unet_resnet34.yaml \
    --run-name domain_gap_frozen \
    --data-dir data \
    --freeze-encoder

echo ""
echo "============================================"
echo "  ALL TRAINING COMPLETE"
echo "============================================"
echo "Results:"
for f in models/*_metrics.json; do
    name=$(python3 -c "import json; m=json.load(open('$f')); print(f\"{m['run_name']:30s} val_iou={m['best_val_iou']:.4f}  test_iou={m['test_iou']:.4f}\")")
    echo "  $name"
done
echo ""
echo "Ready for download."
