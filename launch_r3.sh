#!/bin/bash
# Round 3: Better encoders + 150 epochs
# EfficientNet-B4, ConvNeXt-Small
cd /workspace

# Extract if needed
if [ ! -d data/processed/sar_tiles ]; then
    echo "[1] Extracting dataset..."
    mkdir -p data
    tar xzf titan_sar_dataset.tar.gz -C data/
fi

echo "Tiles: $(ls data/processed/sar_tiles/*.npy | wc -l) SAR, $(ls data/processed/label_tiles/*.npy | wc -l) labels"
mkdir -p models runs

# Install deps (pin numpy<2 for PyTorch compat)
pip install -q segmentation_models_pytorch albumentations tensorboard pyyaml scikit-learn timm 2>&1 | tail -2
pip install -q "numpy<2" 2>&1 | tail -1

echo "[GPU]"
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# Run 1: U-Net + EfficientNet-B4
echo ""
echo "============================================"
echo "Run 1: U-Net + EfficientNet-B4 + Dice 150ep"
echo "============================================"
python3 train.py \
    --config configs/unet_efficientb4_dice_150ep.yaml \
    --run-name r3_unet_effb4_dice \
    --data-dir data

# Run 2: U-Net + ConvNeXt-Small
echo ""
echo "============================================"
echo "Run 2: U-Net + ConvNeXt-Small + Dice 150ep"
echo "============================================"
python3 train.py \
    --config configs/unet_convnext_dice_150ep.yaml \
    --run-name r3_unet_convnext_dice \
    --data-dir data

# Run 3: DeepLabV3+ + EfficientNet-B4
echo ""
echo "============================================"
echo "Run 3: DeepLabV3+ + EfficientNet-B4 + Dice 150ep"
echo "============================================"
python3 train.py \
    --config configs/dlv3_efficientb4_dice_150ep.yaml \
    --run-name r3_dlv3_effb4_dice \
    --data-dir data

echo ""
echo "============================================"
echo "  ROUND 3 COMPLETE"
echo "============================================"
echo "Results:"
for f in models/r3_*_metrics.json; do
    if [ -f "$f" ]; then
        python3 -c "import json; m=json.load(open('$f')); print(f\"  {m['run_name']:30s} val_iou={m['best_val_iou']:.4f}  test_iou={m['test_iou']:.4f}\")"
    fi
done
echo ""
echo "Ready for download."
