#!/bin/bash
# ============================================================
# Run all training experiments on RunPod
# Budget: ~$7-8 total (~26 GPU hours on A5000 @ $0.16/hr)
# ============================================================

set -e
cd /workspace

echo "Starting all training runs at $(date)"
echo ""

# Run 1: U-Net + ResNet34 (ImageNet) — PRIMARY RESULT
echo "============================================================"
echo "RUN 1: U-Net + ResNet34 + Focal Loss (~5h)"
echo "============================================================"
python3 src/train.py \
    --config configs/unet_resnet34.yaml \
    --data-dir data \
    --run-name unet_r34_focal

# Run 2: DeepLabV3+ + ResNet50 — second architecture
echo ""
echo "============================================================"
echo "RUN 2: DeepLabV3+ + ResNet50 + Focal Loss (~5h)"
echo "============================================================"
python3 src/train.py \
    --config configs/deeplabv3_resnet50.yaml \
    --data-dir data \
    --run-name dlv3_r50_focal

# Run 3: Domain gap — random initialisation
echo ""
echo "============================================================"
echo "RUN 3: U-Net + ResNet34 Random Init (domain gap) (~5h)"
echo "============================================================"
python3 src/train.py \
    --config configs/unet_resnet34_random.yaml \
    --data-dir data \
    --run-name domain_gap_random

echo ""
echo "============================================================"
echo "All runs complete at $(date)"
echo "============================================================"
echo ""
echo "Results:"
ls -lh models/*_best.pth models/*_metrics.json 2>/dev/null
echo ""
echo "Download with:"
echo "  scp -r runpod:/workspace/models/ ."
echo "  scp -r runpod:/workspace/runs/ ."
echo "  scp -r runpod:/workspace/data/predictions/ ."
