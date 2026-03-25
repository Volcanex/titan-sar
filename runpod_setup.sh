#!/bin/bash
# ============================================================
# RunPod Setup Script for Titan SAR Deep Learning Training
# ============================================================
# Upload this script + dataset archive to RunPod, then run:
#   bash runpod_setup.sh
#
# Prerequisites:
#   - data/titan_sar_dataset.tar.gz uploaded to /workspace/
#   - RTX A5000 or similar GPU pod
# ============================================================

set -e

echo "============================================================"
echo "Titan SAR — RunPod GPU Training Setup"
echo "============================================================"

cd /workspace

# 1. Extract dataset
echo ""
echo ">>> Extracting dataset..."
mkdir -p data
tar xzf titan_sar_dataset.tar.gz -C data/
echo "Dataset extracted. Contents:"
ls -lh data/processed/sar_tiles/ | head -5
echo "... $(ls data/processed/sar_tiles/*.npy | wc -l) SAR tiles total"
echo "... $(ls data/processed/label_tiles/*.npy | wc -l) label tiles total"

# 2. Install dependencies
echo ""
echo ">>> Installing dependencies..."
pip install -q torch torchvision segmentation-models-pytorch albumentations \
    tensorboard numpy scipy scikit-learn scikit-image pandas matplotlib \
    pyyaml tqdm rasterio

# 3. Clone repo (for src/ and configs/)
echo ""
echo ">>> Setting up code..."
if [ ! -d "titan-sar" ]; then
    git clone https://github.com/Volcanex/titan-sar.git
fi
cp -r titan-sar/src .
cp -r titan-sar/configs .

# 4. Verify GPU
echo ""
echo ">>> GPU check:"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 5. Create output directories
mkdir -p models runs data/predictions

echo ""
echo "============================================================"
echo "Setup complete! Run training with:"
echo ""
echo "  # Run 1: U-Net + ResNet34 (primary)"
echo "  python3 src/train.py --config configs/unet_resnet34.yaml --data-dir data --run-name unet_r34_focal"
echo ""
echo "  # Run 2: DeepLabV3+ + ResNet50"
echo "  python3 src/train.py --config configs/deeplabv3_resnet50.yaml --data-dir data --run-name dlv3_r50_focal"
echo ""
echo "  # Run 3: U-Net on NLDSAR (if nldsar_tiles available)"
echo "  python3 src/train.py --config configs/unet_resnet34.yaml --data-dir data --run-name unet_r34_nldsar"
echo ""
echo "  # Run 4: Domain gap — random init"
echo "  python3 src/train.py --config configs/unet_resnet34_random.yaml --data-dir data --run-name domain_gap_random"
echo ""
echo "When done, download results:"
echo "  scp -r runpod:/workspace/models/ ."
echo "  scp -r runpod:/workspace/runs/ ."
echo "  scp -r runpod:/workspace/data/predictions/ ."
echo "============================================================"
