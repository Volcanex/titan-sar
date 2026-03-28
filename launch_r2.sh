#!/bin/bash
cd /workspace
pkill -f train.py 2>/dev/null
sleep 2
> /workspace/training.log

echo "=== Run 1: U-Net ResNet34 + Dice 100ep ===" >> /workspace/training.log 2>&1
python3 train.py --config configs/unet_r34_dice_100ep.yaml --run-name r2_unet_r34_dice --data-dir data >> /workspace/training.log 2>&1

echo "=== Run 2: DeepLabV3+ ResNet50 + Dice 100ep ===" >> /workspace/training.log 2>&1
python3 train.py --config configs/dlv3_r50_dice_100ep.yaml --run-name r2_dlv3_r50_dice --data-dir data >> /workspace/training.log 2>&1

echo "=== ALL DONE ===" >> /workspace/training.log 2>&1
