# Titan SAR Terrain Classification

Pixel-level semantic segmentation of Saturn's moon Titan from Cassini RADAR/SAR imagery. First application of deep learning to this problem.

## Why This Matters

NASA's Dragonfly rotorcraft launches July 2028 to land on Titan — the only other world with liquid on its surface (methane/ethane seas). The thick atmosphere means we can't photograph the surface; all we have is radar. Scientists have hand-drawn coarse terrain maps from these radar images (Lopes et al. 2020), but nobody has automated this at pixel-level resolution.

This project produces an automated terrain classification at **every 351m pixel** of Titan's surface — roughly 65,000x finer spatial resolution than existing per-tile approaches.

## Results

| Model | Test mIoU | Notes |
|-------|-----------|-------|
| Random Forest (baseline) | 0.293 | Tile-level classification, GLCM texture features |
| U-Net + ResNet34 (ImageNet) | 0.350 | +19% over baseline |
| U-Net + ResNet34 (random init) | 0.348 | ImageNet pretraining barely helps |
| U-Net + ResNet34 (frozen encoder) | 0.339 | Features need fine-tuning |
| **DeepLabV3+ + ResNet50 (ImageNet)** | **0.371** | **Best model, +27% over baseline** |

**Per-class IoU (best model):**

| Terrain | IoU | Coverage | Notes |
|---------|-----|----------|-------|
| Plains | 0.74 | ~65% | Dominant class — well-classified |
| Lakes/Seas | 0.52 | ~4% | Dark radar return makes these distinctive |
| Dunes | 0.38 | ~9% | Linear texture helps, but confused with plains |
| Hummocky | 0.32 | ~13% | Bright radar return, confused with labyrinth |
| Labyrinth | 0.28 | ~6% | Complex dissected terrain, hard to distinguish |
| Craters | 0.00 | ~0.3% | Too few examples to learn from |

### Key findings

1. **Pixel-level segmentation works on Titan SAR.** The DL model draws terrain boundaries within tiles, not just classifying whole patches. This is the main contribution.
2. **Earth-to-Titan domain gap is massive.** ImageNet pretrained weights (0.350) perform almost identically to random initialisation (0.348). Cat photos don't teach you about methane seas.
3. **DeepLabV3+'s larger receptive field helps.** Especially on labyrinth and hummocky terrain, where spatial context over tens of kilometres matters.

### Honest caveats

- **mIoU of 0.37 is modest.** The model is right about terrain type roughly half the time at pixel level. It over-predicts plains (the safe bet at 65% coverage).
- **No independent validation.** We train and evaluate against Lopes et al.'s hand-drawn map. If their map has errors (and it certainly does at boundaries), we're learning those errors. Agreement with the map doesn't prove ground truth accuracy.
- **The global map is a first pass, not a finished product.** Tile boundary artefacts exist. Rare classes (craters, labyrinth) are poorly served. The model has never seen Titan's north polar lakes region well enough to be trusted there.
- **351m resolution is still coarse.** Each pixel covers ~0.12 km² — useful for regional characterisation, not for picking a specific landing spot.

## How do we know it's accurate?

Honestly, we have limited evidence:

1. **Geographic test split.** Train/val/test regions are separated by 10°+ latitude/longitude blocks, so the model can't just memorise local patterns. The test mIoU is on genuinely unseen terrain.
2. **Resolution sanity test.** Performance degrades gracefully when we downsample from 351m to 700m, confirming the model learns terrain morphology rather than resolution-specific noise.
3. **Physical plausibility.** Lakes appear dark in SAR (smooth liquid surface), dunes show linear texture, mountains are bright (rough surfaces scatter radar). The model's predictions are consistent with known SAR physics.
4. **RF baseline comparison.** The DL model beats hand-crafted features designed by humans who understand SAR, suggesting it's learning something real.

What we **don't** have: ground truth from the surface. Dragonfly will provide that in the 2030s. Until then, any Titan terrain map — including Lopes et al.'s — is an educated interpretation of radar data.

## Pipeline

| Notebook | Description | Compute |
|----------|-------------|---------|
| `01_data_acquisition` | Download SAR mosaic, geomorphological map, BIDR swaths | CPU |
| `02_preprocessing_and_tiling` | Align SAR+labels, tile to 256×256, geographic train/val/test split | CPU |
| `03_exploratory_data_analysis` | Class distribution, backscatter per class, GLCM texture, semivariogram | CPU |
| `04_traditional_ml_baseline` | Random Forest on hand-crafted features (performance floor) | CPU |
| `05_resolution_sanity_test` | Multi-resolution robustness and cross-resolution transfer | CPU |
| `06_deep_learning_training` | U-Net, DeepLabV3+ with pretrained encoders (5 runs, ~$0.50) | GPU |
| `07_domain_gap_analysis` | Random vs ImageNet vs frozen encoder comparison | GPU |
| `08_full_map_generation` | Pixel-level global terrain map + Dragonfly landing region | CPU |
| `09_evaluation_and_figures` | Publication-quality figures and final metrics | CPU |

## Data Sources

1. **USGS SAR-HiSAR Global Mosaic** — 351 m/pixel, ~1 GB GeoTIFF, Ku-band backscatter
2. **Lopes et al. (2020) Geomorphological Map** — 6-class ground truth labels (USGS global geology GIS)
3. **Cassini BIDR Swaths** — 175 m/pixel individual SAR passes (PDS, used for resolution testing)

## Project Structure

```
titan-sar/
├── notebooks/          # Jupyter pipeline (01-09, outputs saved)
├── data/
│   ├── raw/            # ~1.5 GB downloaded imagery (not in git)
│   ├── processed/      # 10,711 tiles × 256×256 pixels
│   ├── splits/         # Geographic train/val/test split
│   └── predictions/    # Pixel-level global map (1 GB GeoTIFF)
├── models/             # Trained weights (.pth) and metrics (.json)
├── configs/            # Training configs (YAML)
├── src/
│   ├── train.py        # Standalone GPU training script
│   ├── dataset.py      # PyTorch Dataset
│   ├── metrics.py      # IoU, F1, confusion matrix
│   ├── transforms.py   # Augmentation pipelines
│   └── utils.py        # Paths, GeoTIFF I/O, Titan CRS definition
├── figures/            # 17 publication-quality figures (300 DPI)
├── environment.yml
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GPU training also needs:
pip install -r requirements_gpu.txt
```

## Hardware

- **CPU** (NB 01-05, 08-09): Ryzen 3700X, 32 GB RAM, Ubuntu
- **GPU** (NB 06-07): RunPod RTX A5000 24GB, $0.16/hr — total spend ~$0.50
- Full CPU inference for global map: ~53 minutes on Ryzen 3700X

## What would improve this

- **More training data.** Regional maps (Malaska 2016, Schoenfeld 2023) offer finer sub-classes.
- **Purpose-built architecture.** Since ImageNet doesn't help, a smaller model trained from scratch could be more efficient.
- **Better loss function.** Dice loss or class-balanced sampling to force learning of rare classes.
- **NLDSAR denoised input.** Denoised SAR may improve boundary detection (data downloaded but not yet tiled).
- **Higher resolution BIDR input.** 175m swaths exist for select regions — 4x more spatial detail.
- **Overlapping tile inference.** Averaging predictions in overlap zones would reduce tile boundary artefacts.

## References

- Lopes, R.M.C. et al. (2020). A global geomorphologic map of Saturn's moon Titan. *Nature Astronomy*, 4, 228–233. doi:10.1038/s41550-019-0917-6
- Lorenz, R.D. et al. (2021). Selection and characteristics of the Dragonfly landing site. *The Planetary Science Journal*, 2(1), 24.
