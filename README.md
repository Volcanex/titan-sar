# Titan SAR Terrain Classification

Semantic segmentation of Saturn's moon Titan's surface terrain from Cassini RADAR/SAR imagery using machine learning.

## Motivation

NASA's Dragonfly rotorcraft mission launches July 2028 to land on Titan. Pre-launch terrain characterisation from Cassini SAR data is urgently relevant. Published geomorphological maps (Lopes et al. 2020, Nature Astronomy) provide labelled ground truth for six terrain classes: **plains, dunes, hummocky/mountainous, lakes/seas, labyrinth, craters**.

No one has applied modern deep learning to this problem.

## Pipeline

| Notebook | Description | Compute |
|----------|-------------|---------|
| `01_data_acquisition` | Download SAR mosaic, NLDSAR, geomorphological map, BIDR swaths | CPU |
| `02_preprocessing_and_tiling` | Align data, tile into 256x256 patches, geographic train/val/test split | CPU |
| `03_exploratory_data_analysis` | Class distribution, backscatter stats, GLCM texture, spatial autocorrelation | CPU |
| `04_traditional_ml_baseline` | Random Forest on hand-crafted features (performance floor) | CPU |
| `05_resolution_sanity_test` | Multi-resolution robustness and cross-resolution transfer | CPU |
| `06_deep_learning_training` | U-Net, DeepLabV3+ with pretrained encoders | GPU (RunPod) |
| `07_domain_gap_analysis` | Random vs ImageNet vs SAR-pretrained encoder comparison | GPU (RunPod) |
| `08_full_map_generation` | Global terrain classification map + Dragonfly landing region | CPU/GPU |
| `09_evaluation_and_figures` | Publication-quality figures and final metrics | CPU |

## Data Sources

1. **USGS SAR-HiSAR Global Mosaic** — 351 m/pixel, ~1 GB GeoTIFF
2. **NLDSAR Denoised Dataset** — Non-local means denoised SAR (Zenodo)
3. **Lopes et al. (2020) Geomorphological Map** — 6-class ground truth labels
4. **Cassini BIDR Swaths** — 175 m/pixel individual SAR passes (PDS)

## Project Structure

```
titan-sar/
├── notebooks/          # Jupyter pipeline (01-09)
├── data/
│   ├── raw/            # Downloaded files (not in git)
│   ├── processed/      # Tiled patches, aligned labels
│   ├── splits/         # Train/val/test definitions
│   └── predictions/    # Model outputs
├── models/             # Saved model weights
├── configs/            # Training configs (YAML)
├── src/
│   ├── dataset.py      # PyTorch Dataset for Titan SAR
│   ├── transforms.py   # Augmentation pipelines
│   ├── metrics.py      # IoU, F1, confusion matrix
│   ├── train.py        # Standalone GPU training script
│   └── utils.py        # Paths, logging, GeoTIFF I/O, Titan CRS
├── figures/            # Publication-quality plots
├── environment.yml     # Conda environment
└── requirements.txt    # pip requirements
```

## Setup

```bash
# Option A: conda
conda env create -f environment.yml
conda activate titan-sar

# Option B: pip
pip install -r requirements.txt
```

## Hardware

- **CPU work** (Notebooks 01-05, 08-09): Ryzen 3700X, 32 GB RAM, Ubuntu
- **GPU work** (Notebooks 06-07): RunPod, targeting RTX A5000 (~$0.16/hr), budget £10-20
