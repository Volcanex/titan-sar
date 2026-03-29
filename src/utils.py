"""Shared helpers: paths, logging, GeoTIFF I/O."""

import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NLDSAR_DIR = RAW_DIR / "nldsar"

for _d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, PREDICTIONS_DIR, MODELS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Global normalisation stats ────────────────────────────────────────────
SAR_GLOBAL_MEAN = 0.5946
SAR_GLOBAL_STD = 0.2249
NLDSAR_GLOBAL_MEAN = 0.2671
NLDSAR_GLOBAL_STD = 0.2348
# Default (SAR) — kept for backward compat
GLOBAL_MEAN = SAR_GLOBAL_MEAN
GLOBAL_STD = SAR_GLOBAL_STD

# ── Logging ────────────────────────────────────────────────────────────────
def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# ── GeoTIFF I/O ───────────────────────────────────────────────────────────
def read_geotiff(path, band=1):
    """Read a single band from a GeoTIFF. Returns (array, profile)."""
    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile.copy()
    return data, profile


def write_geotiff(path, data, profile, dtype=None):
    """Write a 2-D array as a single-band GeoTIFF."""
    profile = profile.copy()
    if dtype is not None:
        profile["dtype"] = dtype
    profile["count"] = 1
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def save_array(path, arr):
    """Save a numpy array to .npy."""
    np.save(path, arr)


def load_array(path):
    """Load a numpy array from .npy."""
    return np.load(path)

# ── Checksums ──────────────────────────────────────────────────────────────
def sha256_file(path, chunk_size=1 << 20):
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ── Manifest ───────────────────────────────────────────────────────────────
def load_manifest(path=None):
    path = path or RAW_DIR / "MANIFEST.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"files": []}


def save_manifest(manifest, path=None):
    path = path or RAW_DIR / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2))

# ── Titan CRS ──────────────────────────────────────────────────────────────
TITAN_RADIUS_M = 2_575_150.0  # IAU mean radius in metres

TITAN_SIMPLE_CYLINDRICAL_WKT = (
    'PROJCS["Titan_Simple_Cylindrical",'
    'GEOGCS["Titan 2000",'
    'DATUM["D_Titan_2000",'
    f'SPHEROID["Titan_2000_IAU_IAG",{TITAN_RADIUS_M},0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",0.0],'
    'PARAMETER["Standard_Parallel_1",0.0],'
    'UNIT["Meter",1.0]]'
)

# ── Class definitions (Lopes et al. 2020) ─────────────────────────────────
TERRAIN_CLASSES = {
    0: "plains",
    1: "dunes",
    2: "hummocky",
    3: "lakes_seas",
    4: "labyrinth",
    5: "craters",
}

CLASS_COLORS = {
    0: "#F5DEB3",  # wheat — plains
    1: "#EDC951",  # gold — dunes
    2: "#8B4513",  # saddle brown — hummocky
    3: "#1E90FF",  # dodger blue — lakes/seas
    4: "#6B8E23",  # olive drab — labyrinth
    5: "#DC143C",  # crimson — craters
}

NUM_CLASSES = len(TERRAIN_CLASSES)
