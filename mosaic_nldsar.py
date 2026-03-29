"""Mosaic 28 NLDSAR swath files into a global GeoTIFF aligned with the USGS SAR mosaic."""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import time

SAR_PATH = Path("data/raw/Titan_SAR_HiSAR_Global_Mosaic_351m.tif")
NLDSAR_DIR = Path("data/raw/nldsar")
OUT_PATH = Path("data/processed/nldsar_mosaic_aligned.tif")

# Use the USGS SAR mosaic as the target grid
print("Reading target grid from USGS SAR mosaic...")
with rasterio.open(SAR_PATH) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height
    ref_profile = ref.profile.copy()

print(f"Target grid: {ref_width}x{ref_height}, CRS: {ref_crs}")

# Accumulator arrays: sum and count for averaging overlaps
mosaic_sum = np.zeros((ref_height, ref_width), dtype=np.float64)
mosaic_count = np.zeros((ref_height, ref_width), dtype=np.int32)

swaths = sorted(NLDSAR_DIR.glob("*.cub"))
swaths = [s for s in swaths if "MOSAIC" not in s.name]
print(f"Found {len(swaths)} swath files")

t0 = time.time()
for i, swath_path in enumerate(swaths):
    print(f"  [{i+1}/{len(swaths)}] {swath_path.name}...", end=" ", flush=True)
    try:
        with rasterio.open(swath_path) as src:
            # Read source data
            src_data = src.read(1)
            nodata_val = src.nodata
            
            # Create destination array
            dst_data = np.full((ref_height, ref_width), np.nan, dtype=np.float64)
            
            # Reproject swath to target grid
            reproject(
                source=src_data.astype(np.float64),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                src_nodata=nodata_val,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            
            # Accumulate valid pixels
            valid = ~np.isnan(dst_data) & (dst_data > -1e30) & (dst_data < 1e30)
            mosaic_sum[valid] += dst_data[valid]
            mosaic_count[valid] += 1
            
            n_valid = valid.sum()
            print(f"{n_valid:,} pixels")
    except Exception as e:
        print(f"FAILED: {e}")

elapsed = time.time() - t0
print(f"\nMosaicking done in {elapsed:.0f}s")

# Average overlapping regions
print("Averaging overlaps...")
mosaic = np.full((ref_height, ref_width), np.float32(0), dtype=np.float32)
has_data = mosaic_count > 0
mosaic[has_data] = (mosaic_sum[has_data] / mosaic_count[has_data]).astype(np.float32)

coverage = has_data.sum() / (ref_height * ref_width) * 100
print(f"Coverage: {has_data.sum():,} pixels ({coverage:.1f}%)")
print(f"Value range: {mosaic[has_data].min():.4f} to {mosaic[has_data].max():.4f}")
print(f"Mean: {mosaic[has_data].mean():.4f}, Std: {mosaic[has_data].std():.4f}")

# Save
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
ref_profile.update(dtype="float32", count=1, nodata=0, compress="lzw")
with rasterio.open(OUT_PATH, "w", **ref_profile) as dst:
    dst.write(mosaic, 1)

print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)")
