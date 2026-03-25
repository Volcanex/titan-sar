"""Execute Notebook 01 data acquisition steps as a script."""
import sys, os, json, hashlib, datetime, zipfile, re
from pathlib import Path

# Setup paths
os.chdir('/home/gabriel/titan-sar')
sys.path.insert(0, '.')

from src.utils import RAW_DIR, sha256_file, save_manifest, get_logger
import requests
from tqdm import tqdm

log = get_logger('01_data_acquisition')
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest_path, description="", chunk_size=1 << 20):
    dest_path = Path(dest_path)
    if dest_path.exists():
        log.info(f'Already exists: {dest_path.name} ({dest_path.stat().st_size:,} bytes)')
        return dest_path

    log.info(f'Downloading: {description or url}')
    headers = {}
    partial = dest_path.with_suffix(dest_path.suffix + '.partial')
    resume_pos = 0
    if partial.exists():
        resume_pos = partial.stat().st_size
        headers['Range'] = f'bytes={resume_pos}-'
        log.info(f'Resuming from {resume_pos:,} bytes')

    resp = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0)) + resume_pos
    mode = 'ab' if resume_pos else 'wb'

    with open(partial, mode) as f:
        with tqdm(total=total, initial=resume_pos, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    partial.rename(dest_path)
    log.info(f'Saved: {dest_path.name} ({dest_path.stat().st_size:,} bytes)')
    return dest_path

# ── 1. USGS SAR-HiSAR Global Mosaic ──────────────────────────────────
print("\n" + "="*60)
print("1. USGS SAR-HiSAR Global Mosaic")
print("="*60)

SAR_MOSAIC_FILENAME = "Titan_SAR_HiSAR_Global_Mosaic_351m.tif"
sar_mosaic_path = RAW_DIR / SAR_MOSAIC_FILENAME

SAR_URLS = [
    "https://planetarymaps.usgs.gov/mosaic/Titan/Titan_SAR_HiSAR_Global_Mosaic_351m.tif",
    "https://astropedia.astrogeology.usgs.gov/download/Titan/Cassini/SAR/Titan_SAR_HiSAR_Global_Mosaic_351m.tif",
]

for url in SAR_URLS:
    try:
        download_file(url, sar_mosaic_path, description="USGS SAR-HiSAR Global Mosaic")
        break
    except Exception as e:
        log.warning(f"Failed with {url}: {e}")
else:
    log.error("Could not download SAR mosaic from any URL")

# Verify
if sar_mosaic_path.exists():
    import rasterio
    with rasterio.open(sar_mosaic_path) as src:
        print(f'  CRS:        {src.crs}')
        print(f'  Resolution: {src.res}')
        print(f'  Shape:      {src.height} x {src.width}')
        print(f'  Bounds:     {src.bounds}')
        print(f'  Dtype:      {src.dtypes}')
        print(f'  NoData:     {src.nodata}')

# ── 2. NLDSAR Denoised Dataset ────────────────────────────────────────
print("\n" + "="*60)
print("2. NLDSAR Denoised Dataset (Zenodo)")
print("="*60)

NLDSAR_ZENODO_RECORD = "528545"
NLDSAR_API_URL = f"https://zenodo.org/api/records/{NLDSAR_ZENODO_RECORD}"

nldsar_dir = RAW_DIR / "nldsar"
nldsar_dir.mkdir(exist_ok=True)

try:
    resp = requests.get(NLDSAR_API_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    print(f"Title: {record['metadata']['title']}")
    print(f"DOI:   {record['doi']}")
    print(f"Files: {len(record['files'])}")

    for f in record['files']:
        size_mb = f['size'] / 1e6
        print(f"  {f['key']:40s}  {size_mb:8.1f} MB")
        dest = nldsar_dir / f['key']
        try:
            download_file(f['links']['self'], dest, description=f"NLDSAR: {f['key']}")
        except Exception as e:
            log.warning(f"Failed to download {f['key']}: {e}")
except Exception as e:
    log.warning(f"Could not fetch Zenodo record: {e}")

# ── 3. Geomorphological Map ───────────────────────────────────────────
print("\n" + "="*60)
print("3. Geomorphological Map (USGS)")
print("="*60)

geomorph_dir = RAW_DIR / 'geomorphology'
geomorph_dir.mkdir(exist_ok=True)

GEOMORPH_URLS = [
    ("USGS Titan Geologic Map GIS",
     "https://astropedia.astrogeology.usgs.gov/download/Titan/Geology/Titan_global_geology_GIS.zip"),
    ("USGS Titan Geologic Map (SIM-3414) PDF",
     "https://pubs.usgs.gov/sim/3414/sim3414_map.pdf"),
]

for name, url in GEOMORPH_URLS:
    dest = geomorph_dir / Path(url).name
    try:
        download_file(url, dest, description=name)
    except Exception as e:
        log.warning(f"Failed: {name}: {e}")

# Extract zip if downloaded
gis_zip = geomorph_dir / 'Titan_global_geology_GIS.zip'
if gis_zip.exists():
    extract_dir = geomorph_dir / 'usgs_geology'
    if not extract_dir.exists():
        with zipfile.ZipFile(gis_zip) as zf:
            print(f"Extracting {gis_zip.name}:")
            for info in zf.infolist():
                print(f"  {info.filename:50s}  {info.file_size:>10,} bytes")
            zf.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
    else:
        print(f"Already extracted: {extract_dir}")

# List what we found
shapefiles = list(geomorph_dir.rglob('*.shp'))
tiffiles = list(geomorph_dir.rglob('*.tif'))
print(f"\nShapefiles found: {len(shapefiles)}")
for s in shapefiles:
    print(f"  {s}")
print(f"GeoTIFFs found: {len(tiffiles)}")
for t in tiffiles:
    print(f"  {t}")

# Inspect shapefiles
if shapefiles:
    import fiona
    for shp in shapefiles:
        print(f"\nShapefile: {shp.name}")
        try:
            with fiona.open(shp) as src:
                print(f"  CRS:      {src.crs}")
                print(f"  Schema:   {src.schema}")
                print(f"  Features: {len(src)}")
                # Unique values in key fields
                fields_to_check = {}
                for feat in src:
                    for key, val in feat['properties'].items():
                        fields_to_check.setdefault(key, set()).add(val)
                for key, vals in fields_to_check.items():
                    if len(vals) <= 20:
                        print(f"  Field '{key}': {sorted(vals, key=str)}")
                    else:
                        print(f"  Field '{key}': {len(vals)} unique values")
        except Exception as e:
            log.warning(f"Could not read {shp}: {e}")

# ── 4. BIDR Swaths ───────────────────────────────────────────────────
print("\n" + "="*60)
print("4. BIDR Swaths (PDS) — checking availability")
print("="*60)

PDS_BASE_URL = "https://pds-imaging.jpl.nasa.gov/data/cassini/cassini_orbiter/"

BIDR_TARGETS = [
    {'volume': 'CORADR_0048', 'flyby': 'T8', 'desc': 'Shangri-La dune field'},
    {'volume': 'CORADR_0211', 'flyby': 'T83', 'desc': 'Selk Crater region'},
]

bidr_dir = RAW_DIR / 'bidr'
bidr_dir.mkdir(exist_ok=True)

for target in BIDR_TARGETS:
    vol = target['volume']
    bidr_url = f"{PDS_BASE_URL}{vol}/DATA/BIDR/"
    print(f"\nChecking {vol} ({target['flyby']}: {target['desc']})...")
    try:
        resp = requests.get(bidr_url, timeout=30)
        resp.raise_for_status()
        img_files = re.findall(r'href="(BI[^"]*\.IMG)"', resp.text, re.IGNORECASE)
        s01_files = [f for f in img_files if 'S01' in f.upper()]
        print(f"  Found {len(img_files)} .IMG files, {len(s01_files)} S01 primary beam files")
        if s01_files:
            for f in s01_files[:3]:
                print(f"    {f}")
            if len(s01_files) > 3:
                print(f"    ... and {len(s01_files)-3} more")
            # Download just the first S01 file
            fname = s01_files[0]
            vol_dir = bidr_dir / vol
            vol_dir.mkdir(exist_ok=True)
            file_url = f"{bidr_url}{fname}"
            dest = vol_dir / fname
            try:
                download_file(file_url, dest, description=f"{vol}/{fname}")
            except Exception as e:
                log.warning(f"Failed to download {fname}: {e}")
    except Exception as e:
        log.warning(f"Could not access {bidr_url}: {e}")

# ── 5. Build Manifest ────────────────────────────────────────────────
print("\n" + "="*60)
print("5. Building MANIFEST.json")
print("="*60)

manifest = {
    'created': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'description': 'Titan SAR terrain classification — raw data provenance',
    'files': []
}

for fpath in sorted(RAW_DIR.rglob('*')):
    if fpath.is_file() and fpath.name != 'MANIFEST.json':
        rel_path = str(fpath.relative_to(RAW_DIR))
        size = fpath.stat().st_size

        # Skip checksum for very large files
        if size < 2e9:
            checksum = sha256_file(fpath)
        else:
            checksum = 'skipped-large-file'

        manifest['files'].append({
            'path': rel_path,
            'size_bytes': size,
            'sha256': checksum,
            'access_date': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d'),
        })

save_manifest(manifest)
print(f"Manifest saved with {len(manifest['files'])} files:")
for f in manifest['files']:
    print(f"  {f['path']:60s}  {f['size_bytes']:>12,} bytes")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DATA ACQUISITION SUMMARY")
print("="*60)

checks = {
    'SAR Mosaic': sar_mosaic_path.exists(),
    'NLDSAR Data': any(nldsar_dir.iterdir()) if nldsar_dir.exists() else False,
    'Geomorphological Map': bool(shapefiles) or bool(tiffiles),
    'BIDR Swaths': any(bidr_dir.rglob('*.IMG')) if bidr_dir.exists() else False,
}

for item, ok in checks.items():
    status = 'OK' if ok else 'MISSING'
    print(f"  [{status:7s}] {item}")
