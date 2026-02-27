import os
import zipfile
import traceback
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

datasets_dir = r'c:\Users\anjum\OneDrive\Desktop\Full Stack ML Project\datasets'
os.makedirs(datasets_dir, exist_ok=True)

def download_dataset(slug, target_dir, label):
    """Try to download a dataset, handle errors gracefully."""
    os.makedirs(target_dir, exist_ok=True)
    try:
        print(f"\n  Downloading {label}: {slug}...")
        api.dataset_download_files(slug, path=target_dir, unzip=True)
        # Unzip any remaining zip files
        for f in os.listdir(target_dir):
            if f.endswith('.zip'):
                zp = os.path.join(target_dir, f)
                with zipfile.ZipFile(zp, 'r') as z:
                    z.extractall(target_dir)
                os.remove(zp)
        files = os.listdir(target_dir)
        print(f"  SUCCESS! Found {len(files)} files: {', '.join(files)}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

# ============================================================
# Dataset 1: F1 World Championship -> REGRESSION (lap times)
# ============================================================
print("=" * 60)
print("DATASET 1 (REGRESSION): F1 World Championship")
print("=" * 60)
wc_dir = os.path.join(datasets_dir, 'f1_world_championship')
if os.path.exists(wc_dir) and len(os.listdir(wc_dir)) > 2:
    files = os.listdir(wc_dir)
    print(f"  Already downloaded! {len(files)} files found.")
else:
    download_dataset('rohanrao/formula-1-world-championship-1950-2020', wc_dir, 'F1 World Championship')

# ============================================================
# Dataset 2: F1 Race Results -> CLASSIFICATION (podium)
# ============================================================
print("\n" + "=" * 60)
print("DATASET 2 (CLASSIFICATION): F1 Race Results")
print("=" * 60)
cls_dir = os.path.join(datasets_dir, 'f1_classification')

# Try multiple dataset slugs in order
cls_slugs = [
    'melissamonfared/formula-1-race-data-1950-2024',
    'dubradave/formula-1-drivers-dataset',
    'cjgdev/formula-1-race-data-19502017',
]
for slug in cls_slugs:
    if download_dataset(slug, cls_dir, 'Classification'):
        break

# ============================================================
# Dataset 3: F1 Driver Data -> CLUSTERING (driving style)
# ============================================================
print("\n" + "=" * 60)
print("DATASET 3 (CLUSTERING): F1 Driver/Telemetry Data")
print("=" * 60)
clust_dir = os.path.join(datasets_dir, 'f1_clustering')

clust_slugs = [
    'dubradave/formula-1-drivers-dataset',
    'alexjr2001/formula-1-dataset-2020-2025',
    'anandaramg/formula-1-race-data-and-telemetry-updatable',
]
for slug in clust_slugs:
    if download_dataset(slug, clust_dir, 'Clustering'):
        break

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE!")
print("=" * 60)

# Summary
for name, d in [('Regression', wc_dir), ('Classification', cls_dir), ('Clustering', clust_dir)]:
    if os.path.exists(d):
        files = [f for f in os.listdir(d) if f.endswith('.csv')]
        print(f"\n{name}: {d}")
        for f in files:
            size = os.path.getsize(os.path.join(d, f))
            print(f"  {f} ({size/1024:.1f} KB)")
