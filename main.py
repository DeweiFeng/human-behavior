import os
import requests
import zipfile
from tqdm import tqdm

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"
SAVE_DIR = "/home/dewei/workspace/dewei/dataset/daicwoz/raw"
EXTRACT_DIR = "/home/dewei/workspace/dewei/dataset/daicwoz/extracted"
START_ID = 300
END_ID = 492

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

def download_and_extract_zip(subject_id):
    filename = f"{subject_id}_P.zip"
    url = f"{BASE_URL}/{filename}"
    zip_path = os.path.join(SAVE_DIR, filename)
    extract_path = os.path.join(EXTRACT_DIR, str(subject_id))

    # Skip if already extracted
    if os.path.exists(extract_path):
        print(f"✅ Already extracted: {subject_id}")
        return

    # Download
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True, desc=f"Downloading {filename}"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return

    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Extracted {filename} to {extract_path}")
    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")

# === Batch Process ===
for subject_id in range(START_ID, END_ID + 1):
    download_and_extract_zip(subject_id)
