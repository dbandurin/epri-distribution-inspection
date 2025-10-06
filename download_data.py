# download_data.py

# Download EPRI Distribution Inspection Imagery from Azure Blob Storage

import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# Azure Blob Storage configuration
STORAGE_ACCOUNT_NAME = "publicstorageaccnt"
CONTAINER_NAME = "drone-distribution-inspection-imagery"
BASE_URL = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}"

# Files to download
FILES_TO_DOWNLOAD = [
    "Circuit1.zip",
    "Circuit10.zip",
    "Overhead-Distribution-Labels.csv",
]

OUTPUT_DIR = Path("data/raw")

def download_file(url, output_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def download_from_azure():
    """Download all files from Azure Blob Storage"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting download from Azure Blob Storage...")
    
    for filename in FILES_TO_DOWNLOAD:
        url = f"{BASE_URL}/{filename}"
        output_path = OUTPUT_DIR / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
        
        try:
            print(f"\\nDownloading {filename}...")
            download_file(url, output_path)
            print(f"✓ Downloaded {filename}")
            
            # Extract zip files
            if filename.endswith('.zip'):
                extract_path = OUTPUT_DIR / "images"
                extract_path.mkdir(exist_ok=True)
                
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"✓ Extracted {filename}")
                
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("EPRI Distribution Inspection Imagery - Data Download")
    print("=" * 60)
    
    download_from_azure()
    
    print("\\n" + "=" * 60)
    print("Download complete!")
    print(f"Files saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
