"""
Downloader for Dataset 8: MIMIC PERform from Zenodo Record 6950488.
"""

import os
import requests
from pathlib import Path
from download_datasets import download_file, compute_sha256, record_checksum

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw" / "MIMIC_PERform"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*'
}

def download_mimic_perform():
    print("Downloading MIMIC PERform from Zenodo Record 6950488...")
    
    # Priority benchmark files: WFDB and MAT datasets
    target_files = [
        "mimic_perform_train_all_wfdb.zip",
        "mimic_perform_test_all_wfdb.zip",
        "mimic_perform_non_af_wfdb.zip",
        "mimic_perform_af_wfdb.zip",
        "mimic_perform_non_af_data.mat",
        "mimic_perform_af_data.mat"
    ]
    
    for fname in target_files:
        furl = f"https://zenodo.org/records/6950488/files/{fname}?download=1"
        dest = RAW_DIR / fname
        print(f"\nDownloading {fname}...")
        download_file(furl, dest, "MIMIC_PERform", max_retries=5)
        record_checksum("MIMIC_PERform", dest, "N/A")
            
    print("\nMIMIC PERform priority benchmark datasets downloaded successfully.")

if __name__ == "__main__":
    download_mimic_perform()
