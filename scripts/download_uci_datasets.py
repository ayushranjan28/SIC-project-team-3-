"""
Downloader for UCI physiological datasets:
- PAMAP2 (Physical Activity Monitoring)
- WESAD (Wearable Stress and Affect Detection)
- PPG-DaLiA
"""

import sys
from pathlib import Path
from download_datasets import download_file, compute_sha256, record_checksum

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"

UCI_DATASETS = {
    "PAMAP2": {
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "dest": RAW_DIR / "PAMAP2" / "pamap2.zip"
    },
    "WESAD": {
        "url": "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
        "fallback_url": "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip",
        "dest": RAW_DIR / "WESAD" / "wesad.zip"
    },
    "PPG_DaLiA": {
        "url": "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download",
        "fallback_url": "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
        "dest": RAW_DIR / "PPG_DaLiA" / "ppg_dalia.zip"
    }
}

def download_uci(dataset_name):
    cfg = UCI_DATASETS.get(dataset_name)
    if not cfg:
        print(f"Unknown dataset: {dataset_name}")
        return False
        
    print(f"\n==========================================")
    print(f"Downloading UCI Dataset: {dataset_name}")
    print(f"URL: {cfg['url']}")
    print(f"Destination: {cfg['dest']}")
    print(f"==========================================")
    
    success = download_file(cfg['url'], cfg['dest'], dataset_name, max_retries=5)
    if success:
        record_checksum(dataset_name, cfg['dest'])
    return success

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        download_uci(target)
    else:
        print("Specify dataset name: PAMAP2, WESAD, or PPG_DaLiA")
