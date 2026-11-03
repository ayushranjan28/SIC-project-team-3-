"""
Master Pipeline Orchestrator for Samsung SIC PINN Dataset Acquisition.
Executes the remaining downloads, extraction, verification, signal inspection,
and manifest updates sequentially with full integrity logging.
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.download_datasets import (
    download_file,
    compute_sha256,
    record_checksum,
    RAW_DIR,
    EXTRACTED_DIR,
    MANIFESTS_DIR
)
import scripts.verify_downloads as verifier
import scripts.inspect_datasets as inspector

def step_pamap2():
    print("\n=======================================================")
    print("STEP 1: Verify PAMAP2 Download & Extraction")
    print("=======================================================")
    pamap_zip = RAW_DIR / "PAMAP2" / "pamap2.zip"
    
    if not pamap_zip.exists():
        print("Starting PAMAP2 download from UCI...")
        url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
        download_file(url, pamap_zip, "PAMAP2", max_retries=5)
        
    record_checksum("PAMAP2", pamap_zip)
    
    # Extract if not already extracted
    ext_dir = EXTRACTED_DIR / "PAMAP2"
    if not any(ext_dir.iterdir()):
        verifier.extract_archive(pamap_zip, ext_dir)
        inner_zips = list(ext_dir.rglob("*.zip"))
        for iz in inner_zips:
            print(f"Extracting nested archive {iz.name}...")
            verifier.extract_archive(iz, iz.parent)
            
    res = verifier.verify_dataset_5_pamap2()
    print("PAMAP2 Verified:", res['download_status'], "Recordings:", res['recording_count'])
    inspector.inspect_pamap2()

def step_wesad():
    print("\n=======================================================")
    print("STEP 2: Download, Extract & Verify WESAD")
    print("=======================================================")
    wesad_zip = RAW_DIR / "WESAD" / "wesad.zip"
    if not wesad_zip.exists():
        # Official University of Siegen Sciebo link
        url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
        print(f"Downloading WESAD from University of Siegen ({url})...")
        ok = download_file(url, wesad_zip, "WESAD", max_retries=5)
        if not ok:
            print("Fallback to UCI repository...")
            url_uci = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
            download_file(url_uci, wesad_zip, "WESAD", max_retries=5)
            
    record_checksum("WESAD", wesad_zip)
    
    ext_dir = EXTRACTED_DIR / "WESAD"
    if not any(ext_dir.iterdir()):
        verifier.extract_archive(wesad_zip, ext_dir)
        
    res = verifier.verify_dataset_4_wesad()
    print("WESAD Verified:", res['download_status'], "Subjects:", res['participant_count'])
    inspector.inspect_wesad()

def step_ppg_dalia():
    print("\n=======================================================")
    print("STEP 3: Download, Extract & Verify PPG-DaLiA")
    print("=======================================================")
    dalia_zip = RAW_DIR / "PPG_DaLiA" / "ppg_dalia.zip"
    if not dalia_zip.exists():
        # Official University of Siegen Sciebo link
        url = "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download"
        print(f"Downloading PPG-DaLiA from University of Siegen ({url})...")
        ok = download_file(url, dalia_zip, "PPG_DaLiA", max_retries=5)
        if not ok:
            print("Fallback to UCI repository...")
            url_uci = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
            download_file(url_uci, dalia_zip, "PPG_DaLiA", max_retries=5)
            
    record_checksum("PPG_DaLiA", dalia_zip)
    
    ext_dir = EXTRACTED_DIR / "PPG_DaLiA"
    if not any(ext_dir.iterdir()):
        verifier.extract_archive(dalia_zip, ext_dir)
        
    res = verifier.verify_dataset_1_ppg_dalia()
    print("PPG-DaLiA Verified:", res['download_status'], "Subjects:", res['participant_count'])
    inspector.inspect_ppg_dalia()

def step_final_manifests():
    print("\n=======================================================")
    print("STEP 4: Regenerating Master Manifests & Documentation")
    print("=======================================================")
    verifier.main()
    print("All manifests successfully regenerated.")

def main():
    print("Starting End-to-End Acquisition & Verification Orchestration...")
    step_pamap2()
    step_wesad()
    step_ppg_dalia()
    step_final_manifests()
    print("\nPIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

