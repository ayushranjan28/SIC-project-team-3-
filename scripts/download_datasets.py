"""
Download script for Samsung SIC PINN Wearable Physiological Forecasting Project.
Handles downloading, resume, retries, integrity verification, and manifests logging.
"""

import os
import sys
import time
import hashlib
import csv
import json
import zipfile
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
EXTRACTED_DIR = DATASETS_DIR / "extracted"
MANIFESTS_DIR = DATASETS_DIR / "manifests"

DOWNLOAD_LOG = MANIFESTS_DIR / "download_log.csv"
FILE_CHECKSUMS = MANIFESTS_DIR / "file_checksums.csv"
DATASET_REGISTRY = MANIFESTS_DIR / "dataset_registry.csv"
DATASET_METADATA = MANIFESTS_DIR / "dataset_metadata.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': '*/*'
}

def log_download_event(dataset_name, filename, url, status, bytes_downloaded, total_expected, error_msg=""):
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    header_needed = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "dataset_name", "filename", "url", "status", "bytes_downloaded", "total_expected", "error_msg"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_name,
            filename,
            url,
            status,
            bytes_downloaded,
            total_expected if total_expected is not None else "UNKNOWN",
            error_msg
        ])

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def record_checksum(dataset_name, filepath, expected_hash="N/A"):
    header_needed = not FILE_CHECKSUMS.exists()
    actual_hash = compute_sha256(filepath)
    stat = os.stat(filepath)
    with open(FILE_CHECKSUMS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["dataset_name", "relative_path", "size_bytes", "actual_sha256", "expected_sha256", "status"])
        status = "MATCH" if expected_hash in (actual_hash, "N/A") else "MISMATCH"
        writer.writerow([
            dataset_name,
            os.path.relpath(filepath, BASE_DIR),
            stat.st_size,
            actual_hash,
            expected_hash,
            status
        ])
    return actual_hash

def download_file(url, dest_path, dataset_name, max_retries=5, expected_sha256=None):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded and valid
    if dest_path.exists() and dest_path.stat().st_size > 0:
        if expected_sha256:
            calc_hash = compute_sha256(dest_path)
            if calc_hash == expected_sha256:
                print(f"[{dataset_name}] {dest_path.name} already downloaded and verified.")
                log_download_event(dataset_name, dest_path.name, url, "ALREADY_EXISTS_VERIFIED", dest_path.stat().st_size, dest_path.stat().st_size)
                return True
        else:
            # If zip, check testzip
            if dest_path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        bad = z.testzip()
                        if bad is None:
                            print(f"[{dataset_name}] {dest_path.name} already exists and zip is valid ({dest_path.stat().st_size / 1e6:.2f} MB). Skipping duplicate download.")
                            log_download_event(dataset_name, dest_path.name, url, "ALREADY_EXISTS_VALID", dest_path.stat().st_size, dest_path.stat().st_size)
                            return True
                except Exception:
                    print(f"[{dataset_name}] Existing archive {dest_path.name} is corrupt. Re-downloading...")
            elif dest_path.suffix.lower() in [".dat", ".hea", ".atr", ".csv", ".mat"]:
                # non-zero existing file
                print(f"[{dataset_name}] {dest_path.name} already exists ({dest_path.stat().st_size} bytes).")
                log_download_event(dataset_name, dest_path.name, url, "ALREADY_EXISTS", dest_path.stat().st_size, dest_path.stat().st_size)
                return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{dataset_name}] Downloading {dest_path.name} (Attempt {attempt}/{max_retries})...")
            downloaded = 0
            file_mode = "wb"
            headers = {}
            if "zenodo.org" not in url:
                headers = dict(HEADERS)
            
            # Check if partial file exists for resume
            part_path = dest_path.with_name(dest_path.name + ".part")
            if part_path.exists() and part_path.stat().st_size > 0:
                downloaded = part_path.stat().st_size
                headers['Range'] = f"bytes={downloaded}-"
                file_mode = "ab"
                print(f"[{dataset_name}] Resuming from {downloaded / 1e6:.2f} MB...")

            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code == 416:  # Range not satisfiable, restart from 0
                    headers.pop('Range', None)
                    file_mode = "wb"
                    downloaded = 0
                    r = requests.get(url, headers=headers, stream=True, timeout=30)
                
                if r.status_code in [401, 403]:
                    msg = f"HTTP {r.status_code} Access Restricted / Unauthorized"
                    print(f"[{dataset_name}] {msg}")
                    log_download_event(dataset_name, dest_path.name, url, "FAILED_ACCESS_RESTRICTED", 0, None, msg)
                    return False
                
                r.raise_for_status()
                
                total_length = r.headers.get('content-length')
                if total_length:
                    total_length = int(total_length) + (downloaded if r.status_code == 206 else 0)
                
                last_print = time.time()
                cur_downloaded = downloaded
                with open(part_path, file_mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            cur_downloaded += len(chunk)
                            if time.time() - last_print > 3:
                                pct = f"{cur_downloaded / total_length * 100:.1f}%" if total_length else f"{cur_downloaded / 1e6:.1f}MB"
                                print(f"[{dataset_name}] {dest_path.name}: {cur_downloaded / 1e6:.1f} MB ({pct})", end="\r", flush=True)
                                last_print = time.time()
            
            # Rename part to final
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
            
            final_size = dest_path.stat().st_size
            print(f"\n[{dataset_name}] {dest_path.name} finished: {final_size / 1e6:.2f} MB.")
            
            # Checksum verify if provided
            if expected_sha256:
                calc_hash = compute_sha256(dest_path)
                if calc_hash.lower() != expected_sha256.lower():
                    raise ValueError(f"Checksum mismatch for {dest_path.name}: expected {expected_sha256}, got {calc_hash}")
                print(f"[{dataset_name}] Checksum verified for {dest_path.name}.")
                
            log_download_event(dataset_name, dest_path.name, url, "SUCCESS", final_size, total_length)
            return True

        except Exception as e:
            print(f"\n[{dataset_name}] Error downloading {dest_path.name}: {e}")
            log_download_event(dataset_name, dest_path.name, url, "RETRY_ERROR", 0, None, str(e))
            time.sleep(2 * attempt)
            
    log_download_event(dataset_name, dest_path.name, url, "FAILED", 0, None, f"Failed after {max_retries} attempts")
    return False

def download_dataset_1_ppg_dalia():
    name = "PPG_DaLiA"
    print(f"\n==========================================")
    print(f"Processing Dataset 1: {name}")
    print(f"==========================================")
    # Primary: Official University of Siegen repository (Sciebo)
    url = "https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download"
    dest = RAW_DIR / name / "ppg_dalia.zip"
    res = download_file(url, dest, name)
    if not res:
        print(f"[{name}] Primary source failed, attempting UCI official mirror...")
        fallback = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
        res = download_file(fallback, dest, name)
    return res

def download_dataset_2_ieee_spc_2015():
    name = "IEEE_SPC_2015"
    print(f"\n==========================================")
    print(f"Processing Dataset 2: {name}")
    print(f"==========================================")
    raw_dir = RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # We already cloned the official repository by Dr. Zhilin Zhang
    expected_files = ["competition_data.zip", "Extra_TrainingData.zip", "TestData.zip", "TrueBPM.zip"]
    all_present = True
    for ef in expected_files:
        p = raw_dir / ef
        if not p.exists() or p.stat().st_size == 0:
            all_present = False
            break
            
    if all_present:
        print(f"[{name}] All official IEEE SPC 2015 competition archives present in raw directory.")
        for ef in expected_files:
            p = raw_dir / ef
            log_download_event(name, ef, "https://github.com/zhilinzhang/IEEE-SP-Cup-2015-PPG-Dataset", "SUCCESS", p.stat().st_size, p.stat().st_size)
        return True
    else:
        print(f"[{name}] Cloned repository contents need verification.")
        return False

def download_dataset_3_wrist_ppg_exercise():
    name = "Wrist_PPG_Exercise"
    print(f"\n==========================================")
    print(f"Processing Dataset 3: {name}")
    print(f"==========================================")
    url = "https://physionet.org/static/published-projects/wrist/wrist-ppg-during-exercise-1.0.0.zip"
    dest = RAW_DIR / name / "wrist-ppg-during-exercise-1.0.0.zip"
    return download_file(url, dest, name)

def download_dataset_4_wesad():
    name = "WESAD"
    print(f"\n==========================================")
    print(f"Processing Dataset 4: {name}")
    print(f"==========================================")
    # Primary: Official University of Siegen repository (Sciebo)
    url = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
    dest = RAW_DIR / name / "wesad.zip"
    res = download_file(url, dest, name)
    if not res:
        print(f"[{name}] Primary source failed, attempting UCI official mirror...")
        fallback = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
        res = download_file(fallback, dest, name)
    return res

def download_dataset_5_pamap2():
    name = "PAMAP2"
    print(f"\n==========================================")
    print(f"Processing Dataset 5: {name}")
    print(f"==========================================")
    url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    dest = RAW_DIR / name / "pamap2.zip"
    return download_file(url, dest, name)

def download_dataset_6_pulse_transit_time_ppg():
    name = "Pulse_Transit_Time_PPG"
    print(f"\n==========================================")
    print(f"Processing Dataset 6: {name}")
    print(f"==========================================")
    raw_dir = RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch SHA256SUMS.txt
    sums_url = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/SHA256SUMS.txt?download"
    sums_file = raw_dir / "SHA256SUMS.txt"
    if not sums_file.exists():
        r = requests.get(sums_url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        with open(sums_file, "wb") as f:
            f.write(r.content)
            
    # Parse files and checksums
    file_map = {}
    with open(sums_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                sha, rel_f = parts
                file_map[rel_f.strip()] = sha.strip()
                
    print(f"[{name}] Found {len(file_map)} files in official PhysioNet checksum registry.")
    
    # Download each file
    base_file_url = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/"
    success_count = 0
    for rel_f, sha in file_map.items():
        file_url = f"{base_file_url}{rel_f}?download"
        dest_p = raw_dir / rel_f
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        res = download_file(file_url, dest_p, name, max_retries=3, expected_sha256=sha)
        if res:
            success_count += 1
            
    print(f"[{name}] Downloaded and verified {success_count}/{len(file_map)} files.")
    return success_count == len(file_map)

def check_dataset_7_mimic_iii_ext_ppg():
    name = "MIMIC_III_Ext_PPG"
    print(f"\n==========================================")
    print(f"Checking Access for Dataset 7: {name}")
    print(f"==========================================")
    url = "https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/"
    zip_url = "https://physionet.org/static/published-projects/mimic-iii-ext-ppg/mimic-iii-ext-ppg-1.1.0.zip"
    
    # Check landing page and zip access
    r = requests.get(zip_url, headers=HEADERS, stream=True, timeout=15)
    print(f"[{name}] PhysioNet download attempt returned HTTP {r.status_code}.")
    
    status_desc = "ACCESS_RESTRICTED (Credentialed Access Required)"
    remarks = "Requires PhysioNet credentialed account, CITI training in 'Data or Specimens Only Research', and signed PhysioNet Credentialed Health Data Use Agreement 1.5.0."
    print(f"[{name}] Access Status: {status_desc}")
    print(f"[{name}] Reason: {remarks}")
    
    log_download_event(name, "mimic-iii-ext-ppg-1.1.0.zip", zip_url, "ACCESS_RESTRICTED", 0, None, remarks)
    return False, status_desc, remarks

def download_dataset_8_mimic_perform():
    name = "MIMIC_PERform"
    print(f"\n==========================================")
    print(f"Processing Dataset 8: {name}")
    print(f"==========================================")
    raw_dir = RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zenodo_api = "https://zenodo.org/api/records/6950488"
    r = requests.get(zenodo_api, headers=HEADERS, timeout=20)
    r.raise_for_status()
    zdata = r.json()
    files = zdata.get("files", [])
    
    print(f"[{name}] Zenodo record 6950488 has {len(files)} files.")
    all_ok = True
    for f in files:
        fname = f.get("key")
        furl = f.get("links", {}).get("self")
        fsize = f.get("size")
        dest_p = raw_dir / fname
        res = download_file(furl, dest_p, name, max_retries=4)
        if not res:
            all_ok = False
            
    return all_ok

def check_dataset_9_cardiowave_portable():
    name = "CardioWave_Portable"
    print(f"\n==========================================")
    print(f"Checking Access for Dataset 9: {name}")
    print(f"==========================================")
    zip_url = "https://physionet.org/static/published-projects/cardiowave-portable/cardiowave-portable-1.0.0.zip"
    r = requests.get(zip_url, headers=HEADERS, stream=True, timeout=15)
    print(f"[{name}] PhysioNet download attempt returned HTTP {r.status_code}.")
    
    status_desc = "ACCESS_RESTRICTED (Credentialed Access Required)"
    remarks = "Requires PhysioNet credentialed account, human subjects research compliance, and signed PhysioNet Credentialed Health Data Use Agreement 1.5.0."
    print(f"[{name}] Access Status: {status_desc}")
    print(f"[{name}] Reason: {remarks}")
    
    log_download_event(name, "cardiowave-portable-1.0.0.zip", zip_url, "ACCESS_RESTRICTED", 0, None, remarks)
    return False, status_desc, remarks

def main():
    print("Starting Acquisition Pipeline for Samsung SIC PINN Datasets...")
    
    # 2. IEEE SPC 2015
    download_dataset_2_ieee_spc_2015()
    
    # 3. Wrist PPG Exercise (52 MB)
    download_dataset_3_wrist_ppg_exercise()
    
    # 6. Pulse Transit Time PPG (Open PhysioNet, ~70 MB)
    download_dataset_6_pulse_transit_time_ppg()
    
    # 8. MIMIC PERform (Zenodo Open, ~1.2 GB)
    download_dataset_8_mimic_perform()
    
    # 5. PAMAP2 (~688 MB)
    download_dataset_5_pamap2()
    
    # 4. WESAD (~2.1 GB)
    download_dataset_4_wesad()
    
    # 1. PPG-DaLiA (~2.8 GB)
    download_dataset_1_ppg_dalia()
    
    # 7 & 9 Check Access
    check_dataset_7_mimic_iii_ext_ppg()
    check_dataset_9_cardiowave_portable()
    
    print("\nDownload pipeline execution complete.")

if __name__ == "__main__":
    main()
