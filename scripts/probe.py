import requests
import json
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

print('=== 1. Pulse Transit Time PPG ===')
try:
    r = requests.get('https://physionet.org/content/pulse-transit-time-ppg/1.1.0/', headers=headers)
    links = [m.group(1) for m in re.finditer(r'href=["\']([^"\']+\.zip)["\']', r.text)]
    print('Zip links:', links)
    for line in r.text.splitlines():
        if any(w in line.lower() for w in ['access policy', 'license', 'open']):
            print('  Policy line:', line.strip()[:100])
except Exception as e:
    print('PTT error:', e)

print('=== 2. MIMIC-III-Ext-PPG Access Policy ===')
try:
    r2 = requests.get('https://physionet.org/content/mimic-iii-ext-ppg/1.1.0/', headers=headers)
    for line in r2.text.splitlines():
        if any(k in line.lower() for k in ['credentialed', 'access policy', 'duas', 'restricted', 'license']):
            print('  MIMIC-III-Ext-PPG:', line.strip()[:120])
except Exception as e:
    print('MIMIC-III error:', e)

print('=== 3. CardioWave-Portable Access Policy ===')
try:
    r3 = requests.get('https://physionet.org/content/cardiowave-portable/1.0.0/', headers=headers)
    for line in r3.text.splitlines():
        if any(k in line.lower() for k in ['credentialed', 'access policy', 'duas', 'restricted', 'license']):
            print('  CardioWave:', line.strip()[:120])
except Exception as e:
    print('CardioWave error:', e)

print('=== 4. MIMIC PERform Zenodo Files ===')
try:
    r4 = requests.get('https://zenodo.org/api/records/6950488', headers=headers)
    zdata = r4.json()
    print('Title:', zdata.get('metadata', {}).get('title'))
    for f in zdata.get('files', []):
        print(f"  File: {f.get('key')} | Size: {f.get('size')} ({f.get('size')/1e6:.2f} MB) | URL: {f.get('links', {}).get('self')}")
except Exception as e:
    print('Zenodo error:', e)

print('=== 5. IEEE SP Cup 2015 ===')
try:
    r5 = requests.get('https://sites.google.com/view/ieee-spcup-2015', headers=headers)
    sp_links = [m.group(1) for m in re.finditer(r'href=["\']([^"\']+)["\']', r5.text)]
    print('IEEE SPC relevant links:')
    for l in sp_links:
        if any(k in l.lower() for k in ['data', 'download', 'drive.google', 'dropbox', 'box.com', 'ieee', 'spcup', 'competition']):
            print('  ', l)
except Exception as e:
    print('IEEE error:', e)

