import requests
import json
import re

headers = {'User-Agent': 'Mozilla/5.0'}
urls = [
    'https://sites.google.com/site/researchbyzhang/ieeespcup2015',
    'https://zenodo.org/api/records/3902710'
]
for u in urls:
    try:
        r = requests.get(u, headers=headers, timeout=10)
        print(f'{u} -> {r.status_code}, len={len(r.text)}')
        if 'zenodo' in u:
            z = r.json()
            for f in z.get('files', []):
                print(f"  Zenodo file: {f.get('key')} ({f.get('size')/1e6:.2f} MB) -> {f.get('links', {}).get('self')}")
        else:
            for line in r.text.splitlines():
                if any(w in line.lower() for w in ['download', 'drive.google', 'dropbox', 'data', 'training', 'test', '.zip', '.mat', '.rar']):
                    if 'http' in line:
                        print('  Site line:', line.strip()[:180])
    except Exception as e:
        print(f'{u} -> {e}')

