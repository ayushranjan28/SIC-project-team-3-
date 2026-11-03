import requests
import re

headers = {'User-Agent': 'Mozilla/5.0'}
for u in ['https://ubi29.informatik.uni-siegen.de/usi/data_ppgdalia.html', 'https://kristofvl.github.io/usi/data_wesad.html']:
    try:
        r = requests.get(u, headers=headers, timeout=10)
        print(f'{u} -> {r.status_code}')
        links = [m.group(1) for m in re.finditer(r'href=["\']([^"\']+)["\']', r.text)]
        for l in links:
            if any(k in l.lower() for k in ['zip', 'download', 'data', 'box', 'drive', 'dropbox', 'uci', 'uni-siegen']):
                print('  ', l)
    except Exception as e:
        print(f'{u} -> {e}')

