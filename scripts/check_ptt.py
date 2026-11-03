import requests
import re

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
r = requests.get('https://physionet.org/content/pulse-transit-time-ppg/1.1.0/', headers=headers)
print('All hrefs with download or zip or static:')
for h in re.findall(r'href=["\']([^"\']+)["\']', r.text):
    if any(k in h.lower() for k in ['zip', 'download', 'static', 'files']):
        print(' ', h)

