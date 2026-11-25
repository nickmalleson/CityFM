#!/usr/bin/env python3
"""Write the English Indices of Deprivation 2025 scores for a city's LSOAs to <city>/Data/imd.json.

The LSOAs are those of the Output Areas in <city>/Data/pop_density.json (CityFM_uk_pop.py). Their scores
come from IoD 2025 File 7 (MHCLG), downloaded once to <city>/IMD/. Each LSOA gets its name and fourteen scores:
the IMD, the seven domains and the six sub-domains (utils/imd.py). Used by CityFM_imd_eval.py.
"""

import argparse
import json
import os
import urllib.request

from utils.imd import FILE7_URL, TARGETS, read_file7

parser = argparse.ArgumentParser(description='IoD 2025 deprivation scores by LSOA')
parser.add_argument("-c", type=str, required=True, help='City with Data/pop_density.json')
hp = parser.parse_args()


def fetch(url, path):
    if not os.path.exists(path):
        print('Downloading', url[:90])
        with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


os.makedirs(hp.c + '/IMD', exist_ok=True)
file7 = fetch(FILE7_URL, hp.c + '/IMD/file7.csv')

with open(hp.c + '/Data/pop_density.json', encoding='utf-8') as f:
    pop_d = json.load(f)
codes = sorted({v['lsoa'] for v in pop_d.values()})

df = read_file7(file7, codes)  # stops if any LSOA or score is missing
imd = {c: {'name': row['name'], 'scores': {k: float(row[k]) for k in TARGETS}} for c, row in df.iterrows()}
path = hp.c + '/Data/imd.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(imd, f, indent=1)
print(len(imd), 'LSOAs written to', path)
