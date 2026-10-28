#!/usr/bin/env python3

import argparse
import io
import json
import os
import urllib.parse
import urllib.request
import zipfile

import pandas as pd
from shapely.geometry import shape

parser = argparse.ArgumentParser(description='Build <city>/Data/pop_density.json from Census 2021 (England and Wales)')
parser.add_argument("-c", type=str, required=True, help='City (folder name)')
parser.add_argument("-l", type=str, default=None, help='Local authority name as used in ONS LSOA names (default: city)')

hp = parser.parse_args()
lad = hp.l or hp.c

ONS = 'https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/'
BOUNDARIES = ONS + 'Output_Areas_2021_EW_BFC_V8/FeatureServer/0/query'
TS006 = 'https://www.nomisweb.co.uk/output/census/2021/census2021-ts006.zip'
census_dir = hp.c + '/Census'
os.makedirs(census_dir, exist_ok=True)


def fetch(url, path):
    if not os.path.exists(path):
        print('Downloading', url[:90])
        with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


# Output Area boundaries (full resolution, clipped to the coastline), in pages of 2000
features = []
while True:
    q = urllib.parse.urlencode({'where': "LSOA21NM LIKE '" + lad + " %'", 'outFields': 'OA21CD,LSOA21CD,LSOA21NM,Shape__Area',
                                'outSR': 4326, 'f': 'geojson', 'orderByFields': 'OA21CD',
                                'resultOffset': len(features), 'resultRecordCount': 2000})
    page = json.load(open(fetch(BOUNDARIES + '?' + q, census_dir + '/oa_bfc_' + str(len(features)) + '.geojson')))
    features += page['features']
    if len(page['features']) < 2000:
        break

# Census 2021 TS006: usual residents per square kilometre
with zipfile.ZipFile(fetch(TS006, census_dir + '/ts006.zip')) as z:
    ts006 = pd.read_csv(io.BytesIO(z.read('census2021-ts006-oa.csv')))
density = dict(zip(ts006['geography code'], ts006.iloc[:, -1]))

pop_d = {}
for f in features:
    p = f['properties']
    geom = shape(f['geometry'])
    # a few OAs are multipolygons; keep the largest part (the task code expects one ring)
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda g: g.area)
    pop_d[str(len(pop_d))] = {'polygon': [list(c) for c in geom.exterior.coords],
                              'area': p['Shape__Area'],
                              # thousands of people per square kilometre, as in the shipped cities
                              'population': density[p['OA21CD']] / 1000,
                              'oa': p['OA21CD'], 'lsoa': p['LSOA21CD'],
                              # MSOA name is the LSOA name without its final letter, e.g. "Leeds 011"
                              'msoa': p['LSOA21NM'][:-1]}

with open(hp.c + '/Data/pop_density.json', 'w', encoding='utf-8') as f:
    json.dump(pop_d, f)

print(len(pop_d), 'output areas written to', hp.c + '/Data/pop_density.json')
