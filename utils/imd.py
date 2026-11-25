"""The English Indices of Deprivation 2025 as targets, and the LSOAs they are given for.

Scores (not ranks) are read from IoD 2025 File 7. LSOAs are built as unions of the Census Output Areas in
<city>/Data/pop_density.json (written by CityFM_uk_pop.py), into which they nest exactly.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform, unary_union

FILE7_URL = ('https://assets.publishing.service.gov.uk/media/691ded56d140bbbaa59a2a7d/'
             'File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv')

# short name -> File 7 column: the IMD, the seven domains and the six sub-domains
TARGETS = {
    'IMD': 'Index of Multiple Deprivation (IMD) Score',
    'Income': 'Income Score (rate)',
    'Employment': 'Employment Score (rate)',
    'Education': 'Education, Skills and Training Score',
    'Health': 'Health Deprivation and Disability Score',
    'Crime': 'Crime Score',
    'Barriers': 'Barriers to Housing and Services Score',
    'Living Environment': 'Living Environment Score',
    'Children and Young People': 'Children and Young People Sub-domain Score',
    'Adult Skills': 'Adult Skills Sub-domain Score',
    'Geographical Barriers': 'Geographical Barriers Sub-domain Score',
    'Wider Barriers': 'Wider Barriers Sub-domain Score',
    'Indoors': 'Indoors Sub-domain Score',
    'Outdoors': 'Outdoors Sub-domain Score',
}

SUBDOMAIN_PARENT = {
    'Children and Young People': 'Education', 'Adult Skills': 'Education',
    'Geographical Barriers': 'Barriers', 'Wider Barriers': 'Barriers',
    'Indoors': 'Living Environment', 'Outdoors': 'Living Environment',
}

CODE, NAME = 'LSOA code (2021)', 'LSOA name (2021)'

_to_bng = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True).transform


def read_file7(path, lsoa_codes):
    # The name and fourteen scores of each given LSOA, in the given order.
    df = pd.read_csv(path).set_index(CODE)
    missing = [c for c in lsoa_codes if c not in df.index]
    if missing:
        raise ValueError(f'{len(missing)} LSOAs not in File 7: {missing[:10]}')
    out = df.loc[list(lsoa_codes), [NAME] + list(TARGETS.values())]
    out.columns = ['name'] + list(TARGETS)
    out.index.name = None
    if out[list(TARGETS)].isna().any().any():
        bad = out.index[out[list(TARGETS)].isna().any(axis=1)].tolist()
        raise ValueError(f'missing scores for {len(bad)} LSOAs: {bad[:10]}')
    return out


def lsoa_units(pop_d, oa_members):
    # One row per LSOA, sorted by code. oa_members is aligned with pop_d.values(), as pop_oa_members.npz is.
    # The area is the sum of the Output Areas' areas, so that gaps between Output Areas (CityFM_uk_pop.py keeps
    # only the largest part of each) do not change it. Population = density (thousands per km²) x area.
    parts = {}
    for v, m in zip(pop_d.values(), oa_members):
        g = transform(_to_bng, Polygon(v['polygon']))
        p = parts.setdefault(v['lsoa'], {'msoa': set(), 'members': [], 'geoms': [], 'area': 0.0, 'population': 0.0})
        p['msoa'].add(v['msoa'])
        p['members'] += list(m)
        p['geoms'].append(g)
        p['area'] += g.area / 1e6
        p['population'] += v['population'] * 1000 * g.area / 1e6
    rows = []
    for code in sorted(parts):
        p = parts[code]
        if len(p['msoa']) != 1:
            raise ValueError(f'LSOA {code} lies in more than one MSOA: {sorted(p["msoa"])}')
        geom = unary_union(p['geoms'])
        rows.append({'lsoa': code, 'msoa': p['msoa'].pop(), 'members': p['members'], 'geom': geom,
                     'x': geom.centroid.x, 'y': geom.centroid.y, 'area_km2': p['area'],
                     'population': p['population'], 'density': p['population'] / p['area']})
    return pd.DataFrame(rows)


def target_matrix(imd, lsoas):
    # (len(lsoas), 14) scores from imd.json's layout, rows in the given order and columns in TARGETS order.
    return np.array([[imd[c]['scores'][k] for k in TARGETS] for c in lsoas], dtype=float)
