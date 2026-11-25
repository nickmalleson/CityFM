#!/usr/bin/env python3
"""Compare CityFM models on population density (Census 2021) against simple baselines.

Units are Census Output Areas (as in the shipped pop_density task) and a regular 500 m grid
(population shared out from Output Areas by overlap). Output Areas hold roughly equal numbers of
people, so on them an area's size alone nearly gives its density; the grid removes that shortcut.

Features per unit follow precompute_embs_pop_d: the projected BERT states of all entities in the
unit ('text'), the mean polygon visual embedding ('visual', from a sample of polygons per unit) and
the position encoding of the centroid ('pos'). Each feature set is scored with ridge regression on
log10 density, using spatially grouped cross-validation (MSOAs for Output Areas, 2 km blocks for the grid;
see utils/spatial_cv.py).
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely.geometry import Polygon, box
from shapely.ops import transform

import config
from utils.pre_processing import read_model, residential_summaries
from utils.training import Tokenizer
from utils.spatial_cv import spatial_cv_predict, r2
from utils.unit_features import members_in_polygons, polygon_ids, sample_polygons, hand_features, pos_features, \
    text_features, visual_features, reads_collapsed

parser = argparse.ArgumentParser(description='Population density evaluation')
parser.add_argument("-c", type=str, required=True, help='City with Data/entities.json and Data/pop_density.json')
parser.add_argument("-m", type=str, nargs='+', required=True, help='Model folders (each with Model/model.pkl)')
parser.add_argument("-k", type=int, default=20, help='Polygons sampled per unit for the visual feature')
parser.add_argument("-g", type=int, default=500, help='Grid cell size (m)')
hp = parser.parse_args()

out_dir = hp.c + '/Analysis'
os.makedirs(out_dir, exist_ok=True)
to_bng = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True).transform
to_wgs = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True).transform

with open(hp.c + '/Data/entities.json', encoding='utf-8') as f:
    entities = json.load(f)
with open(hp.c + '/Data/pop_density.json', encoding='utf-8') as f:
    pop_d = json.load(f)

ids = np.array(list(entities))
ex, ey = to_bng(np.array([entities[i]['lon'] for i in ids]), np.array([entities[i]['lat'] for i in ids]))


def cached(name, fn):
    path = out_dir + '/pop_' + name + '.npz'
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)['x']
    t = time.time()
    x = fn()
    np.savez(path, x=x)
    print(name + ': ' + str(np.shape(x)) + ' in ' + str(round(time.time() - t)) + 's', flush=True)
    return x


# ---- units: Output Areas and grid cells, each with a list of entity ids and a target

oa_geoms = [transform(to_bng, Polygon(v['polygon'])) for v in pop_d.values()]


def grid_units():
    # cells fully inside the Output Areas; population from each OA in proportion to the overlap
    s = hp.g
    pop = {}
    cover = {}
    for g, v in zip(oa_geoms, pop_d.values()):
        people = v['population'] * 1000 * g.area / 1e6
        x0, y0, x1, y1 = g.bounds
        for cx in range(int(x0 // s), int(x1 // s) + 1):
            for cy in range(int(y0 // s), int(y1 // s) + 1):
                a = g.intersection(box(cx * s, cy * s, (cx + 1) * s, (cy + 1) * s)).area
                if a > 0:
                    pop[(cx, cy)] = pop.get((cx, cy), 0) + people * a / g.area
                    cover[(cx, cy)] = cover.get((cx, cy), 0) + a
    cells = sorted(c for c in pop if cover[c] >= 0.99 * s * s)
    cell_of = {c: n for n, c in enumerate(cells)}
    members = [[] for _ in cells]
    for j, k in enumerate(zip((ex // s).astype(int), (ey // s).astype(int))):
        if k in cell_of:
            members[cell_of[k]].append(ids[j])
    out = np.empty((len(cells), 4), dtype=object)
    for n, (c, m) in enumerate(zip(cells, members)):
        out[n] = [c[0], c[1], pop[c] / (s * s / 1e6), None]
        out[n, 3] = m
    return out


oa_m = cached('oa_members', lambda: members_in_polygons(oa_geoms, ids, ex, ey))
grid = cached('grid' + str(hp.g), grid_units)

units = {
    'Output Areas': pd.DataFrame({
        'members': list(oa_m),
        'density': [v['population'] * 1000 for v in pop_d.values()],
        'group': [v['msoa'] for v in pop_d.values()],
        'x': [g.centroid.x for g in oa_geoms], 'y': [g.centroid.y for g in oa_geoms],
        'area_km2': [g.area / 1e6 for g in oa_geoms]}),
    str(hp.g) + ' m grid': pd.DataFrame({
        'members': list(grid[:, 3]), 'density': grid[:, 2].astype(float),
        'group': [str(int(cx * hp.g // 2000)) + '_' + str(int(cy * hp.g // 2000)) for cx, cy in grid[:, :2]],
        'x': (grid[:, 0].astype(float) + 0.5) * hp.g, 'y': (grid[:, 1].astype(float) + 0.5) * hp.g,
        'area_km2': hp.g * hp.g / 1e6})}
# the shipped task skips units without entities
for u in units:
    units[u] = units[u][units[u].members.map(len) > 0].reset_index(drop=True)
    print(u + ': ' + str(len(units[u])) + ' units, ' + str(units[u].group.nunique()) + ' CV groups')

# one generator for all units, Output Areas first, so the samples stay as they were
rng = random.Random(0)
is_polygon = polygon_ids(entities)
for u in units.values():
    u['sample'] = sample_polygons(u.members, is_polygon, hp.k, rng)


# ---- model features (utils/unit_features.py, as precompute_embs_pop_d)

summaries = residential_summaries()


# ---- scoring

def score(X, df, seed=0):
    y = np.log10(df.density.values + 1)
    pred = spatial_cv_predict(X, y, df.group.values, seed=seed)
    return {'R2': r2(y, pred), 'Spearman': spearmanr(y, pred).correlation}


results = []
feats = {}
for u, df in units.items():
    tag = u.split()[0].lower()
    f = {'hand': cached('hand_' + tag, lambda: hand_features(df.members, df.area_km2.values, entities)),
         'pos': cached('pos_' + tag, lambda: pos_features(df.x.values, df.y.values)),
         'log area': np.log10(df.area_km2.values.astype(float))[:, None]}
    feats[u] = f
    if u == 'Output Areas':
        results.append((u, 'baseline', 'area only', score(f['log area'], df)))
    results.append((u, 'baseline', 'position only', score(f['pos'], df)))
    results.append((u, 'baseline', 'hand features', score(f['hand'], df)))
    results.append((u, 'baseline', 'hand + position', score(np.hstack([f['hand'], f['pos']]), df)))

from transformers import BertModel
bert = BertModel.from_pretrained(config.lm_names['bert']).eval()
tok = Tokenizer(config.lm)
for u, df in units.items():
    tag = u.split()[0].lower()
    for collapse in [False, True]:
        name = 'bert_text' + ('_collapsed' if collapse else '') + '_' + tag
        # untuned BERT reads names, as it always has here (before this was explicit, it ran before
        # read_model changed config.names_in_text)
        t = cached(name, lambda: text_features(bert, torch.nn.Identity(), tok, df.members, entities, summaries,
                                               collapse, names=True))
        label = 'untuned BERT text' + (' (houses collapsed)' if collapse else '')
        results.append((u, 'baseline', label, score(t, df)))
        results.append((u, 'baseline', label + ' + hand', score(np.hstack([t, feats[u]['hand']]), df)))

for m in hp.m:
    model = read_model(m).eval()  # also sets the raster mode and whether names are read, as the model was trained
    names = config.names_in_text
    tok = Tokenizer(model.lm_name)
    # models trained on collapsed contexts read collapsed units
    collapse = reads_collapsed(m)
    for u, df in units.items():
        tag = u.split()[0].lower() + '_' + os.path.basename(m.rstrip('/'))
        t = cached('text_' + tag, lambda: text_features(model.lm, model.projection_p, tok, df.members, entities,
                                                        summaries, collapse, names))
        v = cached('visual_' + tag, lambda: visual_features(model, df['sample'], entities))
        f = feats[u]
        for label, X in [('text', t), ('visual', v), ('text + visual + position (shipped)', np.hstack([t, v, f['pos']])),
                         ('text + visual + hand', np.hstack([t, v, f['hand']]))]:
            results.append((u, m, label, score(X, df)))

res = pd.DataFrame([{'units': u, 'model': m, 'features': fs, **s} for u, m, fs, s in results])
res.to_csv(out_dir + '/pop_results.csv', index=False)
pd.set_option('display.width', 200)
for u in units:
    print('\n' + u + ' (R2 and Spearman on log10 density, 5-fold spatial CV)')
    print(res[res.units == u].drop(columns='units').round(3).to_string(index=False))
