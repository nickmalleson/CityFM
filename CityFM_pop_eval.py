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
from math import log10

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
from shapely.prepared import prep

import config
from utils import osm_strings
from utils.models import PE
from utils.pre_processing import read_model, poly_to_raster, norm_surface, is_residential, \
    collapse_residential, residential_summaries
from utils.training import Tokenizer, serialize_data
from utils.spatial_cv import spatial_cv_predict, r2

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


def oa_members():
    members = np.empty(len(oa_geoms), dtype=object)
    for n, g in enumerate(oa_geoms):
        x0, y0, x1, y1 = g.bounds
        cand = np.where((ex >= x0) & (ex <= x1) & (ey >= y0) & (ey <= y1))[0]
        pg = prep(g)
        members[n] = [ids[j] for j in cand if pg.contains(Point(ex[j], ey[j]))]
    return members


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


oa_m = cached('oa_members', oa_members)
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

rng = random.Random(0)
polygon_ids = [i for i in ids if entities[i]['type'] == osm_strings.polygon and len(entities[i]['points']) > 3]
is_polygon = set(polygon_ids)
for u in units.values():
    u['sample'] = [rng.sample(p, min(hp.k, len(p))) for p in ([i for i in m if i in is_polygon] for m in u.members)]


# ---- features that need no model

def hand_features(df):
    rows = []
    for m in df.members:
        es = [entities[i] for i in m]
        blds = [e for e in es if e['type'] == osm_strings.polygon and osm_strings.building in e['tags']]
        res = [e for e in blds if is_residential(e)]
        # cap each building at config.max_surface, as the model's own area input does (norm_surface);
        # a few mis-mapped 'houses' cover several km²
        foot = sum(min(e.get('area', 0), config.max_surface) for e in blds)
        rows.append({'log entities': log10(1 + len(es)), 'log buildings': log10(1 + len(blds)),
                     'log houses': log10(1 + len(res)), 'house share': len(res) / max(1, len(blds)),
                     'log footprint m2': log10(1 + foot),
                     'log mean footprint': log10(1 + foot / max(1, len(blds))),
                     'apartments': sum(e['tags'].get(osm_strings.building) == 'apartments' for e in blds),
                     'log POIs': log10(1 + sum(e['type'] == 'node' for e in es))})
    out = pd.DataFrame(rows)
    # a few huge or mis-tagged 'building' polygons exceed their unit, so cap the share at 1
    out['footprint share'] = np.minimum(1, 10 ** out['log footprint m2'] / (df.area_km2.values * 1e6))
    return out.values.astype(float)


def pos_features(df):
    lon, lat = to_wgs(df.x.values, df.y.values)
    return np.stack([PE((la, lo), config.pe_size) for la, lo in zip(lat, lon)])


# ---- model features (as precompute_embs_pop_d, with identical rows encoded once)

summaries = residential_summaries()


def text_features(lm, proj, tok, df, collapse):
    out = []
    with torch.no_grad():
        for m in df.members:
            ents = [summaries[i] if i in summaries else entities[i]
                    for i in (collapse_residential(m, entities) if collapse else m)]
            ids_ = np.array(serialize_data(ents, tok, add_zero=True))
            rows, counts = np.unique(ids_, axis=0, return_counts=True)
            rows = torch.tensor(rows)
            h = lm(rows, attention_mask=(rows != 0).long())[0]
            w = torch.tensor(counts / counts.sum(), dtype=h.dtype)[:, None, None]
            out.append(proj((h * w).sum(0).mean(0)).numpy())
    return np.stack(out)


def visual_features(model, df, bs=128):
    pids = sorted({i for s in df['sample'] for i in s})
    emb = {}
    with torch.no_grad():
        for b in range(0, len(pids), bs):
            batch, imgs, surf = [], [], []
            for i in pids[b:b + bs]:
                try:
                    r = poly_to_raster(Polygon(entities[i]['points']))
                except Exception:
                    continue
                if len(r):
                    batch.append(i)
                    imgs.append(r)
                    surf.append([norm_surface(entities[i]['area'])])
            if not batch:
                continue
            img = model.projection_img(model.cv(torch.tensor(np.array(imgs)).float()).flatten(1))
            s = model.projection_s(torch.tensor(surf).float())
            for i, v in zip(batch, ((img + s) / 2).numpy()):
                emb[i] = v
    zero = np.zeros(model.emb_size, dtype=np.float32)
    return np.stack([np.mean([emb[i] for i in s if i in emb], 0) if any(i in emb for i in s) else zero
                     for s in df['sample']])


# ---- scoring

def score(X, df, seed=0):
    y = np.log10(df.density.values + 1)
    pred = spatial_cv_predict(X, y, df.group.values, seed=seed)
    return {'R2': r2(y, pred), 'Spearman': spearmanr(y, pred).correlation}


results = []
feats = {}
for u, df in units.items():
    tag = u.split()[0].lower()
    f = {'hand': cached('hand_' + tag, lambda: hand_features(df)),
         'pos': cached('pos_' + tag, lambda: pos_features(df)),
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
        t = cached(name, lambda: text_features(bert, torch.nn.Identity(), tok, df, collapse))
        label = 'untuned BERT text' + (' (houses collapsed)' if collapse else '')
        results.append((u, 'baseline', label, score(t, df)))
        results.append((u, 'baseline', label + ' + hand', score(np.hstack([t, feats[u]['hand']]), df)))

for m in hp.m:
    model = read_model(m).eval()  # also sets the raster mode the model was trained with
    tok = Tokenizer(model.lm_name)
    # models trained on collapsed contexts read collapsed units
    collapse = 'residential: few houses' in json.load(open(m + '/Data/entities.json', encoding='utf-8'))
    for u, df in units.items():
        tag = u.split()[0].lower() + '_' + os.path.basename(m.rstrip('/'))
        t = cached('text_' + tag, lambda: text_features(model.lm, model.projection_p, tok, df, collapse))
        v = cached('visual_' + tag, lambda: visual_features(model, df))
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
