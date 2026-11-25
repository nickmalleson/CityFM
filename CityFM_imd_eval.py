#!/usr/bin/env python3
"""Compare CityFM models on deprivation (English Indices of Deprivation 2025) against simple baselines.

The units are the city's LSOAs, built from the Census Output Areas of the population benchmark (utils/imd.py).
The targets are fourteen IoD 2025 scores: the IMD, the seven domains and the six sub-domains, written to
<city>/Data/imd.json by CityFM_uk_imd.py. Each feature set (utils/imd_eval.py) is scored on each target
separately with ridge regression and cross-validation over whole MSOAs (utils/spatial_cv.py), repeated with
different fold draws. The features are those of CityFM_pop_eval.py (utils/unit_features.py), with log
population density added, untuned BERT run both as the model reads text and with names, and the footprint
feature also computed with the image network before CityFM training (ResNet-18 with ImageNet weights).

Writes <city>/Analysis/imd_cv_<model>.npz (every repeat's scores and the first repeat's predictions) and
<city>/Analysis/imd_results.csv (a summary for all models of the run).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

import config
from utils.imd import TARGETS, lsoa_units, target_matrix
from utils.imd_eval import BASELINE, cached_rows, score_sets, set_labels
from utils.pre_processing import read_model, residential_summaries
from utils.training import Tokenizer
from utils.unit_features import members_in_polygons, polygon_ids, sample_polygons, hand_features, pos_features, \
    text_features, visual_features, reads_collapsed, imagenet_cv, imagenet_features

parser = argparse.ArgumentParser(description='Deprivation (IoD 2025) evaluation')
parser.add_argument("-c", type=str, required=True, help='City with Data/entities.json, pop_density.json and imd.json')
parser.add_argument("-m", type=str, nargs='+', required=True, help='Model folders (each with Model/model.pkl)')
parser.add_argument("-k", type=int, default=50, help='Polygons sampled per LSOA for the footprint feature')
parser.add_argument("-r", type=int, default=10, help='Repeats of the cross-validation (seeds 0 to r-1)')
parser.add_argument("-j", type=int, default=8, help='Parallel jobs for the scoring')
hp = parser.parse_args()

out_dir = Path(hp.c) / 'Analysis'
out_dir.mkdir(exist_ok=True)
if not os.path.exists(hp.c + '/Data/imd.json'):
    raise SystemExit(hp.c + '/Data/imd.json not found: run CityFM_uk_imd.py -c ' + hp.c + ' first')

with open(hp.c + '/Data/entities.json', encoding='utf-8') as f:
    entities = json.load(f)
with open(hp.c + '/Data/pop_density.json', encoding='utf-8') as f:
    pop_d = json.load(f)
with open(hp.c + '/Data/imd.json', encoding='utf-8') as f:
    imd = json.load(f)

# ---- units: LSOAs made from the Output Areas of the population benchmark

oa_path = out_dir / 'pop_oa_members.npz'
if oa_path.exists():
    oa_m = np.load(oa_path, allow_pickle=True)['x']
else:
    to_bng = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True).transform
    ids = np.array(list(entities))
    ex, ey = to_bng(np.array([entities[i]['lon'] for i in ids]), np.array([entities[i]['lat'] for i in ids]))
    oa_m = members_in_polygons([transform(to_bng, Polygon(v['polygon'])) for v in pop_d.values()], ids, ex, ey)
    np.savez(oa_path, x=oa_m)

units = lsoa_units(pop_d, oa_m)
empty = units.lsoa[units.members.map(len) == 0].tolist()
assert not empty, 'LSOAs without OSM entities: ' + str(empty)
codes = list(units.lsoa)
Y = target_matrix(imd, codes)
groups = units.msoa.values
print(hp.c + ' LSOAs: ' + str(len(units)) + ' units, ' + str(len(set(groups))) + ' CV groups', flush=True)

# ---- features that need no model

counts = hand_features(units.members, units.area_km2.values, entities)
position = pos_features(units.x.values, units.y.values)
log_density = np.log10(units.density.values)[:, None]
baseline = np.hstack([counts, position, log_density])

summaries = residential_summaries()
polygons = polygon_ids(entities)
bert, bert_tok = None, Tokenizer(config.lm)


def untuned_bert(names, collapse):
    global bert
    if bert is None:
        from transformers import BertModel
        bert = BertModel.from_pretrained(config.lm_names['bert']).eval()
    return cached_rows(out_dir / f'imd_bert_names{int(names)}_collapsed{int(collapse)}.npz', codes,
                       lambda: text_features(bert, torch.nn.Identity(), bert_tok, units.members, entities, summaries,
                                             collapse, names))


imagenet = None


def imagenet_footprints(samples):
    # The same polygons as the CityFM footprints, drawn in the raster mode of the model just read.
    global imagenet
    if imagenet is None:
        imagenet = imagenet_cv()
    return cached_rows(out_dir / f'imd_imagenet_footprints_k{hp.k}_metric{int(config.raster_metric)}.npz', codes,
                       lambda: imagenet_features(imagenet, samples, entities))


# ---- each model

rows = []
tables = {}
for m in hp.m:
    name = os.path.basename(m.rstrip('/'))
    model = read_model(m).eval()  # also sets the raster mode and whether names are read, as the model was trained
    names = config.names_in_text
    collapse = reads_collapsed(m)
    # the model file's size and modification time, so that a model retrained under the same name is re-encoded
    st = os.stat(m.rstrip('/') + '/Model/model.pkl')
    stamp = str(st.st_size) + ':' + str(st.st_mtime)
    print(name + ': names in text ' + str(names) + ', houses collapsed ' + str(collapse), flush=True)
    text = cached_rows(out_dir / f'imd_text_{name}.npz', codes,
                       lambda: text_features(model.lm, model.projection_p, Tokenizer(model.lm_name), units.members,
                                             entities, summaries, collapse, names), stamp)
    samples = sample_polygons(units.members, polygons, hp.k, random.Random(0))
    footprints = cached_rows(out_dir / f'imd_footprints_{name}_k{hp.k}.npz', codes,
                             lambda: visual_features(model, samples, entities), stamp)
    bert_same = untuned_bert(names, collapse)
    imagenet_fp = imagenet_footprints(samples)
    X = {'Position only': position, '9 simple counts': counts, 'Log population density': log_density,
         BASELINE: baseline,
         'Untuned BERT text': bert_same,
         'ImageNet footprints': imagenet_fp, 'ImageNet footprints + position': np.hstack([imagenet_fp, position]),
         'Untuned BERT text + ImageNet footprints + position': np.hstack([bert_same, imagenet_fp, position]),
         'CityFM text': text, 'CityFM footprints': footprints,
         'CityFM footprints + position': np.hstack([footprints, position]),
         'CityFM text + footprints + position': np.hstack([text, footprints, position]),
         'Untuned BERT text + CityFM footprints + position': np.hstack([bert_same, footprints, position]),
         'CityFM all + 9 simple counts + density': np.hstack([text, footprints, position, counts, log_density])}
    if not names:
        X['Untuned BERT text with names'] = untuned_bert(True, collapse)
    labels = set_labels(names)
    X = {l: X[l] for l, _ in labels}

    res = score_sets(X, Y, groups, repeats=hp.r, n_jobs=hp.j)
    np.savez(out_dir / f'imd_cv_{name}.npz', sets=np.array([l for l, _ in labels]),
             kinds=np.array([k for _, k in labels]), targets=np.array(list(TARGETS)), lsoa=np.array(codes), y=Y,
             r2=res['r2'], rho=res['rho'], pred0=res['pred0'], k=hp.k, names_in_text=names)
    for s, (label, kind) in enumerate(labels):
        for t, target in enumerate(TARGETS):
            a = res['r2'][s, t]
            rows.append({'model': name, 'set': label, 'kind': kind, 'target': target, 'r2_mean': a.mean(),
                         'r2_sd': a.std(), 'r2_min': a.min(), 'r2_max': a.max(), 'rho_mean': res['rho'][s, t].mean(),
                         'rho_sd': res['rho'][s, t].std()})
    tables[name] = pd.DataFrame(res['r2'].mean(2), index=[l for l, _ in labels], columns=list(TARGETS))

pd.DataFrame(rows).to_csv(out_dir / 'imd_results.csv', index=False)
pd.set_option('display.width', 250)
for name, tab in tables.items():
    print('\n' + name + ': mean R2 over ' + str(hp.r) + ' repeats of 5-fold CV over MSOAs (' + str(hp.k)
          + ' polygons per LSOA)')
    print(tab.round(2).to_string())
