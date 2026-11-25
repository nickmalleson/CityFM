"""Area features shared by the population (CityFM_pop_eval.py) and deprivation (CityFM_imd_eval.py) benchmarks.

An area (an Output Area, a grid cell or an LSOA) is described by the list of OSM entity ids inside it. The
model features follow the authors' precompute_embs_pop_d: the projected BERT states of all the area's
entities ('text'), the mean footprint embedding of a sample of its polygons ('visual') and the position
encoding of its centre. The nine simple counts need no model. The ImageNet footprint feature replaces CityFM's
footprint embedding with the image network before CityFM training, as a baseline for it.
"""

import json
from math import log10

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

import config
from utils import osm_strings
from utils.models import PE
from utils.pre_processing import poly_to_raster, norm_surface, is_residential, collapse_residential
from utils.training import serialize_data

HAND_NAMES = ['log entities', 'log buildings', 'log houses', 'house share', 'log footprint m2',
              'log mean footprint', 'apartments', 'log POIs', 'footprint share']

_to_wgs = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True).transform


def members_in_polygons(geoms, ids, ex, ey):
    # The entity ids inside each polygon. geoms are in BNG metres, ex and ey are the entities' BNG coordinates.
    members = np.empty(len(geoms), dtype=object)
    for n, g in enumerate(geoms):
        x0, y0, x1, y1 = g.bounds
        cand = np.where((ex >= x0) & (ex <= x1) & (ey >= y0) & (ey <= y1))[0]
        pg = prep(g)
        members[n] = [ids[j] for j in cand if pg.contains(Point(ex[j], ey[j]))]
    return members


def polygon_ids(entities):
    # Polygons the footprint branch can draw (more than three points).
    return {i for i, e in entities.items() if e['type'] == osm_strings.polygon and len(e['points']) > 3}


def sample_polygons(members, polygons, k, rng):
    # Up to k polygons per area, drawn with rng in the order of members.
    return [rng.sample(p, min(k, len(p))) for p in ([i for i in m if i in polygons] for m in members)]


def hand_features(members, area_km2, entities):
    rows = []
    for m in members:
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
    out['footprint share'] = np.minimum(1, 10 ** out['log footprint m2'] / (np.asarray(area_km2) * 1e6))
    return out[HAND_NAMES].values.astype(float)


def pos_features(x, y):
    # The model's position encoding of points given in BNG metres.
    lon, lat = _to_wgs(np.asarray(x), np.asarray(y))
    return np.stack([PE((la, lo), config.pe_size) for la, lo in zip(lat, lon)])


def text_features(lm, proj, tok, members, entities, summaries, collapse, names):
    # As precompute_embs_pop_d, with identical rows encoded once. names sets whether the tag strings include
    # names; config.names_in_text is restored afterwards, so later features are read with the model's setting.
    previous = config.names_in_text
    config.names_in_text = names
    try:
        out = []
        with torch.no_grad():
            for m in members:
                ents = [summaries[i] if i in summaries else entities[i]
                        for i in (collapse_residential(m, entities) if collapse else m)]
                ids_ = np.array(serialize_data(ents, tok, add_zero=True))
                rows, counts = np.unique(ids_, axis=0, return_counts=True)
                rows = torch.tensor(rows)
                h = lm(rows, attention_mask=(rows != 0).long())[0]
                w = torch.tensor(counts / counts.sum(), dtype=h.dtype)[:, None, None]
                out.append(proj((h * w).sum(0).mean(0)).numpy())
        return np.stack(out)
    finally:
        config.names_in_text = previous


def _mean_footprint_rows(embed, size, samples, entities, bs=128):
    # The mean of embed(images, floor areas) over each area's sampled polygons; zeros if none can be drawn.
    pids = sorted({i for s in samples for i in s})
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
            rows = embed(torch.tensor(np.array(imgs)).float(), torch.tensor(surf).float())
            for i, v in zip(batch, rows.numpy()):
                emb[i] = v
    zero = np.zeros(size, dtype=np.float32)
    return np.stack([np.mean([emb[i] for i in s if i in emb], 0) if any(i in emb for i in s) else zero
                     for s in samples])


def visual_features(model, samples, entities, bs=128):
    # The mean footprint embedding (image and floor area) of each area's sampled polygons; zeros if none can be drawn.
    def embed(img, surf):
        return (model.projection_img(model.cv(img).flatten(1)) + model.projection_s(surf)) / 2
    return _mean_footprint_rows(embed, model.emb_size, samples, entities, bs)


def imagenet_cv():
    # The image network as SSModel builds it before stage 2 trains it: ResNet-18 with ImageNet weights and
    # normalisation, without the classification layer. Section 9a's baseline uses the same network.
    from utils.models import SSModel
    assert config.vision_pretrained, 'config.vision_pretrained is off, so SSModel would start from random weights'
    return SSModel('', config.lm, 768).cv.eval()


def imagenet_features(cv, samples, entities, bs=128):
    # The footprint feature without CityFM training: the mean ImageNet image features (512) of each area's sampled
    # polygons, followed by their mean floor area as CityFM's footprint branch receives it (norm_surface).
    # Zeros if none can be drawn.
    return _mean_footprint_rows(lambda img, surf: torch.cat([cv(img).flatten(1), surf], 1), 513, samples, entities,
                                bs)


def reads_collapsed(model_dir):
    # Models trained on collapsed contexts (CityFM_collapse.py) read areas with their houses collapsed.
    with open(model_dir.rstrip('/') + '/Data/entities.json', encoding='utf-8') as f:
        return 'residential: few houses' in json.load(f)
