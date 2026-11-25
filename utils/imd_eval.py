"""Feature sets, feature cache, scoring and the check on combined sets for the deprivation benchmark
(CityFM_imd_eval.py, and section 12 of notebooks/leeds_embeddings.ipynb)."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from utils.spatial_cv import spatial_cv_predict, r2

BASELINE = '9 simple counts + position + density'

# (label, kind) in display order. kind: no network ('simple'), a network without CityFM training ('untuned': untuned
# BERT, or ResNet-18 with ImageNet weights) or CityFM, alone or combined with an untuned network ('cityfm').
SETS = [
    ('Position only', 'simple'),
    ('9 simple counts', 'simple'),
    ('Log population density', 'simple'),
    (BASELINE, 'simple'),
    ('Untuned BERT text', 'untuned'),
    ('Untuned BERT text with names', 'untuned'),
    ('ImageNet footprints', 'untuned'),
    ('ImageNet footprints + position', 'untuned'),
    ('Untuned BERT text + ImageNet footprints + position', 'untuned'),
    ('CityFM text', 'cityfm'),
    ('CityFM footprints', 'cityfm'),
    ('CityFM footprints + position', 'cityfm'),
    ('CityFM text + footprints + position', 'cityfm'),
    # the same combination with untuned BERT's text: what CityFM's text training adds to it
    ('Untuned BERT text + CityFM footprints + position', 'cityfm'),
    ('CityFM all + 9 simple counts + density', 'cityfm'),
]

# (ImageNet set, CityFM set) pairs that differ only in the footprint feature: what CityFM's footprint training adds
FOOTPRINT_PAIRS = [
    ('ImageNet footprints', 'CityFM footprints'),
    ('ImageNet footprints + position', 'CityFM footprints + position'),
    ('Untuned BERT text + ImageNet footprints + position', 'Untuned BERT text + CityFM footprints + position'),
]

# each combined set and the sets whose columns it contains
PARTS = {
    BASELINE: ['Position only', '9 simple counts', 'Log population density'],
    'ImageNet footprints + position': ['ImageNet footprints', 'Position only'],
    'Untuned BERT text + ImageNet footprints + position': ['Untuned BERT text', 'ImageNet footprints + position',
                                                           'ImageNet footprints', 'Position only'],
    'CityFM footprints + position': ['CityFM footprints', 'Position only'],
    'CityFM text + footprints + position': ['CityFM text', 'CityFM footprints + position', 'CityFM footprints',
                                            'Position only'],
    'Untuned BERT text + CityFM footprints + position': ['Untuned BERT text', 'CityFM footprints + position',
                                                         'CityFM footprints', 'Position only'],
    'CityFM all + 9 simple counts + density': [BASELINE, 'CityFM text + footprints + position',
                                               'CityFM footprints + position', 'CityFM text', 'CityFM footprints',
                                               'Position only', '9 simple counts', 'Log population density'],
}


def set_labels(names_in_text):
    # A model that reads names gives the same untuned BERT set twice, so the second is left out.
    return [(l, k) for l, k in SETS if not (names_in_text and l == 'Untuned BERT text with names')]


def cached_rows(path, ids, fn, stamp=None):
    # An (n, d) array cached as an npz with the row ids and an optional stamp (e.g. the size and modification time
    # of the model file); recomputed if the file is absent or its ids or stamp differ.
    path, ids, stamp = Path(path), np.asarray(ids, dtype=str), str(stamp or '')
    if path.exists():
        z = np.load(path)
        if 'stamp' in z and np.array_equal(z['ids'], ids) and str(z['stamp']) == stamp:
            return z['x']
    t = time.time()
    x = np.asarray(fn())
    np.savez(path, ids=ids, x=x, stamp=stamp)
    print(f'{path.name}: {x.shape} in {time.time() - t:.0f}s', flush=True)
    return x


def _score(X, y, groups, repeats):
    with threadpool_limits(1):  # one core per job; jobs run in parallel
        preds = [spatial_cv_predict(X, y, groups, seed=r) for r in range(repeats)]
    return (np.array([r2(y, p) for p in preds]), np.array([spearmanr(y, p).correlation for p in preds]), preds[0])


def score_sets(X_sets, Y, groups, repeats=10, n_jobs=8):
    """Score each feature set on each target with grouped ridge cross-validation, over repeated fold draws.

    Repeat r uses seed r for every set and target, so differences between sets are paired. Returns 'r2' and
    'rho' (Spearman) of shape (sets, targets, repeats) and 'pred0', the first repeat's predictions
    (sets, targets, rows), with sets in X_sets order.
    """
    names = list(X_sets)
    jobs = [(s, t) for s in range(len(names)) for t in range(Y.shape[1])]
    out = Parallel(n_jobs=n_jobs)(delayed(_score)(np.asarray(X_sets[names[s]], dtype=np.float64), Y[:, t], groups,
                                                  repeats) for s, t in jobs)
    S, T = len(names), Y.shape[1]
    res = {'r2': np.zeros((S, T, repeats)), 'rho': np.zeros((S, T, repeats)), 'pred0': np.zeros((S, T, len(Y)))}
    for (s, t), (a, b, p) in zip(jobs, out):
        res['r2'][s, t], res['rho'][s, t], res['pred0'][s, t] = a, b, p
    return res


def combined_flags(r2, labels, targets):
    """Combined sets that score below their best part by more than the spread across repeats.

    The spread is the larger of the two standard deviations across repeats. Such a set's gain is not measured
    by the shared ridge penalty, which suits its many embedding columns and shrinks the rest.
    """
    idx = {l: i for i, l in enumerate(labels)}
    rows = []
    for s, parts in PARTS.items():
        parts = [p for p in parts if p in idx]
        if s not in idx or not parts:
            continue
        for t, target in enumerate(targets):
            best = max(parts, key=lambda p: r2[idx[p], t].mean())
            a, b = r2[idx[s], t], r2[idx[best], t]
            if a.mean() < b.mean() - max(a.std(), b.std()):
                rows.append({'set': s, 'target': target, 'best part': best,
                             'set R²': a.mean(), 'best part R²': b.mean()})
    return pd.DataFrame(rows, columns=['set', 'target', 'best part', 'set R²', 'best part R²'])
