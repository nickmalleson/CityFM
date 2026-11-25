# Tests for the deprivation benchmark's feature-set list, feature cache, scoring and check on combined sets
# (utils.imd_eval). Synthetic data only.
# Run from the repository root: .venv/bin/python -m unittest tests.test_imd_eval
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.imd_eval import BASELINE, FOOTPRINT_PAIRS, PARTS, SETS, cached_rows, combined_flags, score_sets, \
    set_labels


class TestSets(unittest.TestCase):

    def test_set_labels(self):  # Review Focus 2
        self.assertEqual(len(set_labels(names_in_text=False)), 15)
        labels = [l for l, _ in set_labels(names_in_text=True)]
        self.assertEqual(len(labels), 14)
        self.assertNotIn('Untuned BERT text with names', labels)

    def test_untrained_text_counterpart(self):
        # the CityFM combination has a counterpart with untuned BERT text, to show what CityFM's text training adds
        labels = [l for l, _ in SETS]
        self.assertIn('CityFM footprints + position', labels)
        self.assertIn('Untuned BERT text + CityFM footprints + position', labels)
        self.assertEqual(labels.index('Untuned BERT text + CityFM footprints + position'),
                         labels.index('CityFM text + footprints + position') + 1)

    def test_untrained_footprint_counterparts(self):
        # each CityFM footprint set has a counterpart with the ImageNet network, which differs only in the
        # footprint feature, to show what CityFM's footprint training adds
        known = {l for l, _ in SETS}
        self.assertEqual(FOOTPRINT_PAIRS, [
            ('ImageNet footprints', 'CityFM footprints'),
            ('ImageNet footprints + position', 'CityFM footprints + position'),
            ('Untuned BERT text + ImageNet footprints + position', 'Untuned BERT text + CityFM footprints + position')])
        for imagenet, cityfm in FOOTPRINT_PAIRS:
            self.assertIn(imagenet, known)
            self.assertIn(cityfm, known)
        for s in ['ImageNet footprints + position', 'Untuned BERT text + ImageNet footprints + position']:
            self.assertIn(s, PARTS)

    def test_kinds_stay_together(self):
        # the figure in section 12.3a draws a line wherever the kind changes, so each kind must be one block
        kinds = [k for _, k in SETS]
        blocks = [k for n, k in enumerate(kinds) if n == 0 or k != kinds[n - 1]]
        self.assertEqual(len(blocks), len(set(blocks)))

    def test_parts_name_known_sets(self):
        known = {l for l, _ in SETS}
        for s, parts in PARTS.items():
            self.assertIn(s, known)
            self.assertTrue(set(parts) <= known)


class TestCachedRows(unittest.TestCase):

    def test_cached_rows_recomputes_when_ids_change(self):  # Review Focus 4
        calls = []

        def fn():
            calls.append(1)
            return np.arange(6.0).reshape(3, 2)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'imd_x.npz'
            a = cached_rows(p, ['A', 'B', 'C'], fn)
            b = cached_rows(p, ['A', 'B', 'C'], fn)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(a, b)
            cached_rows(p, ['B', 'A', 'C'], fn)
            self.assertEqual(len(calls), 2)

    def test_cached_rows_recomputes_when_stamp_changes(self):  # a model retrained under the same folder name
        calls = []

        def fn():
            calls.append(1)
            return np.zeros((2, 2))

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'imd_x.npz'
            cached_rows(p, ['A', 'B'], fn, stamp='100:1.0')
            cached_rows(p, ['A', 'B'], fn, stamp='100:1.0')
            self.assertEqual(len(calls), 1)
            cached_rows(p, ['A', 'B'], fn, stamp='100:2.0')
            self.assertEqual(len(calls), 2)
            cached_rows(p, ['A', 'B'], fn)  # no stamp given: a cache with a stamp is not reused
            self.assertEqual(len(calls), 3)


class TestScoring(unittest.TestCase):

    def test_score_sets(self):
        rng = np.random.default_rng(0)
        n, groups = 200, np.repeat(np.arange(40), 5)
        good, noise = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        Y = np.column_stack([good @ [1.0, -2.0, 0.5] + 0.1 * rng.normal(size=n), rng.normal(size=n)])
        out = score_sets({'good': good, 'noise': noise, 'good again': good}, Y, groups, repeats=3, n_jobs=1)
        self.assertEqual(out['r2'].shape, (3, 2, 3))
        self.assertEqual(out['rho'].shape, (3, 2, 3))
        self.assertEqual(out['pred0'].shape, (3, 2, 200))
        self.assertGreater(out['r2'][0, 0].min(), 0.9)
        self.assertGreater(out['rho'][0, 0].min(), 0.9)
        self.assertLess(out['r2'][1, 0].max(), 0.2)
        np.testing.assert_array_equal(out['r2'][0], out['r2'][2])  # same seeds, same folds: paired

    def test_combined_flags(self):
        labels = ['Position only', '9 simple counts', 'Log population density', BASELINE]
        r2 = np.array([[[0.20, 0.20, 0.20]], [[0.60, 0.62, 0.58]], [[0.30, 0.30, 0.30]], [[0.50, 0.51, 0.49]]])
        f = combined_flags(r2, labels, ['IMD'])
        self.assertEqual(len(f), 1)
        self.assertEqual((f.iloc[0]['set'], f.iloc[0]['best part']), (BASELINE, '9 simple counts'))
        r2[3, 0] = [0.59, 0.60, 0.61]  # within the spread of its best part
        self.assertEqual(len(combined_flags(r2, labels, ['IMD'])), 0)

    def test_combined_flags_ignores_absent_sets(self):  # Review Focus 2
        labels = ['Position only', 'CityFM text', 'CityFM footprints', 'CityFM text + footprints + position']
        r2 = np.array([[[0.2, 0.2]], [[0.6, 0.6]], [[0.3, 0.3]], [[0.4, 0.4]]])
        f = combined_flags(r2, labels, ['IMD'])
        self.assertEqual(list(f['set']), ['CityFM text + footprints + position'])
        self.assertEqual(list(f['best part']), ['CityFM text'])


if __name__ == '__main__':
    unittest.main()
