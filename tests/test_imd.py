# Tests for the IoD 2025 targets and the LSOA units built from Output Areas (utils.imd). Synthetic data only.
# Run from the repository root: .venv/bin/python -m unittest tests.test_imd
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from pyproj import Transformer

from utils.imd import TARGETS, lsoa_units, read_file7, target_matrix

TO_WGS = Transformer.from_crs('EPSG:27700', 'EPSG:4326', always_xy=True).transform


def square(x0, y0):
    # A 100 m square in BNG metres, as the lon/lat ring that pop_density.json stores.
    return [list(TO_WGS(x, y)) for x, y in [(x0, y0), (x0 + 100, y0), (x0 + 100, y0 + 100), (x0, y0 + 100), (x0, y0)]]


def oa(polygon, density, lsoa, msoa):
    # A pop_density.json entry; density is in thousands of residents per km², as CityFM_uk_pop.py writes it.
    return {'polygon': polygon, 'population': density, 'lsoa': lsoa, 'msoa': msoa}


class TestLsoaUnits(unittest.TestCase):

    def test_lsoa_units_aggregates_output_areas(self):
        pop_d = {'0': oa(square(430000, 433000), 2.0, 'L1', 'M1'), '1': oa(square(430100, 433000), 4.0, 'L1', 'M1'),
                 '2': oa(square(431000, 433000), 1.0, 'L2', 'M1')}
        u = lsoa_units(pop_d, [['a'], ['b', 'c'], []])
        self.assertEqual(list(u.lsoa), ['L1', 'L2'])
        self.assertEqual(list(u.index), [0, 1])
        self.assertEqual(list(u.members[0]), ['a', 'b', 'c'])
        self.assertEqual(list(u.msoa), ['M1', 'M1'])
        np.testing.assert_allclose(u.area_km2, [0.02, 0.01], rtol=1e-3)
        np.testing.assert_allclose(u.population, [60, 10], rtol=1e-3)
        np.testing.assert_allclose(u.density, [3000, 1000], rtol=1e-3)
        np.testing.assert_allclose([u.x[1], u.y[1]], [431050, 433050], atol=0.5)

    def test_lsoa_units_sorted_by_code(self):
        pop_d = {'0': oa(square(430000, 433000), 1.0, 'L2', 'M1'), '1': oa(square(431000, 433000), 1.0, 'L1', 'M2')}
        u = lsoa_units(pop_d, [['a'], ['b']])
        self.assertEqual(list(u.lsoa), ['L1', 'L2'])
        self.assertEqual(list(u.members[0]), ['b'])
        self.assertEqual(list(u.msoa), ['M2', 'M1'])

    def test_lsoa_units_keeps_separate_parts(self):  # Review Focus 1
        pop_d = {'0': oa(square(430000, 433000), 1.0, 'L1', 'M1'), '1': oa(square(430500, 433000), 1.0, 'L1', 'M1')}
        u = lsoa_units(pop_d, [[], []])
        self.assertEqual(u.geom[0].geom_type, 'MultiPolygon')
        np.testing.assert_allclose(u.area_km2[0], 0.02, rtol=1e-3)


class TestTargets(unittest.TestCase):

    def test_targets(self):
        self.assertEqual(len(TARGETS), 14)
        self.assertEqual(list(TARGETS)[0], 'IMD')
        self.assertEqual(list(TARGETS)[-1], 'Outdoors')

    def test_target_matrix_follows_given_order(self):  # Review Focus 5
        imd = {c: {'name': c, 'scores': {k: base + i / 100 for i, k in enumerate(TARGETS)}}
               for c, base in [('B', 2.0), ('A', 1.0)]}
        Y = target_matrix(imd, ['A', 'B'])
        self.assertEqual(Y.shape, (2, 14))
        np.testing.assert_allclose(Y[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(Y[0, 13], 1.13)

    def test_read_file7(self):
        cols = ['LSOA code (2021)', 'LSOA name (2021)', 'Income Rank (where 1 is most deprived)'] + list(TARGETS.values())
        rows = [['A', 'Leeds 001A', 5] + [1.0] * 14, ['B', 'Leeds 001B', 6] + [2.0] * 14]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'file7.csv')
            pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
            df = read_file7(path, ['B', 'A'])
            self.assertEqual(list(df.index), ['B', 'A'])
            self.assertEqual(list(df.columns), ['name'] + list(TARGETS))
            self.assertEqual(df.loc['A', 'name'], 'Leeds 001A')
            self.assertEqual(df.loc['B', 'Education'], 2.0)
            with self.assertRaisesRegex(ValueError, 'Z'):
                read_file7(path, ['A', 'Z'])
            rows[0][3] = None
            pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_file7(path, ['A'])


if __name__ == '__main__':
    unittest.main()
