# Tests for the area features shared by the population and deprivation benchmarks (utils.unit_features).
# They compare the features with the population benchmark's caches in Leeds/Analysis, so they are skipped
# where those caches do not exist.
# Run from the repository root: .venv/bin/python -m unittest tests.test_unit_features
import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

import config
from utils.pre_processing import read_model, residential_summaries
from utils.training import Tokenizer
from utils.unit_features import hand_features, imagenet_cv, imagenet_features, polygon_ids, pos_features, \
    reads_collapsed, sample_polygons, text_features, visual_features

ANALYSIS = Path('Leeds/Analysis')
CACHES = ['pop_oa_members', 'pop_hand_output', 'pop_pos_output', 'pop_bert_text_output',
          'pop_bert_text_collapsed_output', 'pop_visual_output_Leeds_v5']


def cache(name):
    return np.load(ANALYSIS / f'{name}.npz', allow_pickle=True)['x']


@unittest.skipUnless(all((ANALYSIS / f'{c}.npz').exists() for c in CACHES), 'population caches not found')
class TestAgainstPopulationCaches(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The Output Area units as CityFM_pop_eval.py builds them, units without entities dropped.
        from transformers import BertModel
        to_bng = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True).transform
        with open('Leeds/Data/pop_density.json', encoding='utf-8') as f:
            pop_d = json.load(f)
        geoms = [transform(to_bng, Polygon(v['polygon'])) for v in pop_d.values()]
        keep = [n for n, m in enumerate(cache('pop_oa_members')) if len(m) > 0]
        members = cache('pop_oa_members')
        cls.members = [members[n] for n in keep]
        cls.area = np.array([geoms[n].area / 1e6 for n in keep])
        cls.x = np.array([geoms[n].centroid.x for n in keep])
        cls.y = np.array([geoms[n].centroid.y for n in keep])
        with open('Leeds/Data/entities.json', encoding='utf-8') as f:
            cls.entities = json.load(f)
        cls.bert = BertModel.from_pretrained(config.lm_names['bert']).eval()
        cls.tok = Tokenizer(config.lm)

    def test_hand_features_match_cache(self):
        np.testing.assert_allclose(hand_features(self.members, self.area, self.entities),
                                   cache('pop_hand_output').astype(float), atol=1e-9)

    def test_pos_features_match_cache(self):
        np.testing.assert_allclose(pos_features(self.x, self.y), cache('pop_pos_output'), atol=1e-9)

    def test_bert_text_matches_cache(self):
        # the population benchmark's BERT always read names
        for collapse, name in [(False, 'pop_bert_text_output'), (True, 'pop_bert_text_collapsed_output')]:
            out = text_features(self.bert, torch.nn.Identity(), self.tok, self.members[:20], self.entities,
                                residential_summaries(), collapse, names=True)
            np.testing.assert_allclose(out, cache(name)[:20], atol=1e-5)

    def test_visual_features_match_cache(self):
        model = read_model('Leeds_v5').eval()
        samples = sample_polygons(self.members, polygon_ids(self.entities), 20, random.Random(0))
        np.testing.assert_allclose(visual_features(model, samples[:5], self.entities),
                                   cache('pop_visual_output_Leeds_v5')[:5], atol=1e-4)

    def test_text_features_restores_names_setting(self):
        ents = {'p': {'type': 'node', 'tags': {'name': 'The Swan', 'amenity': 'pub'}}}
        config.names_in_text = True
        without = text_features(self.bert, torch.nn.Identity(), self.tok, [['p']], ents, {}, False, names=False)
        self.assertTrue(config.names_in_text)
        with_names = text_features(self.bert, torch.nn.Identity(), self.tok, [['p']], ents, {}, False, names=True)
        self.assertFalse(np.allclose(without, with_names))

    def test_reads_collapsed(self):
        self.assertTrue(reads_collapsed('Leeds_v5'))
        self.assertFalse(reads_collapsed('Leeds'))


NOTEBOOK_IMAGENET = ANALYSIS / 'imagenet_footprint_features.npz'


@unittest.skipUnless(NOTEBOOK_IMAGENET.exists(), 'section 9a ImageNet cache not found')
class TestImageNetFeatures(unittest.TestCase):
    # The footprint feature without CityFM training: ResNet-18 with ImageNet weights, as section 9a of
    # notebooks/leeds_embeddings.ipynb computes it for single buildings (metric rasters), plus the floor area.

    @classmethod
    def setUpClass(cls):
        with open('Leeds/Data/entities.json', encoding='utf-8') as f:
            cls.entities = json.load(f)
        z = np.load(NOTEBOOK_IMAGENET)
        cls.ids, cls.emb = list(z['ids'][:6]), z['emb'][:6]
        config.raster_metric = True
        cls.cv = imagenet_cv()

    def test_single_polygons_match_notebook(self):
        out = imagenet_features(self.cv, [[i] for i in self.ids], self.entities)
        self.assertEqual(out.shape, (6, 513))
        np.testing.assert_allclose(out[:, :512], self.emb, atol=1e-4)
        # the last column is the floor area as CityFM's footprint branch receives it: capped at 5,000 m², / 5,000
        area = np.array([self.entities[i]['area'] for i in self.ids])
        np.testing.assert_allclose(out[:, 512], np.minimum(area, 5000) / 5000, atol=1e-6)

    def test_areas_average_their_polygons_and_empty_areas_are_zero(self):
        a, b, c = self.ids[:3]
        out = imagenet_features(self.cv, [[a, b, c], [], [b]], self.entities)
        np.testing.assert_allclose(out[0, :512], self.emb[:3].mean(0), atol=1e-4)
        np.testing.assert_array_equal(out[1], np.zeros(513))
        np.testing.assert_allclose(out[2, :512], self.emb[1], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
