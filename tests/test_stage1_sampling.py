# Tests for how stage 1 draws its training pairs (utils.training.stage1_segments and stage1_sample).
# Run from the repository root: .venv/bin/python -m unittest tests.test_stage1_sampling
import collections
import random
import unittest

from utils.training import stage1_sample, stage1_segments

ENTITIES = {k: {'id': k, 'tags': {'amenity': k}} for k in 'abcde'}
DATA = {
    'empty': {'context': []},
    'one': {'context': ['a']},
    'one-plus-missing': {'context': ['a', 'zz']},  # 'zz' is not in ENTITIES
    'two': {'context': ['a', 'b']},
    'five': {'context': list('abcde')},
}


def original_sample(context, entities):
    # The pair-drawing logic of train_SS_im before the hold-out option, copied as it was.
    if not context:
        query = []
    else:
        r = random.randint(0, len(context))
        if r == len(context):
            query = []
        elif context[r] in entities:
            query = [entities[context[r]]]
        else:
            query = []
    return query, [entities[k] for k in context if k in entities]


class TestOriginalRegime(unittest.TestCase):

    def test_keeps_every_segment(self):
        self.assertEqual(sorted(stage1_segments(DATA, ENTITIES, hold_out=False)), sorted(DATA))

    def test_draws_exactly_as_before(self):
        for seed in range(50):
            for seg in DATA.values():
                random.seed(seed)
                expected = original_sample(seg['context'], ENTITIES)
                random.seed(seed)
                self.assertEqual(stage1_sample(seg['context'], ENTITIES, hold_out=False), expected)


class TestHoldOutRegime(unittest.TestCase):

    def test_keeps_only_segments_with_two_known_entries(self):
        self.assertEqual(sorted(stage1_segments(DATA, ENTITIES, hold_out=True)), ['five', 'two'])

    def test_query_is_removed_from_its_context(self):
        random.seed(0)
        for _ in range(500):
            query, ctx = stage1_sample(DATA['five']['context'], ENTITIES, hold_out=True)
            self.assertEqual(len(query), 1)
            self.assertNotIn(query[0], ctx)
            self.assertEqual(len(ctx), 4)

    def test_never_draws_context_none(self):
        random.seed(0)
        for _ in range(500):
            query, ctx = stage1_sample(DATA['two']['context'], ENTITIES, hold_out=True)
            self.assertEqual(len(query), 1)
            self.assertEqual(len(ctx), 1)

    def test_draws_every_entry_about_equally(self):
        random.seed(0)
        counts = collections.Counter(stage1_sample(DATA['five']['context'], ENTITIES, hold_out=True)[0][0]['id']
                                     for _ in range(5000))
        self.assertEqual(set(counts), set('abcde'))
        self.assertTrue(all(800 < n < 1200 for n in counts.values()), counts)

    def test_skips_entries_missing_from_entities(self):
        context = ['a', 'zz', 'b']
        random.seed(0)
        for _ in range(200):
            query, ctx = stage1_sample(context, ENTITIES, hold_out=True)
            self.assertEqual(len(ctx), 1)
            self.assertEqual({query[0]['id'], ctx[0]['id']}, {'a', 'b'})

    def test_refuses_a_segment_with_fewer_than_two_entries(self):
        with self.assertRaises(ValueError):
            stage1_sample(DATA['one-plus-missing']['context'], ENTITIES, hold_out=True)


if __name__ == '__main__':
    unittest.main()
