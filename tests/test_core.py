import unittest

from skip_list import SkipList


class DeterministicRNG:
    """A pseudo-RNG whose draws we control, so the skip-list's random structure
    is fixed across test runs. Tests must never depend on wall-clock or OS
    randomness; this class makes that property concrete.
    """

    def __init__(self, draws):
        # draws: a list of floats in [0, 1) consumed one at a time.
        self._draws = list(draws)
        self._idx = 0

    def __call__(self):
        if self._idx >= len(self._draws):
            self._idx = 0
        v = self._draws[self._idx]
        self._idx += 1
        return v


class TestConstructionAndBasics(unittest.TestCase):
    def test_empty_has_len_zero(self):
        sl = SkipList()
        self.assertEqual(len(sl), 0)

    def test_empty_iter_yields_nothing(self):
        sl = SkipList()
        self.assertEqual(list(sl), [])

    def test_repr_empty(self):
        self.assertEqual(repr(SkipList()), "SkipList({})")

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            SkipList(p=0.0)
        with self.assertRaises(ValueError):
            SkipList(p=1.0)
        with self.assertRaises(ValueError):
            SkipList(p=-0.5)


class TestInsertLookup(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def test_insert_and_get_single(self):
        sl = self._make([0.1])
        sl["a"] = 1
        self.assertEqual(sl["a"], 1)
        self.assertEqual(len(sl), 1)
        self.assertIn("a", sl)

    def test_get_missing_raises_keyerror(self):
        sl = self._make([0.1])
        sl[1] = "x"
        with self.assertRaises(KeyError):
            _ = sl[2]

    def test_get_returns_default_when_missing(self):
        sl = self._make([0.1])
        sl[1] = "x"
        self.assertEqual(sl.get(2, "default"), "default")
        self.assertIsNone(sl.get(2))

    def test_contains_false_for_missing(self):
        sl = self._make([0.1])
        sl[5] = "x"
        self.assertNotIn(6, sl)

    def test_replace_existing_value(self):
        # Heights drawn low so the tower stays at level 1; replacement must
        # not change the structure, only the value.
        sl = self._make([0.9, 0.9])
        sl[1] = "a"
        sl[1] = "b"
        self.assertEqual(sl[1], "b")
        self.assertEqual(len(sl), 1)

    def test_insert_many_keys_sorted(self):
        # All draws high keep every node at height 1, so the structure is a
        # plain sorted linked list. This is the simplest configuration that
        # still exercises search and insertion.
        draws = [0.9] * 50
        sl = self._make(draws)
        for i in range(50):
            sl[i] = i * 10
        self.assertEqual(len(sl), 50)
        for i in range(50):
            self.assertEqual(sl[i], i * 10)
        self.assertEqual(list(sl), list(range(50)))

    def test_insert_many_keys_reverse(self):
        draws = [0.9] * 50
        sl = self._make(draws)
        for i in range(49, -1, -1):
            sl[i] = i
        self.assertEqual(list(sl), list(range(50)))

    def test_tall_towers_exercised(self):
        # Draws low enough to grow tall towers, exercising multi-level search
        # and the head-extension path in __setitem__.
        draws = [0.05, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.9, 0.05, 0.9]
        sl = self._make(draws)
        for k, v in [(10, 100), (20, 200), (30, 300), (40, 400), (5, 50)]:
            sl[k] = v
        self.assertEqual(list(sl), [5, 10, 20, 30, 40])
        self.assertEqual(sl[30], 300)
        self.assertEqual(sl[5], 50)


class TestDeletion(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def test_delete_single(self):
        sl = self._make([0.9])
        sl[1] = "x"
        del sl[1]
        self.assertEqual(len(sl), 0)
        self.assertNotIn(1, sl)
        self.assertEqual(list(sl), [])

    def test_delete_missing_raises(self):
        sl = self._make([0.9])
        sl[1] = "x"
        with self.assertRaises(KeyError):
            del sl[2]

    def test_delete_from_middle_preserves_order(self):
        draws = [0.9] * 5
        sl = self._make(draws)
        for i in [1, 2, 3, 4, 5]:
            sl[i] = i
        del sl[3]
        self.assertEqual(list(sl), [1, 2, 4, 5])
        self.assertEqual(len(sl), 4)
        self.assertNotIn(3, sl)

    def test_delete_then_reinsert(self):
        draws = [0.9] * 5
        sl = self._make(draws)
        sl[1] = "a"
        sl[2] = "b"
        del sl[1]
        sl[1] = "c"
        self.assertEqual(sl[1], "c")
        self.assertEqual(sl[2], "b")
        self.assertEqual(list(sl), [1, 2])

    def test_delete_all_then_max_raises(self):
        draws = [0.9] * 3
        sl = self._make(draws)
        sl[1] = "a"
        sl[2] = "b"
        del sl[1]
        del sl[2]
        with self.assertRaises(KeyError):
            sl.max_item()
        with self.assertRaises(KeyError):
            sl.min_item()

    def test_delete_shrinks_max_level(self):
        # Draw a tall tower for the first node so the head grows. Then delete
        # that node; the head's trailing empty levels must be trimmed so the
        # search height tracks the data.
        draws = [0.05, 0.05, 0.05, 0.05, 0.9, 0.9]
        sl = self._make(draws)
        sl[1] = "a"  # tall tower, head grows to level 5
        sl[2] = "b"  # short
        self.assertEqual(sl._max_level, 5)
        del sl[1]
        self.assertLessEqual(sl._max_level, 2)
        self.assertEqual(list(sl), [2])


class TestIteration(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def test_keys_in_order(self):
        sl = self._make([0.9] * 5)
        for k in [3, 1, 4, 1, 5, 9, 2, 6]:
            sl[k] = k * 100
        self.assertEqual(list(sl), [1, 2, 3, 4, 5, 6, 9])

    def test_items_pairs(self):
        sl = self._make([0.9] * 3)
        sl["b"] = 2
        sl["a"] = 1
        sl["c"] = 3
        self.assertEqual(list(sl.items()), [("a", 1), ("b", 2), ("c", 3)])

    def test_values_in_key_order(self):
        sl = self._make([0.9] * 3)
        sl[10] = "x"
        sl[5] = "y"
        sl[15] = "z"
        self.assertEqual(list(sl.values()), ["y", "x", "z"])

    def test_keys_alias(self):
        sl = self._make([0.9] * 3)
        sl[2] = "a"
        sl[1] = "b"
        self.assertEqual(list(sl.keys()), [1, 2])


class TestRange(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def _populate(self, keys):
        sl = self._make([0.9] * len(keys))
        for k in keys:
            sl[k] = k * 10
        return sl

    def test_half_open_default(self):
        sl = self._populate([1, 2, 3, 4, 5])
        self.assertEqual([k for k, _ in sl.range(2, 4)], [2, 3])

    def test_inclusive_stop(self):
        sl = self._populate([1, 2, 3, 4, 5])
        self.assertEqual([k for k, _ in sl.range(2, 4, inclusive_stop=True)], [2, 3, 4])

    def test_exclusive_start(self):
        sl = self._populate([1, 2, 3, 4, 5])
        self.assertEqual([k for k, _ in sl.range(2, 5, inclusive_start=False)], [3, 4])

    def test_open_left_bound(self):
        sl = self._populate([1, 2, 3, 4, 5])
        self.assertEqual([k for k, _ in sl.range(None, 3)], [1, 2])

    def test_open_right_bound(self):
        sl = self._populate([1, 2, 3, 4, 5])
        self.assertEqual([k for k, _ in sl.range(3, None)], [3, 4, 5])

    def test_both_open_yields_all(self):
        sl = self._populate([1, 2, 3])
        self.assertEqual([k for k, _ in sl.range()], [1, 2, 3])

    def test_empty_range_when_no_match(self):
        sl = self._populate([1, 2, 3])
        self.assertEqual(list(sl.range(10, 20)), [])

    def test_empty_range_when_start_equals_stop_exclusive(self):
        sl = self._populate([1, 2, 3])
        self.assertEqual(list(sl.range(2, 2)), [])


class TestMinMax(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def test_min_max_single(self):
        sl = self._make([0.9])
        sl[42] = "x"
        self.assertEqual(sl.min_item(), (42, "x"))
        self.assertEqual(sl.max_item(), (42, "x"))

    def test_min_max_after_mixed_inserts(self):
        sl = self._make([0.9] * 5)
        for k in [5, 1, 9, 3, 7]:
            sl[k] = k
        self.assertEqual(sl.min_item(), (1, 1))
        self.assertEqual(sl.max_item(), (9, 9))

    def test_min_max_empty_raises(self):
        sl = self._make([0.9])
        with self.assertRaises(KeyError):
            sl.min_item()
        with self.assertRaises(KeyError):
            sl.max_item()


class TestKeyTypes(unittest.TestCase):
    def _make(self, draws):
        return SkipList(p=0.5, rng=DeterministicRNG(draws))

    def test_string_keys_sorted_lexicographically(self):
        sl = self._make([0.9] * 3)
        sl["banana"] = 1
        sl["apple"] = 2
        sl["cherry"] = 3
        self.assertEqual(list(sl), ["apple", "banana", "cherry"])

    def test_float_keys(self):
        sl = self._make([0.9] * 4)
        sl[2.5] = "a"
        sl[1.0] = "b"
        sl[3.5] = "c"
        sl[2.0] = "d"
        self.assertEqual(list(sl), [1.0, 2.0, 2.5, 3.5])

    def test_none_value_stored_and_retrieved(self):
        sl = self._make([0.9])
        sl[1] = None
        self.assertIs(sl[1], None)


class TestRepr(unittest.TestCase):
    def test_repr_nonempty(self):
        sl = SkipList(rng=DeterministicRNG([0.9, 0.9]))
        sl["b"] = 2
        sl["a"] = 1
        self.assertEqual(repr(sl), "SkipList({ 'a': 1, 'b': 2 })")


if __name__ == "__main__":
    unittest.main()
