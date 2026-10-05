"""Small invariants that protect composition exclusion and raw truth merging."""
import unittest

import numpy as np

from prepare_data import overlap_mask, recover_raw_action, canonical_action, surface_family


class DataRules(unittest.TestCase):
    def test_half_open_intervals_and_off_grid(self):
        starts = np.array([0, 1, 79, 80, 81, 159, 160])
        np.testing.assert_array_equal(overlap_mask(starts, np.array([80])), [False, True, True, True, True, True, False])
        np.testing.assert_array_equal(overlap_mask(starts, np.array([], dtype=int)), np.zeros(7, dtype=bool))

    def test_many_intervals_against_brute_force(self):
        rng = np.random.default_rng(42)
        starts = rng.integers(0, 2000, 500)
        forbidden = rng.integers(0, 2000, 20)
        brute = np.array([any(s < f + 80 and f < s + 80 for f in forbidden) for s in starts])
        np.testing.assert_array_equal(overlap_mask(starts, forbidden), brute)

    def test_release_truth(self):
        self.assertEqual(recover_raw_action("release", "Releasing an object in midair."), "release midair")
        self.assertEqual(recover_raw_action("release", "Releasing an object from the supporting surface."), "release")
        self.assertEqual(canonical_action("release midair"), "release")
        self.assertEqual(canonical_action("place then release"), "place then release")

    def test_surface_strata_preserve_flat_meaning(self):
        self.assertEqual(surface_family("a fixed flat surface"), "flat")
        self.assertEqual(surface_family("a flat surface"), "flat")
        self.assertEqual(surface_family("a wobbling surface"), "wobble")
        self.assertEqual(surface_family(""), "unknown")


if __name__ == "__main__":
    unittest.main()
