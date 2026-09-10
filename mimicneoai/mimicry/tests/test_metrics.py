from __future__ import annotations

import random
import unittest

from mimicneoai.mimicry import (
    SEQUENCE_MIMICRY_V1_2,
    central_positions,
    compute_pair_metrics,
    is_sequence_mimic,
)
from mimicneoai.mimicry._engine import _build_triplet_index
from mimicneoai.mimicry._metrics import central_triplet_offsets


class SequenceMetricTests(unittest.TestCase):
    def test_central_positions_match_frozen_definition(self):
        self.assertEqual(central_positions(8), (3, 4, 5, 6))
        for length in (9, 10, 11):
            self.assertEqual(central_positions(length), (3, 4, 5, 6, 7))

    def test_primary_call_requires_all_three_conditions(self):
        self.assertTrue(is_sequence_mimic("AAAAAAAA", "CAAAAACC"))
        self.assertFalse(is_sequence_mimic("AAAAAAAAA", "AAAACACAC"))
        self.assertFalse(is_sequence_mimic("AAAAAAAA", "VVVAAAVV"))

    def test_lcs_and_anchor_metrics_are_annotation_only(self):
        metrics = compute_pair_metrics("AAAAAAAA", "CAAAAACC")
        self.assertEqual(metrics.central_matches, 3)
        self.assertEqual(metrics.central_run_length, 3)
        self.assertEqual(metrics.legacy_lcs_length, 5)
        self.assertTrue(metrics.p2_exact_match)
        self.assertFalse(metrics.pomega_exact_match)

    def test_invalid_or_unequal_peptides_do_not_pass(self):
        self.assertFalse(is_sequence_mimic("AAAAAAA", "AAAAAAA"))
        self.assertFalse(is_sequence_mimic("AAAAAAAX", "AAAAAAAA"))
        self.assertFalse(is_sequence_mimic("AAAAAAAA", "AAAAAAAAA"))
        with self.assertRaises(ValueError):
            compute_pair_metrics("AAAAAAAA", "AAAAAAAAA")

    def test_triplet_index_is_equivalent_to_brute_force(self):
        rng = random.Random(20260726)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        for length in SEQUENCE_MIMICRY_V1_2.peptide_lengths:
            source_a = {
                "".join(rng.choice(alphabet) for _ in range(length)) for _ in range(50)
            }
            source_b = {
                "".join(rng.choice(alphabet) for _ in range(length)) for _ in range(50)
            }
            source_b.update(list(source_a)[:3])
            expected = {
                (peptide_a, peptide_b)
                for peptide_a in source_a
                for peptide_b in source_b
                if is_sequence_mimic(peptide_a, peptide_b)
            }
            index = _build_triplet_index(source_b, length, SEQUENCE_MIMICRY_V1_2)
            observed = set()
            for peptide_a in source_a:
                recalled = set()
                for offset in central_triplet_offsets(length):
                    recalled.update(index.get((offset, peptide_a[offset : offset + 3]), set()))
                observed.update(
                    (peptide_a, peptide_b)
                    for peptide_b in recalled
                    if is_sequence_mimic(peptide_a, peptide_b)
                )
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
