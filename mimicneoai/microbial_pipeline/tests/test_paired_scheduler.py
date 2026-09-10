from __future__ import annotations

import unittest

from mimicneoai.microbial_pipeline.microbial import _parse_sample_units


class MicrobialPairedSchedulerTest(unittest.TestCase):
    def test_single_sample_mode_rejects_pair_syntax(self) -> None:
        with self.assertRaisesRegex(ValueError, "tumor_with_matched_normal"):
            _parse_sample_units(["T,N"], tumor_with_matched_normal=False)

    def test_paired_mode_parses_strict_pair_units(self) -> None:
        units = _parse_sample_units(["T1,N1", "T2,N2"], tumor_with_matched_normal=True)
        self.assertEqual([unit.label for unit in units], ["T1,N1", "T2,N2"])
        self.assertEqual(units[0].tumor_sample, "T1")
        self.assertEqual(units[0].normal_sample, "N1")

    def test_paired_mode_rejects_invalid_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            _parse_sample_units(["T,N,extra"], tumor_with_matched_normal=True)
        with self.assertRaisesRegex(ValueError, "different"):
            _parse_sample_units(["T,T"], tumor_with_matched_normal=True)
        with self.assertRaisesRegex(ValueError, "Duplicate microbial tumor"):
            _parse_sample_units(["T,N1", "T,N2"], tumor_with_matched_normal=True)
        with self.assertRaisesRegex(ValueError, "role conflict"):
            _parse_sample_units(["T,N", "N,Other"], tumor_with_matched_normal=True)


if __name__ == "__main__":
    unittest.main()
