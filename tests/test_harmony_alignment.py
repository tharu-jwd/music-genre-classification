import unittest

import numpy as np

from scripts.harmony_alignment import align_chroma_to_intervals


class HarmonyAlignmentTest(unittest.TestCase):
    def test_pools_only_valid_frames_inside_each_token_interval(self):
        chroma = np.zeros((4, 12), dtype=np.float32)
        chroma[0, 0] = 1
        chroma[1, 2] = 1
        chroma[2, 4] = 1
        # The invalid source frame must be zero and must not dilute the target.
        result = align_chroma_to_intervals(
            np.asarray([0.01, 0.03, 0.05, 0.07]),
            chroma,
            np.asarray([True, True, True, False]),
            np.asarray([0.0, 0.04, 2.0]),
            np.asarray([0.04, 0.08, 2.04]),
            np.asarray([True, True, False]),
            max_token_duration_seconds=0.1,
        )
        np.testing.assert_allclose(result.chroma[0, [0, 2]], [0.5, 0.5])
        np.testing.assert_allclose(result.chroma[1, 4], 1.0)
        np.testing.assert_array_equal(result.valid, [True, True, False])
        np.testing.assert_array_equal(result.contributing_frames, [2, 1, 0])
        self.assertEqual(result.coverage, 1.0)

    def test_silent_interval_is_missing_supervision_not_uniform_chroma(self):
        result = align_chroma_to_intervals(
            np.asarray([0.01, 0.03]),
            np.zeros((2, 12), dtype=np.float32),
            np.asarray([False, False]),
            np.asarray([0.0]),
            np.asarray([0.04]),
            np.asarray([True]),
            max_token_duration_seconds=0.1,
        )
        self.assertFalse(result.valid[0])
        self.assertEqual(float(result.chroma.sum()), 0.0)
        self.assertEqual(result.coverage, 0.0)

    def test_window_level_vector_is_rejected_as_too_coarse_for_progressions(self):
        with self.assertRaisesRegex(ValueError, "too coarse"):
            align_chroma_to_intervals(
                np.asarray([1.0]),
                np.eye(12, dtype=np.float32)[:1],
                np.asarray([True]),
                np.asarray([0.0]),
                np.asarray([29.1]),
                np.asarray([True]),
                max_token_duration_seconds=1.0,
            )

    def test_non_normalized_source_chroma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            align_chroma_to_intervals(
                np.asarray([0.01]),
                np.ones((1, 12), dtype=np.float32),
                np.asarray([True]),
                np.asarray([0.0]),
                np.asarray([0.02]),
                np.asarray([True]),
                max_token_duration_seconds=0.1,
            )


if __name__ == "__main__":
    unittest.main()
