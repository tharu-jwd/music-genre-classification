import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.paired_bootstrap_compare import align_pair, compare, load_predictions, macro_metric


class PairedBootstrapCompareTest(unittest.TestCase):
    def setUp(self):
        self.targets = np.asarray(
            [[row % 2, (row // 2) % 2, (row // 4) % 2] for row in range(64)],
            dtype=np.int8,
        )
        self.candidate = self.targets * 0.8 + (1 - self.targets) * 0.2
        pattern = np.linspace(0.05, 0.95, len(self.targets))
        self.control = np.column_stack([pattern, pattern[::-1], np.roll(pattern, 7)])

    def test_strong_candidate_advances(self):
        result = compare(
            self.targets,
            self.control,
            self.candidate,
            metric="macro_pr_auc",
            confidence=0.90,
            n_resamples=200,
            seed=7,
            min_effect=0.01,
        )
        self.assertEqual(result["decision"], "advance")
        self.assertEqual(result["schema_version"], "paired_bootstrap_comparison_v1")
        self.assertGreater(result["point_difference"], 0)

    def test_identical_candidate_is_not_advanced(self):
        result = compare(
            self.targets,
            self.control,
            self.control.copy(),
            metric="macro_roc_auc",
            confidence=0.90,
            n_resamples=200,
            seed=7,
            min_effect=0.001,
        )
        self.assertEqual(result["decision"], "stop")

    def test_alignment_uses_song_ids(self):
        ids = np.asarray([f"song-{i}" for i in range(len(self.targets))])
        labels = np.asarray(["a", "b", "c"])
        control = {
            "song_ids": ids,
            "label_names": labels,
            "targets": self.targets,
            "scores": self.control,
        }
        candidate = {
            "song_ids": ids[::-1],
            "label_names": labels,
            "targets": self.targets[::-1],
            "scores": self.candidate[::-1],
        }
        aligned_ids, aligned_targets, _, aligned_scores = align_pair(control, candidate)
        np.testing.assert_array_equal(aligned_ids, ids)
        np.testing.assert_array_equal(aligned_targets, self.targets)
        np.testing.assert_allclose(aligned_scores, self.candidate)

    def test_undefined_labels_are_excluded(self):
        targets = np.column_stack([self.targets[:, 0], np.zeros(len(self.targets))])
        scores = np.column_stack([self.candidate[:, 0], np.full(len(self.targets), 0.5)])
        value = macro_metric(targets, scores, "macro_pr_auc")
        self.assertAlmostEqual(value, 1.0)

    def test_saved_npz_contract_loads_without_pickle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            np.savez_compressed(
                path,
                song_ids=np.asarray([f"song-{i}" for i in range(len(self.targets))], dtype=str),
                label_names=np.asarray(["a", "b", "c"], dtype=str),
                targets=self.targets,
                scores=self.candidate,
            )
            loaded = load_predictions(path)
        np.testing.assert_array_equal(loaded["targets"], self.targets)


if __name__ == "__main__":
    unittest.main()
