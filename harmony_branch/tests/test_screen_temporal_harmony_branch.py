import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from screen_temporal_harmony_branch import screen_branch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(root: Path) -> Path:
    dataset = root / "screen"
    source = root / "source"
    dataset.mkdir()
    source.mkdir()
    items = []
    for index, split in enumerate(("train", "validation"), start=1):
        song_id = f"{index:07d}"
        feature_path = source / f"{song_id}.npz"
        encoded = np.zeros((4, 6), dtype=np.float32)
        encoded[:, index - 1] = 1
        np.savez(
            feature_path,
            encoded_sequence=encoded,
            sequence_mask=np.asarray([True, True, True, True]),
            sequence_window_index=np.asarray([0, 0, 0, 0], dtype=np.int16),
        )
        target_path = dataset / f"{song_id}.npz"
        chroma = np.zeros((4, 12), dtype=np.float32)
        chroma[:, index - 1] = 1
        np.savez(
            target_path,
            chroma_target=chroma,
            chroma_valid=np.asarray([True, True, True, True]),
            screen_token_mask=np.asarray([True, True, True, True]),
        )
        items.append({
            "song_id": song_id,
            "split": split,
            "feature_artifact": str(feature_path),
            "feature_artifact_sha256": sha256(feature_path),
            "target_artifact": target_path.name,
            "target_artifact_sha256": sha256(target_path),
            "windows": 1,
            "tokens_per_window": 4,
        })
    (dataset / "index.json").write_text(json.dumps({
        "schema_version": "harmony_screen_dataset_v1",
        "status": "ready",
        "failure_count": 0,
        "contains_genre_labels": False,
        "cohort_sha256": "c" * 64,
        "items": items,
    }))
    return dataset


class ScreenTemporalHarmonyBranchTest(unittest.TestCase):
    def test_runs_one_fixed_cpu_screen_and_reports_baselines_without_selecting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(root)
            output = root / "run"
            result = screen_branch(
                dataset,
                output,
                max_epochs=2,
                max_cpu_seconds=30,
                cpu_threads=1,
            )
            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["automatic_selection"])
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["configuration"]["seed"], 42)
            self.assertEqual(result["configuration"]["embedding_dim"], 32)
            self.assertIn("uniform", result["baselines"])
            self.assertIn("training_mean", result["baselines"])
            self.assertGreater(result["parameter_count"], 0)
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "report.json").is_file())

    def test_deadline_before_first_epoch_writes_failed_report_not_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(root)
            output = root / "run"
            result = screen_branch(
                dataset,
                output,
                max_epochs=2,
                max_cpu_seconds=1e-9,
                cpu_threads=1,
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_reason"], "no_complete_validation_epoch")
            self.assertFalse((output / "best.pt").exists())

    def test_hard_caps_cannot_be_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(root)
            with self.assertRaisesRegex(ValueError, r"max_epochs must be in \[1, 20\]"):
                screen_branch(dataset, root / "run", max_epochs=21)

    def test_changed_cached_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = write_dataset(root)
            index = json.loads((dataset / "index.json").read_text())
            path = Path(index["items"][0]["feature_artifact"])
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "feature artifact changed"):
                screen_branch(dataset, root / "run", max_epochs=1)


if __name__ == "__main__":
    unittest.main()
