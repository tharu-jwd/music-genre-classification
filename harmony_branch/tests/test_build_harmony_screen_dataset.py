import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_harmony_screen_dataset import build_screen_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sources(root: Path, *, coarse=False):
    feature_dir = root / "features"
    target_dir = root / "targets"
    feature_dir.mkdir()
    target_dir.mkdir()
    feature_items = []
    target_items = []
    for index, split in enumerate(("train", "validation"), start=1):
        song_id = f"{index:07d}"
        feature_path = feature_dir / f"{song_id}.npz"
        end = 29.1 if coarse else 0.04
        np.savez(
            feature_path,
            encoded_sequence=np.zeros((2, 4), dtype=np.float32),
            sequence_start_times=np.asarray([0.0, end / 2], dtype=np.float32),
            sequence_end_times=np.asarray([end / 2, end], dtype=np.float32),
            sequence_mask=np.asarray([True, True]),
            sequence_window_index=np.asarray([0, 0], dtype=np.int16),
        )
        feature_items.append({
            "song_id": song_id,
            "split": split,
            "artifact": feature_path.name,
            "artifact_sha256": sha256(feature_path),
            "windows": 1,
            "tokens_per_window": 2,
        })
        target_path = target_dir / f"{song_id}__window_000.npz"
        frame_times = np.asarray([end / 4, 3 * end / 4], dtype=np.float32)
        chroma = np.zeros((2, 12), dtype=np.float32)
        chroma[0, 0] = 1
        chroma[1, 4] = 1
        np.savez(
            target_path,
            frame_times_seconds=frame_times,
            chroma=chroma,
            chroma_valid=np.asarray([True, True]),
        )
        target_items.append({
            "song_id": song_id,
            "split": split,
            "window_index": 0,
            "artifact": target_path.name,
            "artifact_sha256": sha256(target_path),
        })
    cohort_hash = "c" * 64
    (feature_dir / "index.json").write_text(json.dumps({
        "schema_version": "harmony_encoder_cache_pilot_v1",
        "status": "ready",
        "failure_count": 0,
        "contains_genre_labels": False,
        "device": "cpu",
        "cohort_sha256": cohort_hash,
        "items": feature_items,
    }))
    (target_dir / "index.json").write_text(json.dumps({
        "schema_version": "harmony_target_pilot_v1",
        "status": "ready",
        "failure_count": 0,
        "contains_genre_labels": False,
        "target_variant": "temporal_chroma_v1",
        "source_cohort_sha256": cohort_hash,
        "minimum_valid_fraction": 0.05,
        "items": target_items,
    }))
    return feature_dir, target_dir


class BuildHarmonyScreenDatasetTest(unittest.TestCase):
    def test_joins_features_and_targets_by_song_window_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, targets = write_sources(root)
            output = root / "screen"
            result = build_screen_dataset(features, targets, output)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["coverage"], 1.0)
            self.assertEqual(result["split_coverage"]["train"]["coverage"], 1.0)
            item = result["items"][0]
            arrays = np.load(output / item["target_artifact"], allow_pickle=False)
            np.testing.assert_array_equal(arrays["chroma_valid"], [True, True])
            np.testing.assert_allclose(arrays["chroma_target"][0, 0], 1.0)
            np.testing.assert_allclose(arrays["chroma_target"][1, 4], 1.0)
            np.testing.assert_array_equal(arrays["screen_token_mask"], [True, True])

    def test_mismatched_cohorts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, targets = write_sources(root)
            index = json.loads((targets / "index.json").read_text())
            index["source_cohort_sha256"] = "d" * 64
            (targets / "index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "different frozen cohorts"):
                build_screen_dataset(features, targets, root / "screen")

    def test_changed_feature_artifact_produces_failed_not_partial_ready_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, targets = write_sources(root)
            with (features / "0000001.npz").open("ab") as handle:
                handle.write(b"changed")
            result = build_screen_dataset(features, targets, root / "screen")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_count"], 1)

    def test_coarse_window_features_cannot_be_called_temporal_screen_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, targets = write_sources(root, coarse=True)
            result = build_screen_dataset(features, targets, root / "screen")
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("too coarse" in item["error"] for item in result["failures"]))


if __name__ == "__main__":
    unittest.main()
