import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache_harmony_encoder_pilot import cache_encoder_pilot
from scripts.freeze_experiment_cohort import freeze_cohort
from shared_encoder import SHARED_ENCODER_ARCHITECTURE, SharedAudioEncoder


def write_fixture(root: Path):
    rows = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        path = root / f"{index:07d}.npy"
        np.save(path, np.ones((96, 10 + index), dtype=np.float32))
        rows.append({"song_id": f"{index:07d}", "split": split, "logmel_path": path.name})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("song_id", "split", "logmel_path"))
        writer.writeheader()
        writer.writerows(rows)
    cohort = root / "cohort.json"
    cohort.write_text(json.dumps(freeze_cohort(manifest, splits=("train", "validation"))))
    model = SharedAudioEncoder()
    checkpoint = root / "instrument.pt"
    torch.save({
        "model": model.state_dict(),
        "encoder_architecture": SHARED_ENCODER_ARCHITECTURE,
        "best_macro_map": 0.5,
        "tags": ["instrument---guitar"],
        "training_config": {
            "input_schema": "mtg_full_audio_logmel_windows_v1",
            "max_windows": 12,
        },
    }, checkpoint)
    return manifest, cohort, checkpoint


class CacheHarmonyEncoderPilotTest(unittest.TestCase):
    def test_caches_masked_temporal_features_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort, checkpoint = write_fixture(root)
            output = root / "cache"
            result = cache_encoder_pilot(
                manifest,
                cohort,
                checkpoint,
                output,
                root=root,
                max_songs=2,
                max_cpu_seconds=60,
                cpu_threads=1,
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["device"], "cpu")
            self.assertFalse(result["contains_genre_labels"])
            self.assertEqual(result["cached_songs"], 2)
            self.assertEqual(result["checkpoint_format"], "shared_cnn_v2")
            self.assertEqual(result["encoder_architecture"], SHARED_ENCODER_ARCHITECTURE)
            self.assertAlmostEqual(result["token_stride_seconds"], 256 * 2 / 12000)
            item = result["items"][0]
            arrays = np.load(output / item["artifact"], allow_pickle=False)
            self.assertEqual(arrays["encoded_sequence"].shape, (683, 128))
            self.assertEqual(arrays["window_repr"].shape, (1, 128))
            self.assertEqual(item["windows"], 1)
            self.assertEqual(item["tokens_per_window"], 683)
            self.assertEqual(item["valid_tokens"], 6)
            self.assertEqual(arrays["sequence_window_index"][:6].tolist(), [0] * 6)
            self.assertTrue(np.all(arrays["sequence_window_index"][6:] == -1))
            self.assertRegex(item["array_content_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue((output / "index.json").is_file())

    def test_test_split_is_rejected_before_feature_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort, checkpoint = write_fixture(root)
            value = json.loads(cohort.read_text())
            value["splits"]["test"] = ["0000003"]
            value["counts"]["test"] = 1
            canonical = json.dumps(value["splits"], sort_keys=True, separators=(",", ":")).encode()
            value["cohort_sha256"] = hashlib.sha256(canonical).hexdigest()
            cohort.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "test songs are forbidden"):
                cache_encoder_pilot(
                    manifest, cohort, checkpoint, root / "cache", root=root
                )

    def test_hard_caps_cannot_be_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort, checkpoint = write_fixture(root)
            with self.assertRaisesRegex(ValueError, r"max_songs must be in \[1, 32\]"):
                cache_encoder_pilot(
                    manifest,
                    cohort,
                    checkpoint,
                    root / "cache",
                    root=root,
                    max_songs=33,
                )

    def test_checkpoint_input_contract_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cohort, checkpoint = write_fixture(root)
            value = torch.load(checkpoint, weights_only=True)
            value["training_config"]["input_schema"] = "wrong"
            torch.save(value, checkpoint)
            with self.assertRaisesRegex(ValueError, "different log-Mel"):
                cache_encoder_pilot(
                    manifest, cohort, checkpoint, root / "cache", root=root
                )


if __name__ == "__main__":
    unittest.main()
