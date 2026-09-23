import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timbre_branch.constants import FEATURE_COLUMNS
from timbre_branch.data import TimbreEmbeddingDataset
from timbre_branch.losses import masked_smooth_l1_loss
from timbre_branch.inference import predict_timbre_concepts
from timbre_branch.model import TimbreBranch
from timbre_branch.preprocessing import TimbreStandardizer, load_timbre_targets
from timbre_branch.training import load_checkpoint, save_checkpoint, train_one_epoch


class TimbreBranchSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_real_target_contract(self):
        targets = load_timbre_targets(PROJECT_ROOT / "data/timbre_features_raw.csv")
        self.assertEqual(len(targets), 7324)
        self.assertEqual(len(FEATURE_COLUMNS), 35)
        self.assertFalse(targets.loc[:, FEATURE_COLUMNS].isna().any().any())

    def test_forward_loss_train_checkpoint_and_inverse(self):
        rows = 48
        embeddings = np.random.normal(size=(rows, 128)).astype(np.float32)
        raw_targets = np.random.normal(size=(rows, 35)).astype(np.float64)
        raw_targets[:, 0] = raw_targets[:, 0] * 600.0 + 1800.0
        mask = np.ones((rows, 35), dtype=bool)
        mask[0, 7] = False
        raw_targets[0, 7] = np.nan

        scaler = TimbreStandardizer().fit(raw_targets[:32], mask[:32])
        standardized = scaler.transform(raw_targets)
        standardized = np.where(mask, standardized, 0.0)
        recovered = scaler.inverse_transform(standardized[1:])
        np.testing.assert_allclose(recovered, raw_targets[1:], rtol=1e-5, atol=1e-5)

        dataset = TimbreEmbeddingDataset(
            [f"track_{i:04d}" for i in range(rows)],
            embeddings,
            standardized,
            mask,
        )
        loader = DataLoader(dataset, batch_size=12, shuffle=False)
        model = TimbreBranch()
        output = model(torch.from_numpy(embeddings[:5]))
        self.assertEqual(tuple(output.shape), (5, 35))
        self.assertTrue(torch.isfinite(output).all())

        loss = masked_smooth_l1_loss(
            output,
            torch.from_numpy(standardized[:5]),
            torch.from_numpy(mask[:5]),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        epoch_loss = train_one_epoch(model, loader, optimizer, "cpu")
        self.assertTrue(np.isfinite(epoch_loss))

        model.eval()
        with torch.no_grad():
            reference = model(torch.from_numpy(embeddings[:5]))
        explanation = predict_timbre_concepts(
            model, torch.from_numpy(embeddings[:5]), scaler
        )
        self.assertEqual(tuple(explanation["z_timbre"].shape), (5, 35))
        self.assertEqual(explanation["d_hat_original_units"].shape, (5, 35))
        self.assertEqual(tuple(explanation["feature_names"]), FEATURE_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "timbre.pt"
            save_checkpoint(
                checkpoint,
                model,
                scaler,
                epoch=1,
                metrics={"smoke_loss": epoch_loss},
                seed=7,
                optimizer=optimizer,
            )
            restored, restored_scaler, payload = load_checkpoint(checkpoint)
            restored.eval()
            with torch.no_grad():
                actual = restored(torch.from_numpy(embeddings[:5]))
            torch.testing.assert_close(actual, reference)
            self.assertEqual(payload["feature_names"], list(FEATURE_COLUMNS))
            np.testing.assert_allclose(restored_scaler.mean, scaler.mean)

    def test_shape_contract_rejects_invalid_input(self):
        model = TimbreBranch()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 64))
        with self.assertRaises(ValueError):
            model(torch.randn(128))

    def test_training_cli_end_to_end(self):
        targets = load_timbre_targets(PROJECT_ROOT / "data/timbre_features_raw.csv").head(60)
        track_ids = targets["TRACK_ID"].to_numpy(dtype=str)
        embeddings = np.random.normal(size=(60, 128)).astype(np.float32)
        split_names = np.array(
            ["train"] * 36 + ["validation"] * 12 + ["test"] * 12,
            dtype=str,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            embedding_path = directory / "embeddings.npz"
            split_path = directory / "splits.csv"
            checkpoint_path = directory / "best.pt"
            np.savez(embedding_path, track_ids=track_ids, embeddings=embeddings)
            pd.DataFrame({"TRACK_ID": track_ids, "split": split_names}).to_csv(
                split_path, index=False
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/train_timbre_branch.py"),
                    "--embeddings", str(embedding_path),
                    "--targets", str(PROJECT_ROOT / "data/timbre_features_raw.csv"),
                    "--splits", str(split_path),
                    "--output", str(checkpoint_path),
                    "--epochs", "2",
                    "--patience", "2",
                    "--batch-size", "12",
                    "--device", "cpu",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(checkpoint_path.with_suffix(".metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
