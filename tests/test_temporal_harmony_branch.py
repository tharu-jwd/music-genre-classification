import tempfile
import unittest
from pathlib import Path

import torch

from scripts.temporal_harmony_branch import (
    TemporalHarmonyBranch,
    masked_hard_classification_loss,
    masked_soft_target_cross_entropy,
)


def fixture():
    sequence = torch.randn(2, 8, 6)
    mask = torch.tensor([
        [True, True, True, True, True, True, False, False],
        [False, False, False, False, False, False, False, False],
    ])
    window_index = torch.tensor([
        [0, 0, 0, 0, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ])
    return sequence, mask, window_index


class TemporalHarmonyBranchTest(unittest.TestCase):
    def test_configurable_outputs_and_entirely_unavailable_song(self):
        model = TemporalHarmonyBranch(
            6, embedding_dim=13, hidden_dim=9, temporal_layers=2, chord_classes=25, dropout=0
        )
        sequence, mask, window_index = fixture()
        output = model(
            sequence, mask, window_index, windows=2, tokens_per_window=4
        )
        self.assertEqual(output.embedding.shape, (2, 13))
        self.assertEqual(output.chroma_logits.shape, (2, 8, 12))
        self.assertEqual(output.chord_logits.shape, (2, 8, 25))
        self.assertEqual(output.availability.tolist(), [True, False])
        self.assertTrue(torch.equal(output.prediction_mask, mask))
        self.assertEqual(output.embedding[1].abs().sum().item(), 0.0)
        self.assertEqual(output.pooling_weights[1].sum().item(), 0.0)
        self.assertTrue(torch.all(output.chroma_logits[~mask] == 0))

    def test_masked_padding_cannot_change_valid_predictions_or_embedding(self):
        torch.manual_seed(4)
        model = TemporalHarmonyBranch(6, hidden_dim=8, temporal_layers=2, dropout=0).eval()
        sequence, mask, window_index = fixture()
        changed = sequence.clone()
        changed[~mask] = 1_000_000
        left = model(sequence, mask, window_index, windows=2, tokens_per_window=4)
        right = model(changed, mask, window_index, windows=2, tokens_per_window=4)
        self.assertTrue(torch.allclose(left.embedding, right.embedding))
        self.assertTrue(torch.allclose(left.chroma_logits[mask], right.chroma_logits[mask]))

    def test_chroma_and_chord_losses_backpropagate_with_independent_masks(self):
        model = TemporalHarmonyBranch(
            6, hidden_dim=8, temporal_layers=1, chord_classes=25, dropout=0
        )
        sequence, mask, window_index = fixture()
        output = model(sequence, mask, window_index, windows=2, tokens_per_window=4)
        chroma = torch.zeros_like(output.chroma_logits)
        chroma[..., 0] = 1
        chroma_mask = torch.zeros_like(mask)
        chroma_mask[0, :4] = True
        chord_labels = torch.zeros(mask.shape, dtype=torch.long)
        chord_mask = torch.zeros_like(mask)
        chord_mask[0, 4:6] = True
        loss = masked_soft_target_cross_entropy(
            output.chroma_logits, chroma, chroma_mask, mask
        ) + masked_hard_classification_loss(
            output.chord_logits, chord_labels, chord_mask, mask
        )
        loss.backward()
        self.assertGreater(model.input_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.embedding_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.chroma_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.chord_head.weight.grad.abs().sum().item(), 0)

    def test_no_supervision_batch_returns_differentiable_zero(self):
        logits = torch.randn(2, 3, 12, requires_grad=True)
        targets = torch.zeros_like(logits)
        missing = torch.zeros(2, 3, dtype=torch.bool)
        loss = masked_soft_target_cross_entropy(logits, targets, missing, missing)
        self.assertEqual(loss.detach().item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_window_layout_mismatch_is_rejected(self):
        model = TemporalHarmonyBranch(6)
        sequence, mask, window_index = fixture()
        window_index[0, 4] = 0
        with self.assertRaisesRegex(ValueError, "window_index"):
            model(sequence, mask, window_index, windows=2, tokens_per_window=4)

    def test_checkpoint_round_trip_preserves_outputs(self):
        torch.manual_seed(8)
        model = TemporalHarmonyBranch(6, embedding_dim=11, hidden_dim=7, dropout=0).eval()
        sequence, mask, window_index = fixture()
        expected = model(sequence, mask, window_index, windows=2, tokens_per_window=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            torch.save(model.state_dict(), path)
            restored = TemporalHarmonyBranch(
                6, embedding_dim=11, hidden_dim=7, dropout=0
            ).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
        actual = restored(sequence, mask, window_index, windows=2, tokens_per_window=4)
        self.assertTrue(torch.equal(expected.embedding, actual.embedding))
        self.assertTrue(torch.equal(expected.chroma_logits, actual.chroma_logits))


if __name__ == "__main__":
    unittest.main()
