import unittest

import torch
from torch import nn

from shared_encoder import (
    SHARED_ENCODER_ARCHITECTURE,
    SharedAudioEncoder,
)
from scripts.shared_audio_encoder import SharedAudioEncoder as CompatibilityEncoder


class LegacyColabEncoder(nn.Module):
    """Previous checkpoint architecture, retained only for rejection coverage."""

    def __init__(self, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, output_dim)


def _metadata(batch, windows, frames, *, valid_frames=None, starts=None):
    if valid_frames is None:
        valid_frames = torch.full((batch, windows), frames, dtype=torch.long)
    if starts is None:
        starts = torch.arange(windows).float().repeat(batch, 1) * 30.0
    return valid_frames > 0, valid_frames, starts


class SharedAudioEncoderTest(unittest.TestCase):
    def test_legacy_script_import_is_the_same_modular_class(self):
        self.assertIs(CompatibilityEncoder, SharedAudioEncoder)

    def test_single_forward_returns_all_branch_representations(self):
        torch.manual_seed(3)
        model = SharedAudioEncoder()
        x = torch.randn(2, 3, 1, 8, 10)
        valid = torch.tensor([[10, 3, 0], [7, 0, 0]])
        mask = valid > 0
        starts = torch.tensor([[0.0, 20.0, 0.0], [4.0, 0.0, 0.0]])

        result = model(x, mask, valid, starts, sample_rate=100, hop_length=1)

        self.assertEqual(result.encoded_sequence.shape, (2, 15, 128))
        self.assertEqual(result.window_repr.shape, (2, 3, 128))
        self.assertEqual(result.pooled_song.shape, (2, 128))
        self.assertIs(result.song_repr, result.pooled_song)
        self.assertEqual(result.sequence_mask[0].sum().item(), 7)
        self.assertEqual(result.sequence_mask[1].sum().item(), 4)
        self.assertEqual(result.sequence_window_index[0].tolist(), [0] * 5 + [1] * 2 + [-1] * 8)
        self.assertAlmostEqual(result.sequence_start_times[0, 5].item(), 20.0, places=6)
        self.assertAlmostEqual(result.sequence_end_times[0, 6].item(), 20.03, places=5)
        self.assertTrue(torch.all(result.encoded_sequence[~result.sequence_mask] == 0))
        self.assertTrue(torch.all(result.sequence_times[~result.sequence_mask] == 0))
        self.assertTrue(torch.all(result.window_repr[~mask] == 0))
        self.assertTrue(result.availability.all())

    def test_variable_window_counts_and_odd_frame_counts(self):
        model = SharedAudioEncoder(output_dim=11).eval()
        for windows, frames in ((1, 9), (2, 10), (4, 15)):
            with self.subTest(windows=windows, frames=frames):
                x = torch.randn(2, windows, 1, 12, frames)
                mask, valid, starts = _metadata(2, windows, frames)
                result = model.encode_temporal(x, mask, valid, starts)
                tokens_per_window = (frames + 1) // 2
                self.assertEqual(
                    result.encoded_sequence.shape,
                    (2, windows * tokens_per_window, 11),
                )
                self.assertEqual(result.window_repr.shape, (2, windows, 11))

    def test_pooling_uses_only_valid_tokens_and_padded_values_are_irrelevant(self):
        torch.manual_seed(7)
        model = SharedAudioEncoder(output_dim=9).eval()
        x = torch.randn(1, 2, 1, 8, 10)
        valid = torch.tensor([[5, 0]])
        mask = valid > 0
        starts = torch.tensor([[3.0, 0.0]])
        changed = x.clone()
        changed[:, 0, :, :, 5:] = 1_000_000
        changed[:, 1] = -1_000_000

        left = model.encode_temporal(x, mask, valid, starts)
        right = model.encode_temporal(changed, mask, valid, starts)

        torch.testing.assert_close(left.encoded_sequence, right.encoded_sequence)
        torch.testing.assert_close(left.window_repr, right.window_repr)
        torch.testing.assert_close(left.pooled_song, right.pooled_song)
        manual_song = left.encoded_sequence[left.sequence_mask].mean(dim=0)
        manual_window = left.encoded_sequence[0, :3].mean(dim=0)
        torch.testing.assert_close(left.pooled_song[0], manual_song)
        torch.testing.assert_close(left.window_repr[0, 0], manual_window)

    def test_all_masked_example_returns_finite_exact_zeros(self):
        model = SharedAudioEncoder(output_dim=13)
        x = torch.randn(2, 2, 1, 8, 10)
        valid = torch.tensor([[10, 3], [0, 0]])
        mask = valid > 0
        starts = torch.tensor([[0.0, 30.0], [0.0, 0.0]])
        result = model.encode_temporal(x, mask, valid, starts)
        self.assertEqual(result.availability.tolist(), [True, False])
        self.assertEqual(result.encoded_sequence[1].count_nonzero().item(), 0)
        self.assertEqual(result.window_repr[1].count_nonzero().item(), 0)
        self.assertEqual(result.pooled_song[1].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(result.pooled_song).all())

    def test_token_times_are_ordered_within_windows_and_preserve_song_gaps(self):
        model = SharedAudioEncoder(output_dim=5)
        x = torch.randn(1, 2, 1, 8, 8)
        mask, valid, starts = _metadata(
            1,
            2,
            8,
            starts=torch.tensor([[2.0, 40.0]]),
        )
        result = model.encode_temporal(
            x, mask, valid, starts, sample_rate=100, hop_length=1
        )
        for window in (0, 1):
            selected = result.sequence_window_index[0] == window
            token_starts = result.sequence_start_times[0, selected]
            token_ends = result.sequence_end_times[0, selected]
            self.assertTrue(torch.all(token_starts[1:] > token_starts[:-1]))
            self.assertTrue(torch.all(token_ends > token_starts))
            torch.testing.assert_close(token_ends - token_starts, torch.full_like(token_starts, 0.02))
        self.assertAlmostEqual(result.sequence_start_times[0, 4].item(), 40.0, places=6)

    def test_cnn_does_not_mix_separate_sampled_windows(self):
        torch.manual_seed(11)
        model = SharedAudioEncoder(output_dim=7).eval()
        x = torch.randn(1, 2, 1, 12, 10)
        mask, valid, starts = _metadata(1, 2, 10)
        changed = x.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 1000

        left = model.encode_temporal(x, mask, valid, starts)
        right = model.encode_temporal(changed, mask, valid, starts)
        first_window = left.sequence_window_index == 0
        torch.testing.assert_close(
            left.encoded_sequence[first_window],
            right.encoded_sequence[first_window],
        )

    def test_pooled_and_temporal_gradients_reach_shared_cnn(self):
        for path in ("pooled", "temporal"):
            with self.subTest(path=path):
                model = SharedAudioEncoder(output_dim=8)
                x = torch.randn(1, 2, 1, 8, 12)
                mask, valid, starts = _metadata(1, 2, 12)
                result = model.encode_temporal(x, mask, valid, starts)
                value = result.pooled_song if path == "pooled" else result.encoded_sequence
                value.square().mean().backward()
                self.assertIsNotNone(model.cnn[0].weight.grad)
                self.assertIsNotNone(model.proj.weight.grad)
                self.assertGreater(model.cnn[0].weight.grad.abs().sum().item(), 0)

    def test_forward_without_metadata_preserves_window_embedding_convenience(self):
        model = SharedAudioEncoder(output_dim=6).eval()
        x = torch.randn(2, 3, 1, 8, 10)
        result = model(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 3, 6))

    def test_v2_checkpoint_formats_round_trip(self):
        source = SharedAudioEncoder(output_dim=6)
        for name, checkpoint in (
            (
                "shared_cnn_v2",
                {
                    "encoder_architecture": SHARED_ENCODER_ARCHITECTURE,
                    "model": {key: value.clone() for key, value in source.state_dict().items()},
                },
            ),
            (
                "shared_cnn_v2_nested",
                {
                    "encoder_architecture": SHARED_ENCODER_ARCHITECTURE,
                    "model": {f"enc.{key}": value.clone() for key, value in source.state_dict().items()},
                },
            ),
        ):
            with self.subTest(name=name):
                restored = SharedAudioEncoder(output_dim=6)
                self.assertEqual(restored.load_instrument_pretraining(checkpoint), name)
                for left, right in zip(source.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(left, right))

    def test_legacy_checkpoint_is_rejected_instead_of_partially_loaded(self):
        model = SharedAudioEncoder()
        with self.assertRaisesRegex(ValueError, "legacy two-convolution"):
            model.load_instrument_pretraining(LegacyColabEncoder().state_dict())

    def test_mask_and_valid_lengths_must_agree(self):
        model = SharedAudioEncoder(output_dim=4)
        x = torch.randn(1, 1, 1, 8, 10)
        with self.assertRaisesRegex(ValueError, "true exactly"):
            model.encode_temporal(
                x,
                torch.tensor([[False]]),
                torch.tensor([[10]]),
                torch.tensor([[0.0]]),
            )

    def test_freeze_and_unfreeze_are_explicit(self):
        model = SharedAudioEncoder()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertIs(model.freeze(), model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertIs(model.unfreeze(), model)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
