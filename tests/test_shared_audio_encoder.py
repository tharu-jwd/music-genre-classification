import unittest

import torch
from torch import nn

from scripts.shared_audio_encoder import SharedAudioEncoder


class LegacyColabEncoder(nn.Module):
    def __init__(self, output_dim=7):
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

    def forward(self, x):
        batch, windows = x.shape[:2]
        values = self.cnn(x.reshape(batch * windows, *x.shape[2:])).flatten(1)
        return self.proj(values).reshape(batch, windows, -1)


class SharedAudioEncoderTest(unittest.TestCase):
    def test_temporal_output_preserves_fine_times_masks_and_window_gaps(self):
        torch.manual_seed(3)
        model = SharedAudioEncoder(output_dim=8)
        x = torch.randn(2, 3, 1, 8, 10)
        window_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        valid_frames = torch.tensor([[10, 3, 0], [7, 0, 0]])
        starts = torch.tensor([[0.0, 20.0, 0.0], [4.0, 0.0, 0.0]])

        result = model.encode_temporal(
            x, window_mask, valid_frames, starts, sample_rate=100, hop_length=1
        )

        self.assertEqual(result.encoded_sequence.shape, (2, 15, 8))
        self.assertEqual(result.sequence_mask[0].sum().item(), 7)
        self.assertEqual(result.sequence_mask[1].sum().item(), 4)
        self.assertEqual(result.sequence_window_index[0].tolist(), [0] * 5 + [1] * 2 + [-1] * 8)
        self.assertAlmostEqual(result.sequence_start_times[0, 5].item(), 20.0, places=6)
        self.assertAlmostEqual(result.sequence_end_times[0, 6].item(), 20.03, places=5)
        self.assertTrue(torch.all(result.encoded_sequence[~result.sequence_mask] == 0))
        self.assertTrue(torch.all(result.sequence_times[~result.sequence_mask] == 0))
        self.assertTrue(result.availability.all())

    def test_temporal_gradients_reach_the_shared_convolution_and_projection(self):
        model = SharedAudioEncoder(output_dim=5)
        x = torch.randn(1, 2, 1, 8, 12, requires_grad=True)
        result = model.encode_temporal(
            x,
            torch.tensor([[1, 1]], dtype=torch.bool),
            torch.tensor([[12, 5]]),
            torch.tensor([[0.0, 10.0]]),
        )
        result.pooled_song.square().mean().backward()
        self.assertIsNotNone(model.cnn[0].weight.grad)
        self.assertIsNotNone(model.proj.weight.grad)
        self.assertGreater(model.cnn[0].weight.grad.abs().sum().item(), 0)

    def test_window_forward_matches_legacy_instrument_encoder_exactly(self):
        torch.manual_seed(5)
        legacy = LegacyColabEncoder(output_dim=7)
        shared = SharedAudioEncoder(output_dim=7)
        loaded_format = shared.load_instrument_pretraining(legacy.state_dict())
        x = torch.randn(2, 3, 1, 8, 10)
        self.assertEqual(loaded_format, "colab_v1")
        self.assertTrue(torch.allclose(shared(x), legacy(x), atol=1e-6))

    def test_kaggle_nested_checkpoint_prefix_is_supported(self):
        source = SharedAudioEncoder(output_dim=6)
        nested = {"model": {f"enc.{key}": value.clone() for key, value in source.state_dict().items()}}
        restored = SharedAudioEncoder(output_dim=6)
        self.assertEqual(restored.load_instrument_pretraining(nested), "kaggle_v1")
        for left, right in zip(source.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(left, right))

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

    def test_incompatible_checkpoint_is_rejected(self):
        model = SharedAudioEncoder(output_dim=4)
        with self.assertRaisesRegex(ValueError, "complete compatible"):
            model.load_instrument_pretraining({"model": {"head.weight": torch.ones(2, 2)}})


if __name__ == "__main__":
    unittest.main()
