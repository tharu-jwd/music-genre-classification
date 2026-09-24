import unittest

import numpy as np
import torch

from harmony_branch.alignment import align_chroma_to_intervals
from scripts.shared_audio_encoder import SharedAudioEncoder
from harmony_branch.losses import masked_soft_target_cross_entropy
from harmony_branch.model import TemporalHarmonyBranch


class HarmonyPipelineContractTest(unittest.TestCase):
    def test_encoder_alignment_branch_and_loss_form_one_gradient_path(self):
        torch.manual_seed(12)
        encoder = SharedAudioEncoder(output_dim=6)
        branch = TemporalHarmonyBranch(
            6, embedding_dim=9, hidden_dim=8, temporal_layers=1, dropout=0
        )
        audio = torch.randn(1, 2, 1, 8, 10)
        encoded = encoder.encode_temporal(
            audio,
            torch.tensor([[True, True]]),
            torch.tensor([[10, 5]]),
            torch.tensor([[0.0, 1.0]]),
            sample_rate=100,
            hop_length=1,
        )

        valid_times = encoded.sequence_times[0, encoded.sequence_mask[0]].detach().numpy()
        chroma = np.zeros((len(valid_times), 12), dtype=np.float32)
        chroma[np.arange(len(valid_times)), np.arange(len(valid_times)) % 12] = 1.0
        aligned = align_chroma_to_intervals(
            valid_times,
            chroma,
            np.ones(len(valid_times), dtype=bool),
            encoded.sequence_start_times[0].detach().numpy(),
            encoded.sequence_end_times[0].detach().numpy(),
            encoded.sequence_mask[0].detach().numpy(),
            max_token_duration_seconds=0.03,
        )
        self.assertEqual(aligned.coverage, 1.0)

        output = branch(
            encoded.encoded_sequence,
            encoded.sequence_mask,
            encoded.sequence_window_index,
            windows=2,
            tokens_per_window=5,
        )
        targets = torch.from_numpy(aligned.chroma).unsqueeze(0)
        target_mask = torch.from_numpy(aligned.valid).unsqueeze(0)
        loss = masked_soft_target_cross_entropy(
            output.chroma_logits,
            targets,
            target_mask,
            encoded.sequence_mask,
        )
        loss.backward()
        self.assertGreater(encoder.cnn[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(branch.chroma_head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(output.embedding.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
