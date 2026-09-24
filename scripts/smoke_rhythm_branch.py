"""CPU-only synthetic contract smoke test; never a research evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rhythm_branch" / "src"))

from rhythm_branch.losses import masked_huber_loss
from rhythm_branch.model import RhythmBranch
from scripts.shared_audio_encoder import SharedAudioEncoder


def main() -> None:
    torch.manual_seed(0)
    encoder = SharedAudioEncoder(output_dim=128)
    branch = RhythmBranch()
    mel = torch.randn(2, 2, 1, 16, 32)
    temporal = encoder.encode_temporal(
        mel,
        torch.ones(2, 2, dtype=torch.bool),
        torch.full((2, 2), 32),
        torch.tensor([[0.0, 30.0], [0.0, 30.0]]),
    )
    output = branch(
        temporal.encoded_sequence,
        temporal.sequence_mask,
        temporal.sequence_window_index,
    )
    synthetic_targets = torch.randn(2, 10)
    loss = masked_huber_loss(output.predictions, synthetic_targets, torch.ones(2, 10))
    loss.backward()
    assert output.embedding.shape == (2, 64)
    assert output.predictions.shape == (2, 10)
    assert encoder.cnn[0].weight.grad is not None
    print("synthetic rhythm contract smoke passed; no dataset score was computed")


if __name__ == "__main__":
    main()
