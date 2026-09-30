"""Harmony v4: exact descriptor identities, masking, gap safety, and targets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from harmony_branch.constants import (
    CHROMA_MEAN_FEATURES,
    DESCRIPTOR_TRANSFORMS,
    HARMONY_DESCRIPTORS,
    TONNETZ_MEAN_FEATURES,
)
from harmony_branch.descriptors import HarmonyTargetTransform, tonnetz_phi
from harmony_branch.model import ChromaGroundedHarmonyBranch

HARMONY_TABLE = Path(__file__).resolve().parents[2] / "data" / "harmony_df.csv"
CHROMA = [HARMONY_DESCRIPTORS.index(n) for n in CHROMA_MEAN_FEATURES]
TONNETZ = [HARMONY_DESCRIPTORS.index(n) for n in TONNETZ_MEAN_FEATURES]


def _inputs(batch=3, windows=2, tokens_per_window=10, width=16):
    torch.manual_seed(0)
    sequence = torch.randn(batch, windows * tokens_per_window, width)
    mask = torch.ones(batch, windows * tokens_per_window, dtype=torch.bool)
    mask[1, 14:] = False  # partial second window
    mask[2] = False  # unavailable song
    index = torch.arange(windows).repeat_interleave(tokens_per_window).expand(batch, -1).clone()
    index[~mask] = -1
    return sequence, mask, index


def _model(width=16):
    return ChromaGroundedHarmonyBranch(width, hidden_dim=24, embedding_dim=20, dropout=0).eval()


def _fitted_transform(rows=200):
    rng = np.random.default_rng(3)
    chroma = rng.dirichlet(np.ones(12), size=rows)
    values = rng.uniform(0.05, 0.9, size=(rows, 45))
    values[:, CHROMA] = chroma
    values[:, TONNETZ] = chroma @ tonnetz_phi().T
    return HarmonyTargetTransform().fit(values)


def test_outputs_have_the_45_descriptor_contract_and_zero_unavailable_songs():
    model = _model()
    sequence, mask, index = _inputs()
    out = model(sequence, mask, index, windows=2, tokens_per_window=10)
    assert out.descriptor_values.shape == (3, 45)
    assert out.descriptor_raw.shape == (3, 45)
    assert out.chroma_logits.shape == (3, 20, 12)
    assert out.availability.tolist() == [True, True, False]
    assert out.descriptor_values[2].abs().sum() == 0
    assert out.embedding[2].abs().sum() == 0
    assert torch.isfinite(out.descriptor_values).all()


def test_chroma_and_tonnetz_means_follow_the_extractor_identities():
    model = _model()
    model.set_target_transform(_fitted_transform())
    sequence, mask, index = _inputs()
    out = model(sequence, mask, index, windows=2, tokens_per_window=10)
    raw = out.descriptor_raw[:2].double()
    torch.testing.assert_close(raw[:, CHROMA].sum(-1), torch.ones(2, dtype=torch.float64))
    expected_mean = torch.softmax(out.chroma_logits[0][mask[0]], -1).mean(0).double()
    torch.testing.assert_close(raw[0, CHROMA], expected_mean, atol=1e-5, rtol=1e-4)
    phi = torch.tensor(tonnetz_phi())
    torch.testing.assert_close(raw[:, TONNETZ], raw[:, CHROMA] @ phi.T, atol=1e-5, rtol=1e-4)


def test_padding_cannot_change_predictions():
    model = _model()
    sequence, mask, index = _inputs()
    changed = sequence.clone()
    changed[~mask] = 1e6
    left = model(sequence, mask, index, windows=2, tokens_per_window=10)
    right = model(changed, mask, index, windows=2, tokens_per_window=10)
    torch.testing.assert_close(left.descriptor_values, right.descriptor_values)
    torch.testing.assert_close(left.embedding, right.embedding)


def test_temporal_context_never_crosses_the_gap_between_windows():
    model = _model()
    sequence, mask, index = _inputs()
    changed = sequence.clone()
    changed[:, 10:] = torch.randn_like(changed[:, 10:])  # second window only
    left = model(sequence, mask, index, windows=2, tokens_per_window=10)
    right = model(changed, mask, index, windows=2, tokens_per_window=10)
    # Per-token chroma inside window 0 depends only on window 0.
    torch.testing.assert_close(left.chroma_logits[:, :10], right.chroma_logits[:, :10])


def test_every_descriptor_gets_gradient_from_its_loss():
    model = ChromaGroundedHarmonyBranch(16, hidden_dim=24, embedding_dim=20, dropout=0)
    model.set_target_transform(_fitted_transform())
    sequence, mask, index = _inputs()
    out = model(sequence, mask, index, windows=2, tokens_per_window=10)
    out.descriptor_values[:2, CHROMA + TONNETZ].sum().backward(retain_graph=True)
    assert model.chroma_head[-1].weight.grad.abs().sum() > 0
    model.zero_grad()
    learned = [i for i in range(45) if i not in CHROMA + TONNETZ]
    out.descriptor_values[:2, learned].sum().backward()
    assert model.descriptor_head[-1].weight.grad.abs().sum() > 0
    assert model.statistic_gain.grad.abs().sum() > 0


def test_target_transform_round_trips_and_rejects_foreign_state():
    transform = _fitted_transform()
    values = np.random.default_rng(5).uniform(0.05, 0.9, size=(7, 45))
    np.testing.assert_allclose(transform.inverse_transform(transform.transform(values)), values,
                               rtol=1e-5, atol=1e-6)
    state = transform.state_dict()
    assert state["transforms"] == [DESCRIPTOR_TRANSFORMS[n] for n in HARMONY_DESCRIPTORS]
    state["feature_names"] = state["feature_names"][::-1]
    with pytest.raises(ValueError, match="feature order"):
        HarmonyTargetTransform.from_state_dict(state)


@pytest.mark.skipif(not HARMONY_TABLE.is_file(), reason="data/harmony_df.csv not available")
def test_real_table_satisfies_the_tonnetz_identity_and_fits():
    table = pd.read_csv(HARMONY_TABLE)
    values = table[list(HARMONY_DESCRIPTORS)].to_numpy()
    assert values.shape == (len(table), 45) and np.isfinite(values).all()
    np.testing.assert_allclose(values[:, TONNETZ], values[:, CHROMA] @ tonnetz_phi().T, atol=1e-7)
    z = HarmonyTargetTransform().fit(values).transform(values)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-3)
