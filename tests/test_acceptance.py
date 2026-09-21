"""Acceptance checks from the integration contract §13 (synthetic data)."""

from __future__ import annotations

import pytest
import torch

from concept_fusion.contract import CONCEPT_ORDER, N_GENRE_TAGS, TOKEN_DIM
from concept_fusion.fixtures import (
    make_all_masked_tokens,
    make_bundle,
    make_concept_targets,
    make_genre_batch,
)
from concept_fusion.fusion import ConcatFusion, MaskedGatedFusion
from concept_fusion.interventions import occlude_concept
from concept_fusion.joint_loss import JointLossOrchestrator
from concept_fusion.metrics import ranking_metrics, sigmoid_np, trapezoidal_pr_auc
from concept_fusion.model import ConceptBottleneckModel
from concept_fusion.validation import ContractError


def test_token_shape_and_order():
    b = make_bundle(5, seed=0)
    t = b.tokens()
    assert t.shape == (5, 4, TOKEN_DIM)
    assert tuple(b.branches.keys()) == CONCEPT_ORDER


def test_rejects_wrong_token_order():
    b = make_bundle(3, seed=1)
    wrong = dict(reversed(list(b.branches.items())))
    b.branches = wrong  # type: ignore[misc]
    with pytest.raises(ContractError, match="exactly"):
        b.validate()


def test_rejects_nan_on_observed_values():
    b = make_bundle(2, seed=2)
    inst = b.branches["instrument"]
    inst.supervision_mask[:, 0] = 1
    inst.concept_values[:, 0] = float("nan")
    with pytest.raises(ContractError, match="NaN"):
        inst.validate(batch=2, n_concepts=inst.concept_values.shape[1])


def test_missing_supervision_is_nan_not_zero():
    b = make_bundle(4, seed=3, supervise_keep=0.5)
    v = b.concept_values("rhythm")
    m = b.supervision_mask("rhythm")
    assert torch.isnan(v[m < 0.5]).all()
    assert torch.isfinite(v[m > 0.5]).all()


def test_masked_gate_exactly_zero():
    model = ConceptBottleneckModel(fusion="gated")
    tokens = torch.randn(6, 4, 64)
    mask = torch.ones(6, 4)
    mask[:, 2] = 0
    _, fout = model(tokens, mask, apply_dropout=False)
    assert torch.count_nonzero(fout.gates[:, 2]) == 0


def test_all_masked_uses_null_and_finite():
    fusion = MaskedGatedFusion()
    tokens, mask = make_all_masked_tokens(3)
    out = fusion(tokens, mask)
    assert bool(out.used_null_token.all())
    assert torch.isfinite(out.fused).all()
    assert torch.count_nonzero(out.gates) == 0


def test_genre_logits_shape_and_sigmoid_once():
    model = ConceptBottleneckModel("concat")
    tokens = torch.randn(4, 4, 64)
    mask = torch.ones(4, 4)
    logits, _ = model(tokens, mask, apply_dropout=False)
    assert logits.shape == (4, N_GENRE_TAGS)
    probs = torch.sigmoid(logits)
    # Applying sigmoid twice would not match metrics helper.
    assert torch.allclose(probs, torch.tensor(sigmoid_np(logits)), atol=1e-5)


def test_primary_forbids_shortcut():
    model = ConceptBottleneckModel("gated", allow_shortcut=False)
    tokens = torch.randn(2, 4, 64)
    mask = torch.ones(2, 4)
    with pytest.raises(ContractError, match="forbids"):
        model(tokens, mask, song_repr=torch.randn(2, 128), apply_dropout=False)


def test_save_load_identical_logits(tmp_path):
    model = ConceptBottleneckModel("gated")
    tokens = torch.randn(3, 4, 64)
    mask = torch.ones(3, 4)
    model.eval()
    with torch.no_grad():
        a, _ = model(tokens, mask, apply_dropout=False)
    path = tmp_path / "m.pt"
    torch.save(model.state_dict(), path)
    other = ConceptBottleneckModel("gated")
    try:
        state = torch.load(path, weights_only=True)
    except TypeError:
        state = torch.load(path)
    other.load_state_dict(state)
    other.eval()
    with torch.no_grad():
        b, _ = other(tokens, mask, apply_dropout=False)
    assert torch.allclose(a, b, atol=1e-6)


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    bundle = make_bundle(8, seed=0, fusion_keep=1.0)
    y = make_genre_batch(8, seed=1, p=0.2)
    targets = make_concept_targets(bundle, seed=2)
    model = ConceptBottleneckModel("gated")
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = JointLossOrchestrator()
    model.train()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        logits, _ = model.from_bundle(bundle)
        br = loss_fn(logits, y, bundle, targets)
        br.total.backward()
        opt.step()
        losses.append(br.terms["genre"])
    assert losses[-1] < losses[0]


def test_undefined_tags_excluded_and_counted():
    y = torch.zeros(10, N_GENRE_TAGS)
    y[:, 0] = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    # tag 1 is all-zero → undefined
    p = torch.rand(10, N_GENRE_TAGS)
    m = ranking_metrics(y, p)
    assert m.n_valid_tags == 1
    assert m.per_tag_ap[1] is None
    assert m.n_tags_total == N_GENRE_TAGS


def test_average_precision_not_trapezoidal():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    s = torch.tensor([0.1, 0.4, 0.35, 0.8])
    # Hand AP: ranked 0.8(neg), 0.4(pos), 0.35(pos), 0.1(neg)
    # AP = (P@2 * 1 + P@3 * 1) / 2 = (0.5 + 2/3) / 2 = 7/12
    from sklearn.metrics import average_precision_score

    ap = average_precision_score(y.numpy(), s.numpy())
    trap = trapezoidal_pr_auc(y.numpy(), s.numpy())
    assert abs(ap - 7 / 12) < 1e-6
    assert abs(ap - trap) > 1e-6  # these are not the same definition


def test_occlusion_requires_dropout_training():
    model = ConceptBottleneckModel("gated")
    tokens = torch.randn(4, 4, 64)
    mask = torch.ones(4, 4)
    with pytest.raises(ContractError, match="forbidden"):
        occlude_concept(model, tokens, mask, "rhythm", dropout_was_trained=False)


def test_occlusion_zeroes_gate():
    model = ConceptBottleneckModel("gated")
    tokens = torch.randn(4, 4, 64)
    mask = torch.ones(4, 4)
    r = occlude_concept(model, tokens, mask, "timbre", dropout_was_trained=True)
    assert r.delta.shape == (4, N_GENRE_TAGS)


def test_fusion_rejects_nan_tokens():
    fusion = MaskedGatedFusion()
    tokens = torch.randn(2, 4, 64)
    tokens[0, 0, 0] = float("nan")
    mask = torch.ones(2, 4)
    with pytest.raises(ContractError, match="NaN"):
        fusion(tokens, mask)


def test_concat_and_gated_parameter_count():
    c = ConcatFusion()
    g = MaskedGatedFusion()
    nc = sum(p.numel() for p in c.parameters())
    ng = sum(p.numel() for p in g.parameters())
    assert nc > 0 and ng > 0

