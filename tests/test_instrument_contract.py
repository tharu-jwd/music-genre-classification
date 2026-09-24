"""Instrument v2 contract published on main: no fusion_token, fusion owns Linear(40,64)."""

import torch

from concept_fusion.contract import INSTRUMENT_TAGS, N_INSTRUMENT_TAGS
from concept_fusion.fixtures import make_bundle
from concept_fusion.instrument_adapter import from_instrument_branch
from concept_fusion.projections import TokenAssembler
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError
import pytest


def test_official_instrument_vocabulary():
    assert len(INSTRUMENT_TAGS) == N_INSTRUMENT_TAGS == 40
    assert INSTRUMENT_TAGS == tuple(sorted(INSTRUMENT_TAGS))
    assert INSTRUMENT_TAGS[0] == "instrument---accordion"
    assert INSTRUMENT_TAGS[-1] == "instrument---voice"


def test_rejects_instrument_fusion_token():
    b = make_bundle(2, seed=0)
    inst = b.branches["instrument"]
    inst.fusion_token = torch.randn(2, 64)
    with pytest.raises(ContractError, match="must not return fusion_token"):
        inst.validate(batch=2, n_concepts=40)


def test_adapter_rejects_fusion_token_key():
    with pytest.raises(ContractError, match="must not return fusion_token"):
        from_instrument_branch(
            {
                "concept_values": torch.rand(3, 40),
                "logits": torch.randn(3, 40),
                "supervision_mask": torch.ones(3, 40),
                "fusion_mask": torch.ones(3, 1),
                "fusion_token": torch.randn(3, 64),
            }
        )


def test_adapter_and_projection():
    raw = {
        "concept_values": torch.rand(4, 40).clamp(0.02, 0.98),
        "logits": torch.randn(4, 40),
        "supervision_mask": torch.ones(4, 40),
        "fusion_mask": torch.ones(4, 1),
        "diagnostics": {"hidden": torch.randn(4, 128)},
    }
    br = from_instrument_branch(raw)
    assert br.fusion_token is None
    assert br.hidden_token.shape == (4, 128)
    assert not br.hidden_token.requires_grad
    bundle = make_bundle(4, seed=1, fusion_keep=1.0)
    bundle.branches["instrument"] = br
    bundle.validate()
    asm = TokenAssembler()
    tokens = asm(bundle)
    assert tokens.shape == (4, 4, 64)
    assert torch.isfinite(tokens).all()


def test_mask_applied_after_instrument_projection():
    bundle = make_bundle(3, seed=2, fusion_keep=1.0)
    bundle.branches["instrument"].fusion_mask.zero_()
    tokens = TokenAssembler()(bundle)
    assert torch.count_nonzero(tokens[:, 0]) == 0
    assert torch.isfinite(tokens[:, 1:]).all()


def test_instrument_probs_always_finite():
    inst = make_bundle(2, seed=3).branches["instrument"]
    inst.concept_values[0, 0] = float("nan")
    with pytest.raises(ContractError, match="NaN"):
        inst.validate(batch=2, n_concepts=40)
