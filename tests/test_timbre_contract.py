"""Timbre v2 contract: 35 standardized concepts, no fusion_token, fusion owns Linear(35,64)."""

import torch
import pytest

from concept_fusion.contract import N_TIMBRE_CONCEPTS, TIMBRE_FEATURES
from concept_fusion.fixtures import make_bundle
from concept_fusion.projections import TokenAssembler
from concept_fusion.timbre_adapter import from_timbre_branch
from concept_fusion.validation import ContractError


def test_published_timbre_feature_order():
    assert len(TIMBRE_FEATURES) == N_TIMBRE_CONCEPTS == 35
    assert TIMBRE_FEATURES[0] == "spectral_centroid_mean"
    assert TIMBRE_FEATURES[7] == "hnr_mean_db"
    assert TIMBRE_FEATURES[-1] == "mfcc_13_std"


def test_rejects_timbre_fusion_token():
    b = make_bundle(2, seed=0)
    tim = b.branches["timbre"]
    assert tim.concept_values.shape == (2, 35)
    assert tim.fusion_token is None
    tim.fusion_token = torch.randn(2, 64)
    with pytest.raises(ContractError, match="must not return fusion_token"):
        tim.validate(batch=2, n_concepts=35)


def test_adapter_rejects_h_audio_and_fusion_token():
    z = torch.randn(3, 35)
    with pytest.raises(ContractError, match="fusion_token"):
        from_timbre_branch({"z_timbre": z, "fusion_token": torch.randn(3, 64)})
    with pytest.raises(ContractError, match="h_audio"):
        from_timbre_branch({"z_timbre": z, "h_audio": torch.randn(3, 128)})


def test_adapter_and_projection():
    z = torch.randn(4, 35)
    br = from_timbre_branch({"z_timbre": z, "feature_names": TIMBRE_FEATURES})
    assert br.fusion_token is None
    assert br.concept_values.shape == (4, 35)
    bundle = make_bundle(4, seed=1, fusion_keep=1.0)
    bundle.branches["timbre"] = br
    bundle.validate()
    tokens = TokenAssembler()(bundle)
    assert tokens.shape == (4, 4, 64)
    assert torch.isfinite(tokens).all()


def test_mask_applied_after_timbre_projection():
    bundle = make_bundle(3, seed=2, fusion_keep=1.0)
    bundle.branches["timbre"].fusion_mask.zero_()
    tokens = TokenAssembler()(bundle)
    assert torch.count_nonzero(tokens[:, 2]) == 0


def test_timbre_values_always_finite():
    tim = make_bundle(2, seed=3).branches["timbre"]
    tim.concept_values[0, 0] = float("nan")
    with pytest.raises(ContractError, match="NaN"):
        tim.validate(batch=2, n_concepts=35)
