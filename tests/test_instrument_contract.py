"""Instrument v2 contract published on main: no fusion_token, fusion owns Linear(41,64)."""

import torch

from concept_fusion.contract import INSTRUMENT_TAGS, N_INSTRUMENT_TAGS
from concept_fusion.fixtures import make_bundle
from concept_fusion.instrument_adapter import from_instrument_branch
from concept_fusion.projections import TokenAssembler
from concept_fusion.types import BranchOutput
from concept_fusion.validation import ContractError
import pytest


@pytest.fixture(scope="module")
def notebook_scope():
    from instrument_branch.scripts.validate_instrument_branch_notebook import notebook_scope

    scope, _ = notebook_scope()
    return scope


def test_standalone_branch_matches_fusion_vocabulary(notebook_scope):
    assert tuple(notebook_scope["VOCAB"]) == INSTRUMENT_TAGS
    model = notebook_scope["InstrumentBranch"]().eval()
    bundle = make_bundle(2, seed=4)
    bundle.branches["instrument"] = from_instrument_branch(model(torch.randn(2, 128)))
    bundle.validate()
    assert TokenAssembler()(bundle).shape == (2, 4, 64)


def test_ukulele_target_and_filtered_split_mask(notebook_scope):
    arrays = notebook_scope["annotation_arrays"]
    index = INSTRUMENT_TAGS.index("ukulele")
    targets, mask = arrays(["1"], {"1": {"tags": {"instrument---ukulele"}}})
    assert targets.shape == mask.shape == (1, 41)
    assert targets[0, index] == mask[0, index] == 1
    _, filtered_mask = arrays(
        ["1"], {"1": {"tags": {"instrument---guitar"}}},
        observed_tags=notebook_scope["OFFICIAL_SPLIT_TAGS"],
    )
    assert filtered_mask.sum() == 40
    assert filtered_mask[0, index] == 0


def test_local_annotations_preserve_ukulele_and_custom_splits(notebook_scope, tmp_path):
    import pandas as pd
    import numpy as np

    labels = pd.DataFrame(np.zeros((3, 41), dtype=int), columns=INSTRUMENT_TAGS)
    labels.insert(0, "TRACK_ID", ["track_1", "track_2", "track_3"])
    labels.loc[0, "ukulele"] = 1
    labels.loc[1, "guitar"] = 1
    # Deliberately shuffle both rows and columns: IDs and vocabulary determine order.
    labels.iloc[::-1, ::-1].to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"track_id": ["track_2", "track_1", "track_3", "track_4"],
                  "split": ["test", "train", "validation", "train"]}).to_csv(tmp_path / "splits.csv", index=False)
    config = dict(notebook_scope["CFG"], instrument_csv=str(tmp_path / "labels.csv"),
                  manifest=str(tmp_path / "splits.csv"), output=str(tmp_path / "audit"))
    manifest, targets, mask, _ = notebook_scope["audit_dataset"](config)
    j = INSTRUMENT_TAGS.index("ukulele")
    assert manifest.split.tolist() == ["test", "train", "validation", "train"]
    assert targets[:, j].tolist() == [0, 1, 0, 0]
    assert mask[:, j].tolist() == [1, 1, 0, 0]
    logits = torch.zeros(4, 41, requires_grad=True)
    notebook_scope["masked_bce"](logits, torch.from_numpy(targets), torch.from_numpy(mask)).backward()
    assert logits.grad[1, j] < 0  # Ukulele positives contribute a learning signal.
    assert logits.grad[0, j] > 0
    assert logits.grad[2:, j].count_nonzero() == 0
    labels.loc[0, "ukulele"] = 2
    labels.to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="binary"):
        notebook_scope["audit_dataset"](config)


def test_training_rejects_missing_project_cohort_labels(notebook_scope, tmp_path):
    config = dict(notebook_scope["CFG"], instrument_csv=None,
                  manifest=str(tmp_path / "splits.csv"), output=str(tmp_path / "audit"))
    with pytest.raises(ValueError, match="7,324-track project split"):
        notebook_scope["audit_dataset"](config)


def test_official_instrument_vocabulary():
    assert len(INSTRUMENT_TAGS) == N_INSTRUMENT_TAGS == 41
    assert INSTRUMENT_TAGS == tuple(sorted(INSTRUMENT_TAGS))
    assert INSTRUMENT_TAGS[0] == "accordion"
    assert INSTRUMENT_TAGS[-1] == "voice"


def test_rejects_instrument_fusion_token():
    b = make_bundle(2, seed=0)
    inst = b.branches["instrument"]
    inst.fusion_token = torch.randn(2, 64)
    with pytest.raises(ContractError, match="must not return fusion_token"):
        inst.validate(batch=2, n_concepts=41)


def test_adapter_rejects_fusion_token_key():
    with pytest.raises(ContractError, match="must not return fusion_token"):
        from_instrument_branch(
            {
                "concept_values": torch.rand(3, 41),
                "logits": torch.randn(3, 41),
                "supervision_mask": torch.ones(3, 41),
                "fusion_mask": torch.ones(3, 1),
                "fusion_token": torch.randn(3, 64),
            }
        )


def test_adapter_and_projection():
    raw = {
        "concept_values": torch.rand(4, 41).clamp(0.02, 0.98),
        "logits": torch.randn(4, 41),
        "supervision_mask": torch.ones(4, 41),
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
        inst.validate(batch=2, n_concepts=41)
