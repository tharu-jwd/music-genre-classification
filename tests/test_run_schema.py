import pytest

from concept_fusion.contract import FUSION_CONTRACT_VERSION, PRIMARY_FUSION_INPUT_MODE
from concept_fusion.run_schema import seeds_for, validate_fusion_checkpoint_metadata


def test_seed_policy():
    assert seeds_for("B1") == (0, 1, 2)
    assert seeds_for("F-Concat") == (0, 1, 2)
    assert seeds_for("F-Gated") == (0, 1, 2)
    assert seeds_for("F-Embedding") == (0, 1, 2)
    assert seeds_for("C-I") == (0,)
    assert seeds_for("F-Shortcut") == (0,)
    assert seeds_for("F-Hidden") == (0,)


def test_checkpoint_metadata_rejects_legacy_and_wrong_mode():
    with pytest.raises(RuntimeError, match="fusion contract"):
        validate_fusion_checkpoint_metadata({}, expected_input_mode=PRIMARY_FUSION_INPUT_MODE)
    with pytest.raises(RuntimeError, match="input mode"):
        validate_fusion_checkpoint_metadata(
            {
                "fusion_contract_version": FUSION_CONTRACT_VERSION,
                "fusion_input_mode": "embedding_fusion",
            },
            expected_input_mode=PRIMARY_FUSION_INPUT_MODE,
        )
    validate_fusion_checkpoint_metadata(
        {
            "fusion_contract_version": FUSION_CONTRACT_VERSION,
            "fusion_input_mode": PRIMARY_FUSION_INPUT_MODE,
        },
        expected_input_mode=PRIMARY_FUSION_INPUT_MODE,
    )
