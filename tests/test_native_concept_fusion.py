"""Native prediction values reach fusion unchanged, with correct masks and widths."""
from itertools import combinations

import pytest
import torch

from concept_fusion.contract import CONCEPT_ORDER, ConceptCounts
from concept_fusion.fixtures import make_branch_output
from concept_fusion.native import NativeConceptFusionModel
from concept_fusion.types import BranchBundle


def bundle():
    counts = ConceptCounts()
    return BranchBundle({name: make_branch_output(name, batch=3,
                        n_concepts=counts.for_name(name), fusion_keep=1)
                        for name in CONCEPT_ORDER}, counts)


@pytest.mark.parametrize('branches', [subset for n in range(1, 5)
                                     for subset in combinations(CONCEPT_ORDER, n)])
def test_exact_native_values_and_checkpoint_roundtrip(branches):
    model = NativeConceptFusionModel(branches).eval()
    data = bundle()
    captured = []
    hook = model.fusion[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    logits, output = model.from_bundle(data)
    hook.remove()
    expected = torch.cat([data.concept_values(name) for name in branches], dim=-1)
    torch.testing.assert_close(captured[0], expected)
    assert model.input_dim == expected.shape[1]
    assert logits.shape == (3, 6)
    assert output.fused.shape == (3, 128)
    assert not any('projection' in name for name in model.state_dict())
    restored = NativeConceptFusionModel(branches).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(logits, restored.from_bundle(data)[0])


def test_missing_branch_cannot_affect_prediction_and_all_missing_is_finite():
    model = NativeConceptFusionModel().eval()
    data = bundle()
    data.branches['timbre'].fusion_mask.zero_()
    before, _ = model.from_bundle(data)
    data.branches['timbre'].concept_values.fill_(10000)
    after, _ = model.from_bundle(data)
    torch.testing.assert_close(before, after)
    for branch in data.branches.values():
        branch.fusion_mask.zero_()
    logits, output = model.from_bundle(data)
    assert torch.isfinite(logits).all()
    assert not output.fused.any()
    assert not output.gates.any()


def test_concept_dropout_preserves_one_selected_branch_and_eval_disables_it():
    model = NativeConceptFusionModel(('instrument', 'harmony'), dropout_p=0.99)
    data = bundle()
    _, output = model.from_bundle(data, generator=torch.Generator().manual_seed(3))
    assert (output.effective_mask.sum(1) >= 1).all()
    assert not output.effective_mask[:, 1:3].any()
    model.eval()
    _, output = model.from_bundle(data)
    assert (output.effective_mask.sum(1) == 2).all()
