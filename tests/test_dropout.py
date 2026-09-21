import torch
from concept_fusion.dropout import apply_concept_dropout


def test_dropout_never_all_four_when_all_were_on():
    mask = torch.ones(64, 4)
    g = torch.Generator().manual_seed(0)
    out = apply_concept_dropout(mask, p=0.9, generator=g)
    assert (out.sum(dim=1) >= 1).all()
