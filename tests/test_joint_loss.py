from concept_fusion.contract import CONCEPT_ORDER
from concept_fusion.fixtures import make_bundle, make_concept_targets, make_genre_batch
from concept_fusion.joint_loss import JointLossOrchestrator
from concept_fusion.model import ConceptBottleneckModel
import torch


def test_missing_concept_labels_keep_genre_batch():
    bundle = make_bundle(8, seed=0, supervise_keep=0.0)  # no concept labels
    y = make_genre_batch(8, seed=1)
    targets = make_concept_targets(bundle, seed=2)
    model = ConceptBottleneckModel("concat")
    logits, _ = model.from_bundle(bundle, apply_dropout=False)
    br = JointLossOrchestrator()(logits, y, bundle, targets)
    assert br.n_observed["genre"] == 8 * 6
    assert br.n_observed["instrument"] == 0
    assert torch.isfinite(br.total)
    # Genre term still trains every track.
    assert br.terms["genre"] > 0
