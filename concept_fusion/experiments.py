"""Frozen experiment matrix. One command runs every spec the fusion stack owns."""

from __future__ import annotations

from dataclasses import dataclass

from concept_fusion.contract import CONCEPT_DROPOUT_P, CONCEPT_ORDER
from concept_fusion.run_schema import seeds_for


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    purpose: str
    model_kind: str = "bottleneck"
    fusion: str = "gated"
    enabled_concepts: tuple[str, ...] = CONCEPT_ORDER
    allow_shortcut: bool = False
    use_hidden: bool = False
    dropout_p: float = CONCEPT_DROPOUT_P
    allow_no_dropout: bool = False
    aux: bool = True
    use_kendall: bool = False

    @property
    def dropout_was_trained(self) -> bool:
        return self.model_kind == "bottleneck" and self.dropout_p > 0 and not self.allow_no_dropout


def experiment_specs() -> tuple[ExperimentSpec, ...]:
    """All fusion-owned experiments, including required ablations.

    B0 (legacy notebook CNN) is not runnable from this package.
    B1 is a fixture song_repr MLP until Dehan's compact CNN is wired.
    """
    i, r, t, h = CONCEPT_ORDER
    return (
        ExperimentSpec(
            "B1",
            "Matched-protocol compact audio baseline (fixture stand-in)",
            model_kind="direct",
            fusion="direct",
            dropout_p=0.0,
            allow_no_dropout=True,
            aux=False,
        ),
        ExperimentSpec("C-I", "Instrument concept only", enabled_concepts=(i,)),
        ExperimentSpec("C-R", "Rhythm concept only", enabled_concepts=(r,)),
        ExperimentSpec("C-T", "Timbre concept only", enabled_concepts=(t,)),
        ExperimentSpec("C-H", "Harmony concept only", enabled_concepts=(h,)),
        ExperimentSpec("F-Concat", "All concepts, concatenation baseline", fusion="concat"),
        ExperimentSpec("F-Gated", "All concepts, masked gating (primary)", fusion="gated"),
        ExperimentSpec("F-Attn", "All concepts, self-attention", fusion="attention"),
        ExperimentSpec("F-Hidden", "Hidden branch embeddings (faithfulness tax)", use_hidden=True),
        ExperimentSpec("F-Shortcut", "Fusion plus direct audio path (bottleneck tax)", allow_shortcut=True),
        ExperimentSpec("F-Gated-NoAux", "Gated fusion, no auxiliary concept losses", aux=False),
        ExperimentSpec("F-Gated-Kendall", "Gated fusion, Kendall uncertainty weights", use_kendall=True),
        ExperimentSpec(
            "F-Gated-NoDropout",
            "Gated fusion without concept dropout (no occlusion claims)",
            dropout_p=0.0,
            allow_no_dropout=True,
        ),
        ExperimentSpec("F-Gated-no-I", "Leave-one-out instrument", enabled_concepts=(r, t, h)),
        ExperimentSpec("F-Gated-no-R", "Leave-one-out rhythm", enabled_concepts=(i, t, h)),
        ExperimentSpec("F-Gated-no-T", "Leave-one-out timbre", enabled_concepts=(i, r, h)),
        ExperimentSpec("F-Gated-no-H", "Leave-one-out harmony", enabled_concepts=(i, r, t)),
        ExperimentSpec("F-Inc-IR", "Incremental instrument+rhythm", enabled_concepts=(i, r)),
        ExperimentSpec("F-Inc-IRT", "Incremental instrument+rhythm+timbre", enabled_concepts=(i, r, t)),
    )


def jobs_for(*, quick: bool = False) -> list[tuple[ExperimentSpec, int]]:
    jobs: list[tuple[ExperimentSpec, int]] = []
    for spec in experiment_specs():
        seeds = (0,) if quick else seeds_for(spec.experiment_id)
        for seed in seeds:
            jobs.append((spec, seed))
    return jobs
