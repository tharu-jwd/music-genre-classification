"""Frozen shared-architecture contract v0.1 (Thevindu integration owner).

Reject violating tensors. Do not silently reshape, reorder, impute, or reinterpret.
"""

from __future__ import annotations

from dataclasses import dataclass

CONCEPT_ORDER: tuple[str, ...] = ("instrument", "rhythm", "timbre", "harmony")
N_CONCEPTS = 4
TOKEN_DIM = 64
FUSED_DIM = 128
N_GENRE_TAGS = 87
N_INSTRUMENT_TAGS = 40

# Instrument 40 and rhythm 10 are contract-fixed. Timbre/harmony remain
# provisional until owners freeze target schemas — they only affect aux loss C_k.
PROVISIONAL_TIMBRE = 6
PROVISIONAL_HARMONY = 18

CONCEPT_DROPOUT_P = 0.15
SUPPORT_FLOOR = 10
GLOBAL_THRESHOLD_FALLBACK = 0.5

PRIMARY_METRIC = "macro_ap"
FUSION_PRIMARY = "gated"
FUSION_BASELINE = "concat"
FUSION_SECONDARY = "attention"

# Seeds: three on the paper comparison; one elsewhere (Colab budget).
MULTI_SEED_EXPERIMENTS = frozenset({"B1", "F-Concat", "F-Gated"})
N_SEEDS_FINAL = 3
N_SEEDS_OTHER = 1
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class ConceptCounts:
    instrument: int = N_INSTRUMENT_TAGS
    rhythm: int = 10
    timbre: int = PROVISIONAL_TIMBRE
    harmony: int = PROVISIONAL_HARMONY

    def for_name(self, name: str) -> int:
        if name not in CONCEPT_ORDER:
            raise KeyError(name)
        return int(getattr(self, name))

    def as_dict(self) -> dict[str, int]:
        return {k: self.for_name(k) for k in CONCEPT_ORDER}


EXPERIMENT_MATRIX: tuple[tuple[str, str, str], ...] = (
    ("B0", "legacy_cnn", "Historical anchor (upstream notebook CNN)"),
    ("B1", "direct_cnn", "Matched-protocol compact audio CNN"),
    ("C-I", "concept_instrument", "Instrument concept only"),
    ("C-R", "concept_rhythm", "Rhythm concept only"),
    ("C-T", "concept_timbre", "Timbre concept only"),
    ("C-H", "concept_harmony", "Harmony concept only"),
    ("F-Concat", "concat", "All concepts, concatenation baseline"),
    ("F-Gated", "gated", "All concepts, masked gating (primary)"),
    ("F-Attn", "attention", "All concepts, self-attention (optional)"),
    ("F-Hidden", "hidden", "Hidden branch embeddings (faithfulness tax)"),
    ("F-Shortcut", "shortcut", "Fusion plus direct audio path (bottleneck tax)"),
)
