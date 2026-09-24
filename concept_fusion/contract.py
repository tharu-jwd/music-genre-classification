"""Shared-architecture contract v0.2.

Reject violating tensors. Do not silently reshape, reorder, impute, or reinterpret.

Instrument v2: 40 probabilities + logits, no fusion token. Fusion owns Linear(40, 64).
Timbre v2 (merged from `timbre_branch`): 35 standardized concepts, no fusion token.
Fusion owns Linear(35, 64).
Harmony v1: temporal 12-bin chroma predictions plus an independently configurable
song embedding. Fusion owns the embedding-to-64 projection. Chords are an optional
25-class temporal auxiliary target, never part of an 18-value song summary.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

CONCEPT_ORDER: tuple[str, ...] = ("instrument", "rhythm", "timbre", "harmony")
N_CONCEPTS = 4
TOKEN_DIM = 64
FUSED_DIM = 128
N_GENRE_TAGS = 87
N_INSTRUMENT_TAGS = 40
N_TIMBRE_CONCEPTS = 35
N_HARMONY_CHROMA = 12
N_HARMONY_CHORDS = 25
DEFAULT_HARMONY_EMBEDDING_DIM = 32
INSTRUMENT_HIDDEN_DIM = 128
# Published branches that do not own a 64-D token. Fusion projects them.
BRANCHES_WITHOUT_FUSION_TOKEN: frozenset[str] = frozenset(
    {"instrument", "timbre", "harmony"}
)


def _load_instrument_tags() -> tuple[str, ...]:
    path = Path(__file__).resolve().parents[1] / "instrument_branch" / "docs" / "instrument-vocabulary.json"
    if not path.is_file():
        raise FileNotFoundError(f"instrument vocabulary missing: {path}")
    tags = json.loads(path.read_text(encoding="utf-8"))
    if len(tags) != N_INSTRUMENT_TAGS or len(set(tags)) != N_INSTRUMENT_TAGS:
        raise ValueError(f"instrument vocabulary must be {N_INSTRUMENT_TAGS} unique tags")
    if tags != sorted(tags):
        raise ValueError("instrument vocabulary must stay in official alphabetical order")
    return tuple(tags)


INSTRUMENT_TAGS: tuple[str, ...] = _load_instrument_tags()


def _load_timbre_features() -> tuple[str, ...]:
    path = (
        Path(__file__).resolve().parents[1]
        / "timbre_branch"
        / "src"
        / "timbre_branch"
        / "constants.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"timbre feature list missing: {path}")
    spec = importlib.util.spec_from_file_location("timbre_branch_constants", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load timbre constants from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cols = tuple(mod.FEATURE_COLUMNS)
    if len(cols) != N_TIMBRE_CONCEPTS or len(set(cols)) != N_TIMBRE_CONCEPTS:
        raise ValueError(f"timbre FEATURE_COLUMNS must be {N_TIMBRE_CONCEPTS} unique names")
    return cols


TIMBRE_FEATURES: tuple[str, ...] = _load_timbre_features()
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
    timbre: int = N_TIMBRE_CONCEPTS
    harmony: int = N_HARMONY_CHROMA

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
