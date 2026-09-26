"""Shared-architecture contract v0.3.

Reject violating tensors. Do not silently reshape, reorder, impute, or reinterpret.

Primary fusion uses predicted concepts from all four branches. The former rhythm
embedding / harmony embedding path remains available only as ``embedding_fusion``.

Instrument v2: 40 official probabilities + logits, no fusion token. Fusion owns Linear(40, 64).
Rhythm v2: 10 standardized predictions are primary; the 64D embedding is retained
for the embedding-fusion ablation. Fusion owns Linear(10, 64) in the primary mode.
Timbre v2 (merged from `timbre_branch`): 35 standardized concepts, no fusion token.
Fusion owns Linear(35, 64).
Harmony v2: masked-pooled probabilities from temporal 12-bin chroma logits are
primary. Its 32D song embedding is retained for the embedding-fusion ablation.
Chords remain an optional temporal auxiliary target and never enter primary fusion.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FUSION_CONTRACT_VERSION = "predicted_concept_fusion_v1"
FusionInputMode = Literal["predicted_concepts", "embedding_fusion"]
PRIMARY_FUSION_INPUT_MODE: FusionInputMode = "predicted_concepts"
EMBEDDING_FUSION_INPUT_MODE: FusionInputMode = "embedding_fusion"
FUSION_INPUT_MODES: tuple[FusionInputMode, ...] = (
    PRIMARY_FUSION_INPUT_MODE,
    EMBEDDING_FUSION_INPUT_MODE,
)

CONCEPT_ORDER: tuple[str, ...] = ("instrument", "rhythm", "timbre", "harmony")
N_CONCEPTS = 4
TOKEN_DIM = 64
FUSED_DIM = 128
# Official MTG-Jamendo split-0 has 87 genres. The current joint-training table
# only labels these six; GenreHead is sized to this scoped set until an 87-column
# table exists. Do not report scoped-6 scores as official 87-tag results.
OFFICIAL_N_GENRE_TAGS = 87
N_GENRE_TAGS = 6
GENRE_TAGS: tuple[str, ...] = (
    "classical", "electronic", "folk", "hiphop", "jazz", "rock"
)
GENRE_SCOPE = "scoped_6_from_genres_df"
N_INSTRUMENT_TAGS = 40
N_RHYTHM_CONCEPTS = 10
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
    # Strip the 'instrument---' prefix used by MTG-Jamendo for display; store bare names.
    tags = [t.replace("instrument---", "") for t in tags]
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


def _load_rhythm_features() -> tuple[str, ...]:
    path = (
        Path(__file__).resolve().parents[1]
        / "rhythm_branch"
        / "src"
        / "rhythm_branch"
        / "constants.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"rhythm feature list missing: {path}")
    spec = importlib.util.spec_from_file_location("rhythm_branch_constants", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load rhythm constants from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cols = tuple(mod.RHYTHM_FEATURES)
    if len(cols) != N_RHYTHM_CONCEPTS or len(set(cols)) != N_RHYTHM_CONCEPTS:
        raise ValueError(f"rhythm RHYTHM_FEATURES must be {N_RHYTHM_CONCEPTS} unique names")
    return cols


RHYTHM_FEATURES: tuple[str, ...] = _load_rhythm_features()
CONCEPT_DROPOUT_P = 0.15
SUPPORT_FLOOR = 10
GLOBAL_THRESHOLD_FALLBACK = 0.5

PRIMARY_METRIC = "macro_ap"
FUSION_PRIMARY = "gated"
FUSION_BASELINE = "concat"
FUSION_SECONDARY = "attention"

# Seeds: three on the paper comparison; one elsewhere (Colab budget).
MULTI_SEED_EXPERIMENTS = frozenset({"B1", "F-Concat", "F-Gated", "F-Embedding"})
N_SEEDS_FINAL = 3
N_SEEDS_OTHER = 1
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class ConceptCounts:
    instrument: int = N_INSTRUMENT_TAGS
    rhythm: int = N_RHYTHM_CONCEPTS
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
    ("F-Embedding", "embedding_fusion", "Previous rhythm/harmony embedding route"),
    ("F-Attn", "attention", "All concepts, self-attention (optional)"),
    ("F-Hidden", "hidden", "Hidden branch embeddings (faithfulness tax)"),
    ("F-Shortcut", "shortcut", "Fusion plus direct audio path (bottleneck tax)"),
)
