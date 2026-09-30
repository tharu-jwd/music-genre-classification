"""Frozen harmony descriptor contract shared by extraction, training, and fusion.

The 45 names are the exact column order of ``data/harmony_df.csv``; see
``harmony_branch/docs/feature-contract.md`` for their extraction definitions.
This module has no dependencies so ``concept_fusion.contract`` can load it by path.
"""

HARMONY_SCHEMA_VERSION = "harmony_descriptors_45_v1"

PITCH_CLASSES: tuple[str, ...] = (
    "c", "csharp", "d", "dsharp", "e", "f", "fsharp", "g", "gsharp", "a", "asharp", "b",
)

CHROMA_MEAN_FEATURES = tuple(f"chroma_{pitch}_mean" for pitch in PITCH_CLASSES)
CHROMA_STD_FEATURES = tuple(f"chroma_{pitch}_std" for pitch in PITCH_CLASSES)
TONNETZ_MEAN_FEATURES = tuple(f"tonnetz_{index:02d}_mean" for index in range(1, 7))
TONNETZ_STD_FEATURES = tuple(f"tonnetz_{index:02d}_std" for index in range(1, 7))
TONAL_DYNAMICS_FEATURES: tuple[str, ...] = (
    "tonal_concentration_mean", "tonal_concentration_std",
    "chroma_entropy_mean", "chroma_entropy_std",
    "chroma_flux_mean", "chroma_flux_std",
    "tonnetz_movement_mean", "tonnetz_movement_std",
    "valid_tonal_ratio",
)

HARMONY_DESCRIPTORS: tuple[str, ...] = (
    *CHROMA_MEAN_FEATURES,
    *CHROMA_STD_FEATURES,
    *TONNETZ_MEAN_FEATURES,
    *TONNETZ_STD_FEATURES,
    *TONAL_DYNAMICS_FEATURES,
)
N_HARMONY_DESCRIPTORS = len(HARMONY_DESCRIPTORS)

# The hand-picked 12-target subset trained by Harmony v3; kept for comparisons.
HARMONY_V3_DESCRIPTORS: tuple[str, ...] = (
    "tonal_concentration_mean", "tonal_concentration_std",
    "chroma_entropy_mean", "chroma_entropy_std",
    "chroma_flux_mean", "chroma_flux_std",
    "tonnetz_movement_mean", "tonnetz_movement_std",
    "valid_tonal_ratio", "tonnetz_01_mean", "tonnetz_02_mean", "tonnetz_03_mean",
)

DESCRIPTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "chroma_mean": CHROMA_MEAN_FEATURES,
    "chroma_std": CHROMA_STD_FEATURES,
    "tonnetz_mean": TONNETZ_MEAN_FEATURES,
    "tonnetz_std": TONNETZ_STD_FEATURES,
    "tonal_dynamics": TONAL_DYNAMICS_FEATURES,
}

# Per-descriptor variance-stabilizing transform applied before the train-only
# z-score. "log" is log(x + LOG_EPSILON) for strictly positive right-skewed
# values; "log1m" is -log(1 - x + LOG1M_EPSILON) for the ratio that piles up at
# 1.0; "identity" is used where log does not reduce skew on the 7,324-track table.
LOG_EPSILON = 1e-4
LOG1M_EPSILON = 1e-3
_IDENTITY = {*TONNETZ_MEAN_FEATURES, "chroma_entropy_mean", "tonal_concentration_std"}
DESCRIPTOR_TRANSFORMS: dict[str, str] = {
    name: (
        "identity" if name in _IDENTITY
        else "log1m" if name == "valid_tonal_ratio"
        else "log"
    )
    for name in HARMONY_DESCRIPTORS
}

assert N_HARMONY_DESCRIPTORS == 45
assert len(set(HARMONY_DESCRIPTORS)) == 45
assert set(HARMONY_V3_DESCRIPTORS) <= set(HARMONY_DESCRIPTORS)
