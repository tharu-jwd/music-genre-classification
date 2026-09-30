"""Stable feature order shared by extraction, training, fusion, and explanations."""

FEATURE_COLUMNS = (
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_contrast_mean",
    "spectral_flatness_mean",
    "spectral_rolloff_mean",
    "hnr_mean_db",
    "inharmonicity_mean",
    *(f"mfcc_{index:02d}_mean" for index in range(1, 14)),
    *(f"mfcc_{index:02d}_std" for index in range(1, 14)),
)

ARCHITECTURE_VERSION = "timbre_branch_v2"
PREPROCESSING_VERSION = "timbre_v2_log_flatness_zscore_v1"
FLATNESS_FEATURE = "spectral_flatness_mean"
FLATNESS_INDEX = FEATURE_COLUMNS.index(FLATNESS_FEATURE)
DEFAULT_FLATNESS_EPSILON = 1e-12

V2_MAX_EPOCHS = 30

assert len(FEATURE_COLUMNS) == 35
