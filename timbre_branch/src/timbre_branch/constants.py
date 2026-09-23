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

FEATURE_GROUPS = {
    "spectral_shape": tuple(range(0, 7)),
    "harmonic_noise": tuple(range(7, 9)),
    "mfcc_envelope": tuple(range(9, 35)),
}

assert len(FEATURE_COLUMNS) == 35
