"""Stable feature order shared by extraction, training, fusion, and explanations."""

FEATURE_COLUMNS = (
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_skewness_mean",
    "spectral_contrast_mean",
    "spectral_flatness_mean",
    "spectral_rolloff_mean",
    "spectral_flux_mean",
    "spectral_flux_std",
    "hnr_mean_db",
    "inharmonicity_mean",
)

FEATURE_GROUPS = {
    "spectral_shape": tuple(range(0, 8)),
    "spectral_dynamics": tuple(range(8, 10)),
    "harmonic_noise": tuple(range(10, 12)),
}

assert len(FEATURE_COLUMNS) == 12
