"""Frozen rhythm target contract shared by extraction, training, and fusion."""

RHYTHM_SCHEMA_VERSION = "acousticbrainz_rhythm_v1"

RHYTHM_FEATURES: tuple[str, ...] = (
    "bpm",
    "beats_count",
    "beats_loudness_mean",
    "bpm_histogram_first_peak_bpm",
    "bpm_histogram_first_peak_spread",
    "bpm_histogram_first_peak_weight",
    "onset_rate",
    "danceability",
    "beat_interval_mean",
    "beat_interval_std",
)

N_RHYTHM_FEATURES = len(RHYTHM_FEATURES)
RHYTHM_INPUT_DIM = 128
RHYTHM_EMBEDDING_DIM = 64
