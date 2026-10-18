"""The timbre extraction notebooks emit the current named contract."""
import ast
import json
import math
from pathlib import Path
import warnings
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from concept_fusion.contract import TIMBRE_FEATURES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "timbre_branch/notebooks").glob("extract_timbre_*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_emits_current_12_named_features_on_short_audio(path, tmp_path):
    # Stub existing DSP calls: this test checks the new schema and aggregation,
    # without triggering librosa/numba compilation or re-validating its algorithms.
    magnitude = np.tile(np.linspace(0.1, 1, 1025)[:, None], (1, 4))
    librosa = SimpleNamespace(
        get_duration=lambda **kwargs: 0.15,
        load=lambda *args, **kwargs: (np.ones(3307, dtype=np.float32), 22050),
        stft=lambda *args, **kwargs: magnitude,
        feature=SimpleNamespace(
            spectral_centroid=lambda **kwargs: np.full((1, 4), 2000),
            spectral_bandwidth=lambda **kwargs: np.full((1, 4), 1000),
            spectral_contrast=lambda **kwargs: np.full((7, 4), 20),
            spectral_flatness=lambda **kwargs: np.full((1, 4), 0.1),
            spectral_rolloff=lambda **kwargs: np.full((1, 4), 4000),
        ),
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    config = next(source for source in sources if "FEATURE_COLUMNS = " in source)
    tree = ast.parse(config)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "FEATURE_COLUMNS" for target in node.targets))
    names = ast.literal_eval(assignment.value)
    assert tuple(names) == TIMBRE_FEATURES
    functions = next(source for source in sources if "def extract_timbre_features(" in source)
    namespace = dict(np=np, librosa=librosa, Path=Path, warnings=warnings, math=math,
                     SAMPLE_RATE=22050, N_FFT=2048, HOP_LENGTH=512, ROLLOFF_PERCENT=0.85,
                     ANALYSIS_DURATION_SECONDS=240, HARMONIC_FRAME_LENGTH=4096,
                     WINDOW_SECONDS=15, NUM_WINDOWS=3,
                     FEATURE_COLUMNS=names)
    exec(compile(functions, str(path), "exec"), namespace)
    # Short enough to exercise the missing-harmonic-target path without pitch estimation.
    waveform = np.sin(2 * np.pi * 440 * np.arange(3307) / 22050) * 0.5
    audio = tmp_path / "tone.wav"
    with wave.open(str(audio), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes((waveform * 32767).astype("<i2").tobytes())
    features, quality = namespace["extract_timbre_features"](audio)
    assert tuple(features) == TIMBRE_FEATURES
    assert np.isfinite([features[name] for name in TIMBRE_FEATURES[:10]]).all()
    assert np.isnan(features["hnr_mean_db"])
    assert np.isnan(features["inharmonicity_mean"])
    assert features["spectral_flux_mean"] == 0
    assert features["spectral_flux_std"] == 0
    assert quality["hnr_valid_frames"] == 0
