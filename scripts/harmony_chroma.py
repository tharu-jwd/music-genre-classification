"""CPU-only candidate extractor for temporal harmony pseudo-supervision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "temporal_chroma_cqt_candidate_v1"
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEFAULT_HOP_LENGTH = 512
DEFAULT_BINS_PER_OCTAVE = 36
DEFAULT_N_OCTAVES = 7
DEFAULT_RELATIVE_SILENCE_THRESHOLD = 0.01
DEFAULT_ABSOLUTE_SILENCE_THRESHOLD = 1e-5


@dataclass(frozen=True)
class ChromaSequence:
    timestamps: np.ndarray
    chroma: np.ndarray
    tonal_concentration: np.ndarray
    valid: np.ndarray
    tuning_value: float
    tuning_unit: str
    extractor: str
    sample_rate: int
    hop_length: int

    def validate(self) -> None:
        frames = len(self.timestamps)
        if self.chroma.shape != (frames, 12):
            raise ValueError(f"expected chroma shape {(frames, 12)}, got {self.chroma.shape}")
        if self.tonal_concentration.shape != (frames,) or self.valid.shape != (frames,):
            raise ValueError("timestamps, confidence, and validity must share a frame axis")
        if not np.isfinite(self.chroma).all() or not np.isfinite(self.timestamps).all():
            raise ValueError("chroma output contains non-finite values")
        if (self.chroma < 0).any() or (self.chroma > 1).any():
            raise ValueError("normalized chroma must stay in [0, 1]")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        valid_sums = self.chroma[self.valid].sum(axis=1)
        if valid_sums.size and not np.allclose(valid_sums, 1.0, atol=1e-5):
            raise ValueError("valid chroma frames must sum to one")
        if np.any(self.chroma[~self.valid] != 0):
            raise ValueError("invalid frames must have zero chroma")


def _align_frames(values: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) >= frame_count:
        return values[:frame_count]
    return np.pad(values, (0, frame_count - len(values)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_cqt_chroma(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    bins_per_octave: int = DEFAULT_BINS_PER_OCTAVE,
    n_octaves: int = DEFAULT_N_OCTAVES,
    relative_silence_threshold: float = DEFAULT_RELATIVE_SILENCE_THRESHOLD,
    absolute_silence_threshold: float = DEFAULT_ABSOLUTE_SILENCE_THRESHOLD,
) -> ChromaSequence:
    """Extract an ordered CQT-chroma candidate without using genre labels."""
    import librosa

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"extract_cqt_chroma expects mono audio, got shape {audio.shape}")
    if sample_rate <= 0 or hop_length <= 0 or len(audio) == 0:
        raise ValueError("audio, sample_rate, and hop_length must be non-empty and positive")
    if not np.isfinite(audio).all():
        raise ValueError("waveform contains non-finite samples")

    peak = float(np.max(np.abs(audio)))
    if peak <= absolute_silence_threshold:
        frame_count = max(1, 1 + len(audio) // hop_length)
        sequence = ChromaSequence(
            timestamps=np.arange(frame_count, dtype=np.float32) * hop_length / sample_rate,
            chroma=np.zeros((frame_count, 12), dtype=np.float32),
            tonal_concentration=np.zeros(frame_count, dtype=np.float32),
            valid=np.zeros(frame_count, dtype=bool),
            tuning_value=0.0,
            tuning_unit="fraction_of_cqt_bin",
            extractor="librosa.feature.chroma_cqt",
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        sequence.validate()
        return sequence

    tuning = float(librosa.estimate_tuning(y=audio, sr=sample_rate, bins_per_octave=bins_per_octave))
    raw = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
        n_chroma=12,
        bins_per_octave=bins_per_octave,
        n_octaves=n_octaves,
        tuning=tuning,
        norm=None,
        threshold=0.0,
    )
    frame_count = raw.shape[1]
    rms = librosa.feature.rms(y=audio, hop_length=hop_length, center=True)[0]
    rms = _align_frames(rms, frame_count)
    threshold = max(absolute_silence_threshold, relative_silence_threshold * float(rms.max(initial=0.0)))
    valid = rms > threshold

    sums = raw.sum(axis=0)
    valid &= np.isfinite(sums) & (sums > np.finfo(np.float32).eps)
    normalized = np.zeros_like(raw, dtype=np.float32)
    normalized[:, valid] = raw[:, valid] / sums[valid]
    normalized = normalized.T
    concentration = normalized.max(axis=1).astype(np.float32)
    timestamps = librosa.frames_to_time(
        np.arange(frame_count), sr=sample_rate, hop_length=hop_length
    ).astype(np.float32)

    sequence = ChromaSequence(
        timestamps=timestamps,
        chroma=normalized,
        tonal_concentration=concentration,
        valid=valid.astype(bool),
        tuning_value=tuning,
        tuning_unit="fraction_of_cqt_bin",
        extractor="librosa.feature.chroma_cqt",
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    sequence.validate()
    return sequence


def extract_hpcp(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 4096,
    hop_length: int = DEFAULT_HOP_LENGTH,
    max_frequency: float = 5000.0,
    harmonic_margin: float = 1.0,
    relative_silence_threshold: float = DEFAULT_RELATIVE_SILENCE_THRESHOLD,
    absolute_silence_threshold: float = DEFAULT_ABSOLUTE_SILENCE_THRESHOLD,
) -> ChromaSequence:
    """Extract a 12-bin Essentia HPCP sequence in the same C-first order as CQT."""
    import essentia.standard as es
    import librosa

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"extract_hpcp expects mono audio, got shape {audio.shape}")
    if sample_rate <= 0 or hop_length <= 0 or frame_size <= 0 or len(audio) == 0:
        raise ValueError("audio and analysis sizes must be non-empty and positive")
    if not np.isfinite(audio).all():
        raise ValueError("waveform contains non-finite samples")
    if max_frequency >= sample_rate / 2:
        max_frequency = sample_rate / 2 - 1.0
    if max_frequency <= 700:
        raise ValueError("sample rate is too low for the configured HPCP frequency bands")

    if harmonic_margin <= 0:
        raise ValueError("harmonic_margin must be positive")
    analysis_audio = librosa.effects.harmonic(audio, margin=harmonic_margin).astype(np.float32)
    peak = float(np.max(np.abs(analysis_audio)))
    if peak <= absolute_silence_threshold:
        frame_count = max(1, 1 + len(audio) // hop_length)
        sequence = ChromaSequence(
            timestamps=(
                (np.arange(frame_count, dtype=np.float32) * hop_length + frame_size / 2)
                / sample_rate
            ).astype(np.float32),
            chroma=np.zeros((frame_count, 12), dtype=np.float32),
            tonal_concentration=np.zeros(frame_count, dtype=np.float32),
            valid=np.zeros(frame_count, dtype=bool),
            tuning_value=440.0,
            tuning_unit="reference_frequency_hz",
            extractor="essentia.standard.HPCP+librosa.effects.harmonic",
            sample_rate=sample_rate,
            hop_length=hop_length,
        )
        sequence.validate()
        return sequence

    tuning_values = np.asarray(
        es.TuningFrequencyExtractor(frameSize=frame_size, hopSize=hop_length)(analysis_audio),
        dtype=np.float32,
    )
    plausible = tuning_values[np.isfinite(tuning_values) & (tuning_values >= 400) & (tuning_values <= 480)]
    reference_frequency = float(np.median(plausible)) if plausible.size else 440.0

    window = es.Windowing(type="hann", normalized=False, zeroPadding=0)
    spectrum = es.Spectrum(size=frame_size)
    peaks = es.SpectralPeaks(
        sampleRate=sample_rate,
        minFrequency=40.0,
        maxFrequency=max_frequency,
        maxPeaks=60,
        magnitudeThreshold=1e-5,
        orderBy="magnitude",
    )
    hpcp = es.HPCP(
        size=12,
        sampleRate=sample_rate,
        minFrequency=40.0,
        maxFrequency=max_frequency,
        referenceFrequency=reference_frequency,
        normalized="none",
        nonLinear=False,
        harmonics=0,
        windowSize=1.0,
        weightType="squaredCosine",
    )

    raw_frames = []
    rms = []
    for frame in es.FrameGenerator(
        analysis_audio, frameSize=frame_size, hopSize=hop_length,
        startFromZero=True, lastFrameToEndOfFile=True,
    ):
        frequencies, magnitudes = peaks(spectrum(window(frame)))
        raw_frames.append(hpcp(frequencies, magnitudes))
        rms.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))

    raw = np.asarray(raw_frames, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 12:
        raise RuntimeError(f"Essentia returned an unexpected HPCP shape: {raw.shape}")
    raw = np.roll(raw, -3, axis=1)  # Essentia is A-first; project artifacts are C-first.
    rms = np.asarray(rms, dtype=np.float32)
    threshold = max(absolute_silence_threshold, relative_silence_threshold * float(rms.max(initial=0.0)))
    sums = raw.sum(axis=1)
    valid = (rms > threshold) & np.isfinite(sums) & (sums > np.finfo(np.float32).eps)
    normalized = np.zeros_like(raw, dtype=np.float32)
    normalized[valid] = raw[valid] / sums[valid, None]
    concentration = normalized.max(axis=1).astype(np.float32)
    timestamps = (
        (np.arange(len(raw), dtype=np.float32) * hop_length + frame_size / 2) / sample_rate
    ).astype(np.float32)

    sequence = ChromaSequence(
        timestamps=timestamps,
        chroma=normalized,
        tonal_concentration=concentration,
        valid=valid.astype(bool),
        tuning_value=reference_frequency,
        tuning_unit="reference_frequency_hz",
        extractor="essentia.standard.HPCP+librosa.effects.harmonic",
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    sequence.validate()
    return sequence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract one CPU CQT-chroma candidate artifact; this is not full-dataset generation."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--force", action="store_true", help="replace an existing candidate artifact")
    args = parser.parse_args()

    import librosa

    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    if not args.force and (args.output.exists() or metadata_path.exists()):
        parser.error("output already exists; choose a new path or pass --force")

    waveform, sample_rate = librosa.load(args.audio, sr=args.sample_rate, mono=True)
    sequence = extract_cqt_chroma(waveform, sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        timestamps=sequence.timestamps,
        chroma=sequence.chroma,
        tonal_concentration=sequence.tonal_concentration,
        valid=sequence.valid,
        pitch_classes=np.asarray(PITCH_CLASSES, dtype=str),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "extractor": sequence.extractor,
        "extractor_library_version": importlib.metadata.version("librosa"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audio": str(args.audio),
        "source_audio_sha256": _sha256_file(args.audio),
        "source_region_seconds": [0.0, len(waveform) / sample_rate],
        "sample_rate": sequence.sample_rate,
        "hop_length": sequence.hop_length,
        "bins_per_octave": DEFAULT_BINS_PER_OCTAVE,
        "n_octaves": DEFAULT_N_OCTAVES,
        "relative_silence_threshold": DEFAULT_RELATIVE_SILENCE_THRESHOLD,
        "absolute_silence_threshold": DEFAULT_ABSOLUTE_SILENCE_THRESHOLD,
        "tuning_value": sequence.tuning_value,
        "tuning_unit": sequence.tuning_unit,
        "frames": len(sequence.timestamps),
        "valid_frames": int(sequence.valid.sum()),
        "tonal_concentration_semantics": "maximum L1-normalized chroma bin; diagnostic, not calibrated confidence",
        "candidate_only": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
