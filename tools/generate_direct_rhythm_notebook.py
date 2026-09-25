"""Generate the Colab notebook for direct first-four-minute rhythm extraction."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rhythm_branch" / "notebooks" / "extract_rhythm_features_first4min.ipynb"


def _lines(value: str) -> list[str]:
    value = dedent(value).strip("\n") + "\n"
    return value.splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(value),
    }


cells = [
    markdown(
        """
        # Direct rhythm-feature extraction — first four minutes

        This notebook creates one interpretable rhythm-target row for every track in
        `split_csv.csv`. It uses the raw MP3 files and Essentia directly—**no
        AcousticBrainz files are downloaded or read**.

        For each track, the analyzed interval is `[0, min(track duration, 240 s)]`.
        The ten outputs are BPM, excerpt beat count, mean beat loudness, three first
        BPM-histogram-peak descriptors, onset rate, DFA danceability, and the mean and
        standard deviation of consecutive beat intervals.

        Run the cells in order. First edit only the paths in **Configuration**. The
        extraction is sequential, checkpointed, and safely resumable after a Colab
        disconnect. A synthetic smoke test and a real-file smoke test run before the
        full dataset loop.
        """
    ),
    markdown("## 1. Install dependencies and mount Google Drive"),
    code(
        """
        # The TensorFlow build provides maintained Linux wheels and includes all
        # classical Essentia algorithms used below; no neural model is used.
        !pip -q install "essentia-tensorflow>=2.1b6.dev1389" "pandas>=2.0" "tqdm>=4.66"
        """
    ),
    code(
        """
        from google.colab import drive
        drive.mount('/content/drive')
        """
    ),
    markdown("## 2. Configuration"),
    code(
        """
        from pathlib import Path
        import gc
        import json
        import math
        import os
        import re
        import shutil
        import time
        import traceback

        import essentia
        import essentia.standard as es
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm

        # Change these three paths only if your Drive layout is different.
        PROJECT_ROOT = Path('/content/drive/MyDrive/MTG_Jamendo_7324_full_quality')
        CSV_PATH = PROJECT_ROOT / 'split_csv.csv'
        AUDIO_ROOT = PROJECT_ROOT / 'audio'
        OUTPUT_ROOT = PROJECT_ROOT / 'rhythm_features'

        CHECKPOINT_CSV = OUTPUT_ROOT / 'rhythm_features_checkpoint.csv'
        FINAL_CSV = OUTPUT_ROOT / 'rhythm_features_raw.csv'
        ERROR_CSV = OUTPUT_ROOT / 'rhythm_feature_errors.csv'
        AUDIT_CSV = OUTPUT_ROOT / 'rhythm_audio_path_audit.csv'
        CONFIG_JSON = OUTPUT_ROOT / 'rhythm_extraction_config.json'

        SAMPLE_RATE = 44_100
        MAX_DURATION_SECONDS = 240.0
        RHYTHM_METHOD = 'multifeature'  # slower and more accurate than 'degara'
        CHECKPOINT_EVERY = 10
        RETRY_FAILED_TRACKS = True
        LIMIT_TRACKS = None  # Set to 10 for a trial; None processes all remaining tracks.

        SOURCE = 'essentia_direct_first4min'
        SCHEMA_VERSION = 'essentia_rhythm_first4min_v1'

        FEATURE_COLUMNS = [
            'bpm',
            'beats_count',
            'beats_loudness_mean',
            'bpm_histogram_first_peak_bpm',
            'bpm_histogram_first_peak_spread',
            'bpm_histogram_first_peak_weight',
            'onset_rate',
            'danceability',
            'beat_interval_mean',
            'beat_interval_std',
        ]

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print('Essentia:', getattr(essentia, '__version__', 'unknown'))
        print('Manifest:', CSV_PATH)
        print('Audio root:', AUDIO_ROOT)
        print('Output root:', OUTPUT_ROOT)
        """
    ),
    markdown("## 3. Load the selected-track manifest and resolve MP3 paths"),
    code(
        """
        assert CSV_PATH.is_file(), f'Missing manifest: {CSV_PATH}'
        assert AUDIO_ROOT.is_dir(), f'Missing audio directory: {AUDIO_ROOT}'

        manifest = pd.read_csv(CSV_PATH)

        def find_column(frame, candidates, required=True):
            lookup = {str(column).strip().lower(): column for column in frame.columns}
            for candidate in candidates:
                if candidate.lower() in lookup:
                    return lookup[candidate.lower()]
            if required:
                raise KeyError(f'None of {candidates} found. Columns: {list(frame.columns)}')
            return None

        id_col = find_column(manifest, ['TRACK_ID', 'track_id', 'song_id'])
        path_col = find_column(manifest, ['PATH', 'path', 'rel_path', 'audio_path'], required=False)
        split_col = find_column(manifest, ['split', 'dataset_split', 'subset'], required=False)

        def normalize_song_id(value):
            match = re.fullmatch(r'(?:track_)?(\\d+)', str(value).strip(), flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f'Invalid track ID: {value!r}')
            return f'{int(match.group(1)):07d}'

        def resolve_audio_path(row):
            song_id = row['song_id']
            candidates = []
            if path_col is not None and pd.notna(row[path_col]):
                relative = str(row[path_col]).strip().replace('\\\\', '/')
                candidates.append(AUDIO_ROOT / Path(relative))
            candidates.extend([
                AUDIO_ROOT / song_id[-2:] / f'{int(song_id)}.mp3',
                AUDIO_ROOT / song_id[-2:] / f'{song_id}.mp3',
                AUDIO_ROOT / f'{int(song_id)}.mp3',
                AUDIO_ROOT / f'{song_id}.mp3',
            ])
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            return candidates[0]

        selected = manifest.copy()
        selected['song_id'] = selected[id_col].map(normalize_song_id)
        if selected['song_id'].duplicated().any():
            duplicate = selected.loc[selected['song_id'].duplicated(False), 'song_id'].iloc[0]
            raise ValueError(f'Duplicate selected track ID: {duplicate}')

        selected['audio_path'] = selected.apply(resolve_audio_path, axis=1).map(str)
        selected['audio_exists'] = selected['audio_path'].map(lambda value: Path(value).is_file())
        selected['size_bytes'] = selected['audio_path'].map(
            lambda value: Path(value).stat().st_size if Path(value).is_file() else np.nan
        )
        selected['split'] = (
            selected[split_col].astype(str).str.strip() if split_col is not None else ''
        )

        audit_columns = ['song_id', 'audio_path', 'audio_exists', 'size_bytes']
        selected[audit_columns].to_csv(AUDIT_CSV, index=False)

        print('Selected tracks:', len(selected))
        print('Unique IDs:', selected['song_id'].nunique())
        print('Existing audio:', int(selected['audio_exists'].sum()))
        print('Missing audio:', int((~selected['audio_exists']).sum()))
        if len(selected) != 7_324:
            print('WARNING: expected 7,324 selected rows, found', len(selected))
        if not selected['audio_exists'].all():
            display(selected.loc[~selected['audio_exists'], audit_columns].head(20))
            raise FileNotFoundError('Some selected MP3 files are missing; inspect the audit CSV.')
        """
    ),
    markdown("## 4. Feature-extraction implementation"),
    code(
        """
        def finite_or_nan(value):
            value = float(value)
            return value if np.isfinite(value) else np.nan

        def extract_rhythm_from_signal(audio, sample_rate=SAMPLE_RATE):
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or audio.size == 0:
                raise ValueError('Expected a non-empty mono waveform')

            duration = audio.size / float(sample_rate)
            bpm, beats, confidence, bpm_estimates, bpm_intervals = es.RhythmExtractor2013(
                method=RHYTHM_METHOD
            )(audio)

            beats = np.asarray(beats, dtype=np.float64)
            bpm_intervals = np.asarray(bpm_intervals, dtype=np.float64)
            beat_differences = np.diff(beats)

            first_peak_bpm = first_peak_spread = first_peak_weight = np.nan
            if bpm_intervals.size:
                (
                    first_peak_bpm,
                    first_peak_weight,
                    first_peak_spread,
                    _second_peak_bpm,
                    _second_peak_weight,
                    _second_peak_spread,
                    _histogram,
                ) = es.BpmHistogramDescriptors()(bpm_intervals.astype(np.float32))

            beat_loudness_mean = np.nan
            if beats.size:
                beat_loudness, _band_ratios = es.BeatsLoudness(
                    sampleRate=sample_rate,
                    beats=beats.astype(np.float32),
                )(audio)
                beat_loudness = np.asarray(beat_loudness, dtype=np.float64)
                if beat_loudness.size:
                    beat_loudness_mean = float(np.mean(beat_loudness))

            onset_times, onset_rate = es.OnsetRate()(audio)
            danceability, _dfa = es.Danceability(sampleRate=sample_rate)(audio)

            features = {
                'bpm': finite_or_nan(bpm),
                'beats_count': float(beats.size),
                'beats_loudness_mean': finite_or_nan(beat_loudness_mean),
                'bpm_histogram_first_peak_bpm': finite_or_nan(first_peak_bpm),
                'bpm_histogram_first_peak_spread': finite_or_nan(first_peak_spread),
                'bpm_histogram_first_peak_weight': finite_or_nan(first_peak_weight),
                'onset_rate': finite_or_nan(onset_rate),
                'danceability': finite_or_nan(danceability),
                'beat_interval_mean': (
                    finite_or_nan(np.mean(beat_differences)) if beat_differences.size else np.nan
                ),
                'beat_interval_std': (
                    finite_or_nan(np.std(beat_differences, ddof=0))
                    if beat_differences.size else np.nan
                ),
            }
            diagnostics = {
                'analysis_duration_sec': duration,
                'beat_tracker_confidence': finite_or_nan(confidence),
                'detected_onsets_count': int(len(onset_times)),
                'detected_beats_count': int(beats.size),
            }
            return features, diagnostics

        def extract_rhythm_file(audio_path, max_duration_seconds=MAX_DURATION_SECONDS):
            # MonoLoader decodes the file; slicing guarantees an exact first-240-s cap.
            audio = es.MonoLoader(filename=str(audio_path), sampleRate=SAMPLE_RATE)()
            decoded_duration = len(audio) / float(SAMPLE_RATE)
            max_samples = int(round(max_duration_seconds * SAMPLE_RATE))
            audio = np.ascontiguousarray(audio[:max_samples], dtype=np.float32)
            features, diagnostics = extract_rhythm_from_signal(audio)
            diagnostics['decoded_duration_sec'] = decoded_duration
            diagnostics['input_scope'] = 'first_4min' if decoded_duration > 240.0 else 'entire_track'
            return features, diagnostics
        """
    ),
    markdown("## 5. Smoke tests (run before full extraction)"),
    code(
        """
        # Synthetic pulse train: validates every Essentia call and the ten-field contract.
        smoke_seconds = 30.0
        expected_bpm = 120.0
        synthetic = np.zeros(int(SAMPLE_RATE * smoke_seconds), dtype=np.float32)
        pulse = np.hanning(int(0.03 * SAMPLE_RATE)).astype(np.float32)
        for beat_time in np.arange(0.5, smoke_seconds - 0.5, 60.0 / expected_bpm):
            start = int(beat_time * SAMPLE_RATE)
            synthetic[start:start + len(pulse)] += 0.8 * pulse

        smoke_features, smoke_diagnostics = extract_rhythm_from_signal(synthetic)
        assert list(smoke_features) == FEATURE_COLUMNS
        assert len(smoke_features) == 10
        assert np.isfinite(smoke_features['bpm'])
        assert smoke_features['beats_count'] >= 2
        assert np.isfinite(smoke_features['onset_rate'])
        assert np.isfinite(smoke_features['danceability'])
        print('Synthetic smoke test passed')
        print(json.dumps({**smoke_features, **smoke_diagnostics}, indent=2))
        """
    ),
    code(
        """
        # Real-file smoke test: validates Drive path resolution and MP3 decoding.
        smoke_row = selected.iloc[0]
        started = time.time()
        real_features, real_diagnostics = extract_rhythm_file(
            smoke_row['audio_path'], max_duration_seconds=30.0
        )
        assert list(real_features) == FEATURE_COLUMNS
        assert real_diagnostics['analysis_duration_sec'] <= 30.0 + (1.0 / SAMPLE_RATE)
        assert sum(np.isfinite(list(real_features.values()))) >= 7
        print('Real-file smoke test passed:', smoke_row['song_id'])
        print('Elapsed seconds:', round(time.time() - started, 2))
        print(json.dumps({**real_features, **real_diagnostics}, indent=2))
        """
    ),
    markdown("## 6. Resumable full-dataset extraction"),
    code(
        """
        METADATA_COLUMNS = [
            'song_id', 'split', 'source', 'schema_version', 'status', 'audio_path',
            'analysis_start_sec', 'analysis_end_sec', 'analysis_duration_sec',
            'decoded_duration_sec', 'input_scope', 'sample_rate', 'rhythm_method',
            'beat_tracker_confidence', 'detected_onsets_count', 'detected_beats_count',
            'elapsed_sec', 'error_type', 'error_message',
        ]
        ALL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

        if CHECKPOINT_CSV.is_file():
            checkpoint = pd.read_csv(CHECKPOINT_CSV, dtype={'song_id': str}, keep_default_na=True)
            checkpoint['song_id'] = checkpoint['song_id'].map(normalize_song_id)
            records_by_id = {
                row['song_id']: row.to_dict() for _, row in checkpoint.iterrows()
            }
            print('Loaded checkpoint rows:', len(records_by_id))
        else:
            records_by_id = {}

        def save_checkpoint(records):
            frame = pd.DataFrame(records.values())
            for column in ALL_COLUMNS:
                if column not in frame:
                    frame[column] = np.nan
            order = {song_id: index for index, song_id in enumerate(selected['song_id'])}
            frame['_order'] = frame['song_id'].map(order)
            frame = frame.sort_values('_order').drop(columns='_order').loc[:, ALL_COLUMNS]
            temporary = CHECKPOINT_CSV.with_suffix('.tmp.csv')
            frame.to_csv(temporary, index=False)
            os.replace(temporary, CHECKPOINT_CSV)
            return frame

        pending = []
        for _, row in selected.iterrows():
            previous = records_by_id.get(row['song_id'])
            if previous is None:
                pending.append(row)
            elif RETRY_FAILED_TRACKS and previous.get('status') != 'ok':
                pending.append(row)

        if LIMIT_TRACKS is not None:
            pending = pending[:int(LIMIT_TRACKS)]

        print('Already checkpointed:', len(records_by_id))
        print('Tracks to process now:', len(pending))

        for index, row in enumerate(tqdm(pending, desc='Extract first-4-min rhythm features'), start=1):
            song_id = row['song_id']
            started = time.time()
            base = {
                'song_id': song_id,
                'split': row['split'],
                'source': SOURCE,
                'schema_version': SCHEMA_VERSION,
                'audio_path': row['audio_path'],
                'analysis_start_sec': 0.0,
                'sample_rate': SAMPLE_RATE,
                'rhythm_method': RHYTHM_METHOD,
            }
            try:
                features, diagnostics = extract_rhythm_file(row['audio_path'])
                missing = [name for name, value in features.items() if not np.isfinite(value)]
                status = 'ok' if not missing else 'incomplete_features'
                record = {
                    **base,
                    **diagnostics,
                    **features,
                    'status': status,
                    'analysis_end_sec': diagnostics['analysis_duration_sec'],
                    'error_type': '' if not missing else 'NonFiniteFeatures',
                    'error_message': '' if not missing else ','.join(missing),
                }
            except Exception as error:
                record = {
                    **base,
                    'status': 'error',
                    'error_type': type(error).__name__,
                    'error_message': str(error)[:1000],
                }
            record['elapsed_sec'] = time.time() - started
            records_by_id[song_id] = record

            if index % CHECKPOINT_EVERY == 0:
                save_checkpoint(records_by_id)
                gc.collect()

        checkpoint = save_checkpoint(records_by_id)
        print(checkpoint['status'].value_counts(dropna=False))
        """
    ),
    markdown("## 7. Finalize, audit, and save outputs"),
    code(
        """
        final = pd.read_csv(CHECKPOINT_CSV, dtype={'song_id': str}, keep_default_na=True)
        final['song_id'] = final['song_id'].map(normalize_song_id)

        selected_ids = set(selected['song_id'])
        output_ids = set(final['song_id'])
        missing_ids = sorted(selected_ids - output_ids)
        extra_ids = sorted(output_ids - selected_ids)

        if final['song_id'].duplicated().any():
            raise AssertionError('Checkpoint contains duplicate song IDs')
        if missing_ids:
            raise AssertionError(f'Extraction is not complete; {len(missing_ids)} selected tracks are missing')
        if extra_ids:
            raise AssertionError(f'Checkpoint contains {len(extra_ids)} unselected tracks')

        feature_values = final[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')
        finite_mask = np.isfinite(feature_values.to_numpy(dtype=np.float64))
        complete_feature_rows = finite_mask.all(axis=1)
        status_ok = final['status'].eq('ok').to_numpy()
        if not np.array_equal(complete_feature_rows, status_ok):
            raise AssertionError('Status values do not agree with feature finiteness')

        errors = final.loc[~final['status'].eq('ok')].copy()
        final.to_csv(FINAL_CSV, index=False)
        errors.to_csv(ERROR_CSV, index=False)

        configuration = {
            'schema_version': SCHEMA_VERSION,
            'source': SOURCE,
            'selected_rows': int(len(selected)),
            'output_rows': int(len(final)),
            'ok_rows': int(final['status'].eq('ok').sum()),
            'non_ok_rows': int((~final['status'].eq('ok')).sum()),
            'sample_rate': SAMPLE_RATE,
            'maximum_duration_seconds': MAX_DURATION_SECONDS,
            'analysis_interval': '[0, min(decoded duration, 240 seconds)]',
            'channel_mode': 'mono',
            'rhythm_method': RHYTHM_METHOD,
            'feature_columns': FEATURE_COLUMNS,
            'beats_count_definition': 'number of beats detected in the analyzed excerpt',
            'beat_interval_definition': 'np.diff(detected beat positions), population std (ddof=0)',
            'split_column_found': split_col is not None,
            'essentia_version': getattr(essentia, '__version__', 'unknown'),
        }
        CONFIG_JSON.write_text(json.dumps(configuration, indent=2), encoding='utf-8')

        print('Final rows:', len(final))
        print('Unique tracks:', final['song_id'].nunique())
        print('OK rows:', int(final['status'].eq('ok').sum()))
        print('Non-OK rows:', len(errors))
        print('Feature count:', len(FEATURE_COLUMNS))
        print('NaN feature cells:', int(feature_values.isna().sum().sum()))
        print('Infinite feature cells:', int(np.isinf(feature_values.to_numpy()).sum()))
        print('Final CSV:', FINAL_CSV)
        print('Error CSV:', ERROR_CSV)
        print('Config JSON:', CONFIG_JSON)
        display(final.head())
        if len(errors):
            display(errors[['song_id', 'status', 'error_type', 'error_message']].head(20))
        """
    ),
    markdown(
        """
        ## Interpretation and training note

        - `beats_count` is the count inside the analyzed excerpt, not necessarily the
          complete original track.
        - `danceability` is Essentia's continuous DFA descriptor, not a Spotify score
          or a neural classifier probability.
        - Keep the raw values. Fit normalization statistics using training rows only.
        - The current repository loader was originally restricted to
          `source=acousticbrainz`. Before training, update that validation and schema
          to accept `essentia_direct_first4min`; never mislabel these values as
          AcousticBrainz.
        - If `split_csv.csv` has no actual train/validation/test column, this notebook
          leaves `split` empty. Join the official split later by `song_id`; do not
          invent or randomly regenerate it during feature extraction.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "CPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
