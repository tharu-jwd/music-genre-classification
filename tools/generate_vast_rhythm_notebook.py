"""Generate a four-worker Vast.ai rhythm extraction notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "rhythm_branch" / "notebooks" / "extract_rhythm_features_first4min.ipynb"
OUTPUT = ROOT / "rhythm_branch" / "notebooks" / "extract_rhythm_features_first4min_vastai.ipynb"


def lines(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def md(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


if not SOURCE.is_file():
    raise FileNotFoundError(f"Generate the Colab notebook first: {SOURCE}")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]

cells[0] = md(
    """
    # Vast.ai direct rhythm extraction — first four minutes, four CPU workers

    This Vast.ai edition copies the selected 7,324-track dataset from Google Drive
    to the instance's local NVMe storage using `rclone`, extracts ten rhythm concepts
    directly from the first `min(duration, 240 s)` of each MP3, and periodically
    backs checkpoints up to Drive. It never uses AcousticBrainz.

    The extraction uses **four independent CPU workers**. The rented GPU is not used
    by these classical Essentia algorithms. Run the cells in order and do not start
    the full run until both smoke tests pass.
    """
)
cells[1] = md("## 1. Install system and Python dependencies")
cells[2] = code(
    """
    !apt-get -qq update
    !apt-get -qq install -y rclone
    # This maintained Linux wheel contains the classical Essentia algorithms used here.
    !pip -q install "essentia-tensorflow>=2.1b6.dev1389" "pandas>=2.0" "tqdm>=4.66" "joblib>=1.4"
    """
)
cells[3] = md(
    """
    ## 2. Authenticate Google Drive once with rclone

    Open a **Jupyter Terminal** on the Vast.ai instance and run:

    ```bash
    rclone config
    ```

    Create a Google Drive remote named exactly `gdrive`. On a headless instance,
    answer `n` when asked whether to use a browser and follow the displayed
    `rclone authorize drive` instructions on your local computer. Do not paste an
    OAuth token into this notebook or commit `rclone.conf` to Git.

    Return here only after `rclone lsd gdrive:` succeeds in the terminal.
    """
)
cells[4] = md("## 3. Configuration")
cells[5] = code(
    """
    from pathlib import Path
    import gc
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    import time
    import traceback

    # Prevent each worker from creating its own large native thread pool.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    import essentia
    import essentia.standard as es
    from joblib import Parallel, delayed
    import numpy as np
    import pandas as pd
    from tqdm.auto import tqdm

    # Local NVMe working copy and matching Google Drive directory.
    PROJECT_ROOT = Path('/workspace/MTG_Jamendo_7324_full_quality')
    REMOTE_PROJECT = 'gdrive:MTG_Jamendo_7324_full_quality'
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
    RHYTHM_METHOD = 'multifeature'
    N_WORKERS = 4
    CHECKPOINT_EVERY = 10
    DRIVE_BACKUP_EVERY = 50
    RETRY_FAILED_TRACKS = True
    LIMIT_TRACKS = None  # Use 10 for a trial, then restore None.

    SOURCE = 'essentia_direct_first4min'
    SCHEMA_VERSION = 'essentia_rhythm_first4min_v1'
    FEATURE_COLUMNS = [
        'bpm', 'beats_count', 'beats_loudness_mean',
        'bpm_histogram_first_peak_bpm',
        'bpm_histogram_first_peak_spread',
        'bpm_histogram_first_peak_weight',
        'onset_rate', 'danceability',
        'beat_interval_mean', 'beat_interval_std',
    ]

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print('Essentia:', getattr(essentia, '__version__', 'unknown'))
    print('Workers:', N_WORKERS)
    print('Local project:', PROJECT_ROOT)
    print('Drive project:', REMOTE_PROJECT)
    """
)

drive_cells = [
    md("## 4. Verify Drive access and copy the dataset to local NVMe"),
    code(
        """
        def run_rclone(arguments, *, capture=False):
            command = ['rclone', *arguments]
            return subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=capture,
            )

        remotes = run_rclone(['listremotes'], capture=True).stdout.splitlines()
        assert 'gdrive:' in remotes, (
            'Missing rclone remote gdrive:. Open a Jupyter Terminal and run rclone config.'
        )
        run_rclone(['lsd', 'gdrive:'])
        print('Google Drive access verified.')
        """
    ),
    code(
        """
        # Safe to rerun: rclone skips files that already match.
        initial_usage = shutil.disk_usage(PROJECT_ROOT)
        required_free_bytes = 80 * 2**30
        assert initial_usage.free >= required_free_bytes, (
            f'Need at least 80 GiB free before download; found '
            f'{initial_usage.free / 2**30:.2f} GiB'
        )
        run_rclone([
            'copyto',
            f'{REMOTE_PROJECT}/split_csv.csv',
            str(CSV_PATH),
            '--progress',
        ])
        run_rclone([
            'copy',
            f'{REMOTE_PROJECT}/audio',
            str(AUDIO_ROOT),
            '--progress',
            '--transfers', '8',
            '--checkers', '16',
        ])
        print('Manifest present:', CSV_PATH.is_file())
        print('Downloaded MP3 files:', sum(1 for _ in AUDIO_ROOT.rglob('*.mp3')))
        usage = shutil.disk_usage(PROJECT_ROOT)
        print('Local free space (GiB):', round(usage.free / 2**30, 2))
        """
    ),
]

# Insert Drive transfer before the original manifest-loading section.
cells[6:6] = drive_cells

# Renumber inherited Colab section headings after inserting Vast.ai setup cells.
for cell in cells:
    if cell["cell_type"] != "markdown":
        continue
    heading = "".join(cell["source"])
    heading = heading.replace(
        "## 3. Load the selected-track manifest and resolve MP3 paths",
        "## 5. Load the selected-track manifest and resolve MP3 paths",
    )
    heading = heading.replace(
        "## 4. Feature-extraction implementation",
        "## 6. Feature-extraction implementation",
    )
    heading = heading.replace(
        "## 5. Smoke tests (run before full extraction)",
        "## 7. Smoke tests (run before full extraction)",
    )
    cell["source"] = heading.splitlines(keepends=True)

# Locate and replace the sequential full-extraction cell by its heading.
heading_index = next(
    index
    for index, cell in enumerate(cells)
    if cell["cell_type"] == "markdown"
    and "Resumable full-dataset extraction" in "".join(cell["source"])
)
cells[heading_index] = md("## 8. Four-worker resumable full-dataset extraction")
cells[heading_index + 1] = code(
    """
    METADATA_COLUMNS = [
        'song_id', 'split', 'source', 'schema_version', 'status', 'audio_path',
        'analysis_start_sec', 'analysis_end_sec', 'analysis_duration_sec',
        'decoded_duration_sec', 'input_scope', 'sample_rate', 'rhythm_method',
        'beat_tracker_confidence', 'detected_onsets_count', 'detected_beats_count',
        'elapsed_sec', 'error_type', 'error_message',
    ]
    ALL_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

    def load_records():
        if not CHECKPOINT_CSV.is_file():
            return {}
        frame = pd.read_csv(CHECKPOINT_CSV, dtype={'song_id': str}, keep_default_na=True)
        frame['song_id'] = frame['song_id'].map(normalize_song_id)
        return {row['song_id']: row.to_dict() for _, row in frame.iterrows()}

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

    def backup_results_to_drive():
        run_rclone([
            'copy', str(OUTPUT_ROOT), f'{REMOTE_PROJECT}/rhythm_features',
            '--exclude', '*.tmp.csv', '--checkers', '8',
        ])

    def process_track(row):
        song_id = row['song_id']
        started = time.time()
        base = {
            'song_id': song_id,
            'split': row.get('split', ''),
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
            record = {
                **base, **diagnostics, **features,
                'status': 'ok' if not missing else 'incomplete_features',
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
        return record

    records_by_id = load_records()
    pending = []
    for row in selected.to_dict(orient='records'):
        previous = records_by_id.get(row['song_id'])
        if previous is None or (RETRY_FAILED_TRACKS and previous.get('status') != 'ok'):
            pending.append(row)
    if LIMIT_TRACKS is not None:
        pending = pending[:int(LIMIT_TRACKS)]

    print('Loaded checkpoint rows:', len(records_by_id))
    print('Tracks to process now:', len(pending))
    print('Parallel workers:', N_WORKERS)

    if pending:
        result_stream = Parallel(
            n_jobs=N_WORKERS,
            backend='loky',
            batch_size=1,
            pre_dispatch='2*n_jobs',
            return_as='generator_unordered',
        )(delayed(process_track)(row) for row in pending)

        for completed, record in enumerate(
            tqdm(result_stream, total=len(pending), desc='Four-worker rhythm extraction'),
            start=1,
        ):
            records_by_id[record['song_id']] = record
            if completed % CHECKPOINT_EVERY == 0:
                save_checkpoint(records_by_id)
            if completed % DRIVE_BACKUP_EVERY == 0:
                save_checkpoint(records_by_id)
                backup_results_to_drive()
                gc.collect()

    checkpoint = save_checkpoint(records_by_id)
    backup_results_to_drive()
    print(checkpoint['status'].value_counts(dropna=False))
    """
)

# Rename the following finalization heading and append a final Drive verification.
final_heading_index = next(
    index
    for index, cell in enumerate(cells)
    if cell["cell_type"] == "markdown"
    and "Finalize, audit" in "".join(cell["source"])
)
cells[final_heading_index] = md("## 9. Finalize, audit, and save outputs")
cells.insert(
    final_heading_index + 2,
    code(
        """
        backup_results_to_drive()
        remote_listing = run_rclone(
            ['lsl', f'{REMOTE_PROJECT}/rhythm_features'], capture=True
        ).stdout
        print('Final Drive backup verified:')
        print(remote_listing)
        """
    ),
)

notebook["cells"] = cells
notebook["metadata"]["accelerator"] = "GPU"
notebook["metadata"].pop("colab", None)
notebook["metadata"]["vastai"] = {
    "cpu_workers": 4,
    "local_project_root": "/workspace/MTG_Jamendo_7324_full_quality",
    "drive_transport": "rclone-copy",
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
