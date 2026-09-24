"""Shared MTG-Jamendo parsing code embedded into generated hosted notebooks."""

NOTEBOOK_DATA_CONTRACT = r'''
import csv

LOGMEL_SCHEMA_VERSION = "mtg_full_audio_logmel_windows_v1"
LOGMEL_N_MELS = 96
LOGMEL_WINDOW_FRAMES = 1366
LOGMEL_MAX_WINDOWS = 12
LOGMEL_SAMPLE_RATE = 12000
LOGMEL_HOP_LENGTH = 256


def normalize_track_id(raw) -> str | None:
    """Return the canonical seven-digit ID used by every project artifact."""
    value = str(raw).strip()
    match = re.fullmatch(r"(?:track_)?(\d+)", value, flags=re.IGNORECASE)
    if match is None:
        return None
    return f"{int(match.group(1)):07d}"


def iter_tsv_rows(path: Path):
    """Parse MTG's six fixed fields plus its variable number of tag columns."""
    with Path(path).open(newline="", encoding="utf-8", errors="strict") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        expected = ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]
        if header != expected:
            raise ValueError(f"unexpected MTG header in {path}: {header!r}")
        for line_number, fields in enumerate(reader, start=2):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected at least 6 tab-separated fields")
            song_id = normalize_track_id(fields[0])
            if song_id is None:
                raise ValueError(f"{path}:{line_number}: invalid TRACK_ID {fields[0]!r}")
            tags = tuple(field.strip() for field in fields[5:] if field.strip())
            yield {
                "TRACK_ID": fields[0],
                "song_id": song_id,
                "ARTIST_ID": fields[1],
                "ALBUM_ID": fields[2],
                "PATH": fields[3],
                "DURATION": fields[4],
                "TAGS": tags,
            }


def _canonical_logmel(raw, n_mels: int):
    mel = np.asarray(raw, dtype=np.float32)
    mel = np.squeeze(mel)
    if mel.ndim == 2:
        if mel.shape[0] == n_mels:
            full = mel
        elif mel.shape[1] == n_mels:
            full = mel.T
        else:
            raise ValueError(f"expected one log-Mel axis of size {n_mels}, got {mel.shape}")
    elif mel.ndim == 3:
        if mel.shape[1] == n_mels:
            full = mel.transpose(1, 0, 2).reshape(n_mels, -1)
        elif mel.shape[2] == n_mels:
            full = mel.transpose(2, 0, 1).reshape(n_mels, -1)
        else:
            raise ValueError(f"expected stacked log-Mels with {n_mels} bands, got {mel.shape}")
    else:
        raise ValueError(f"expected a 2D song log-Mel or 3D window stack, got {mel.shape}")

    if full.shape[1] == 0:
        raise ValueError("log-Mel has no time frames")
    return full


def _selected_chunk_indices(frame_count: int, n_frames: int, max_windows: int):
    if frame_count < 1 or n_frames < 1 or max_windows < 1:
        raise ValueError("frame counts and window limits must be positive")
    n_chunks = (frame_count + n_frames - 1) // n_frames
    if n_chunks <= max_windows:
        return np.arange(n_chunks, dtype=int)
    return np.linspace(0, n_chunks - 1, max_windows, dtype=int)


def logmel_window_plan(
    raw, *, n_mels: int, n_frames: int, max_windows: int,
    sample_rate: int = LOGMEL_SAMPLE_RATE, hop_length: int = LOGMEL_HOP_LENGTH,
):
    """Describe the exact ordered source-frame/time regions consumed by the model."""
    if sample_rate < 1 or hop_length < 1:
        raise ValueError("sample_rate and hop_length must be positive")
    full = _canonical_logmel(raw, n_mels)
    plan = []
    for output_index, chunk_index in enumerate(
        _selected_chunk_indices(full.shape[1], n_frames, max_windows)
    ):
        frame_start = int(chunk_index) * n_frames
        frame_end = min(frame_start + n_frames, full.shape[1])
        plan.append({
            "window_index": output_index,
            "chunk_index": int(chunk_index),
            "frame_start": frame_start,
            "frame_end_exclusive": frame_end,
            "valid_frames": frame_end - frame_start,
            "start_seconds": frame_start * hop_length / sample_rate,
            "end_seconds": frame_end * hop_length / sample_rate,
        })
    return plan


def segment_logmel_with_metadata(raw, *, n_mels: int, n_frames: int, max_windows: int):
    """Return windows plus the exact valid-frame counts and song-relative starts."""
    full = _canonical_logmel(raw, n_mels)
    plan = logmel_window_plan(
        full, n_mels=n_mels, n_frames=n_frames, max_windows=max_windows,
    )

    windows = np.zeros((max_windows, n_mels, n_frames), dtype=np.float32)
    mask = np.zeros(max_windows, dtype=np.float32)
    valid_frames = np.zeros(max_windows, dtype=np.int64)
    start_seconds = np.zeros(max_windows, dtype=np.float32)
    for item in plan:
        output_index = item["window_index"]
        start = item["frame_start"]
        piece = full[:, start:item["frame_end_exclusive"]]
        windows[output_index, :, :piece.shape[1]] = piece
        mask[output_index] = 1.0
        valid_frames[output_index] = item["valid_frames"]
        start_seconds[output_index] = item["start_seconds"]
    return windows, mask, valid_frames, start_seconds


def segment_logmel(raw, *, n_mels: int, n_frames: int, max_windows: int):
    """Return ordered, evenly covered song windows and a real-window mask."""
    windows, mask, _, _ = segment_logmel_with_metadata(
        raw, n_mels=n_mels, n_frames=n_frames, max_windows=max_windows,
    )
    return windows, mask


def split_annotation_path(split: str, subset: str = "genre") -> Path:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split!r}")
    path = ANN_DIR / "splits" / "split-0" / f"autotagging_{subset}-{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_split_ids(split: str, subset: str = "genre") -> set[str]:
    path = split_annotation_path(split, subset)
    ids = {row["song_id"] for row in iter_tsv_rows(path)}
    if not ids:
        raise ValueError(f"no track IDs parsed from {path}")
    print(f"{split:12s} {len(ids):6d} ids ← {path}")
    return ids


def load_split_multihot(song_ids, subset: str, category: str):
    """Load fixed split-0 vocabulary and labels without treating missing rows as negatives."""
    annotations = {}
    split_for_song = {}
    vocabulary_by_split = {}
    prefix = f"{category}---"

    for split in ("train", "validation", "test"):
        path = split_annotation_path(split, subset)
        split_vocabulary = set()
        for row in iter_tsv_rows(path):
            sid = row["song_id"]
            if sid in split_for_song:
                raise ValueError(f"song {sid} occurs in both {split_for_song[sid]} and {split}")
            tags = {tag for tag in row["TAGS"] if tag.startswith(prefix)}
            unexpected = set(row["TAGS"]) - tags
            if unexpected:
                raise ValueError(f"unexpected non-{category} tags in {path}: {sorted(unexpected)[:3]}")
            if not tags:
                raise ValueError(f"song {sid} has no {category} tags in {path}")
            annotations[sid] = tags
            split_for_song[sid] = split
            split_vocabulary.update(tags)
        vocabulary_by_split[split] = split_vocabulary

    reference = vocabulary_by_split["train"]
    for split in ("validation", "test"):
        if vocabulary_by_split[split] != reference:
            missing = sorted(reference - vocabulary_by_split[split])
            extra = sorted(vocabulary_by_split[split] - reference)
            raise ValueError(
                f"{subset} vocabulary differs in {split}; missing={missing[:5]} extra={extra[:5]}"
            )

    label_names = sorted(reference)
    label_index = {name: index for index, name in enumerate(label_names)}
    normalized_ids = [normalize_track_id(sid) for sid in song_ids]
    if any(sid is None for sid in normalized_ids):
        raise ValueError("song_ids contains a non-MTG identifier")
    targets = np.zeros((len(normalized_ids), len(label_names)), dtype=np.float32)
    available = np.zeros(len(normalized_ids), dtype=bool)
    for row_index, sid in enumerate(normalized_ids):
        tags = annotations.get(sid)
        if tags is None:
            continue
        available[row_index] = True
        for tag in tags:
            targets[row_index, label_index[tag]] = 1.0
    return targets, label_names, available
'''
