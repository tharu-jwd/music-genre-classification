"""Generate the standalone instrument notebook; no project imports at runtime."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CELLS = []


def cell(kind, source, tag):
    item = {"cell_type": kind, "id": tag, "metadata": {"tags": [tag]},
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    CELLS.append(item)


cell("markdown", r'''
# Instrument concept branch

One notebook for representative-cohort auditing, a shared-input PyTorch branch,
masked loss, validation, checkpoints and integration. No Essentia dependency.

```
song_repr (B,128) -> LayerNorm -> residual 128D MLP
                 -> Linear(128,41) -> logits (B,41) -> sigmoid
                 -> concept_values (B,41) -> external fusion
```

Fusion consumes the 41 probabilities directly or owns a projection if equal-width
tokens are needed. This branch no longer returns a 64-D `fusion_token`.
Train jointly with genre loss plus masked concept losses; separate instrument
pretraining is optional. Hidden states cannot reach primary fusion. The optional
`window_repr (B,W,128)` is accepted and validated, but song-level inference does
not use it. Label-specific attention remains a team-approved future experiment.

Run locally, on Colab or on Kaggle. The contract tests run
without audio, downloads or encoder exports. Switches below enable
real data steps explicitly; disabled steps do not produce fabricated results.
''', "overview")

cell("code", r'''
# In a fresh notebook runtime, uncomment this installation line:
# %pip install torch numpy pandas scikit-learn
from pathlib import Path
import copy, hashlib, io, json, platform, random, re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

BASE = Path("/kaggle/working/MTG_Instrument") if Path("/kaggle").exists() else Path.cwd() / "data/instrument_run"
CFG = {
    "output": str(BASE),
    "manifest": str(BASE / "dataset/track_split_assignments.csv"),
    "genre_labels_csv": str(BASE / "dataset/genres_df.csv"),
    "instrument_labels_csv": str(BASE / "dataset/instrument_df.csv"),
    "representations": str(BASE / "shared_representations.npz"),
    "encoder_provenance": str(BASE / "shared_encoder_provenance.json"),
    "run_audit": False, "run_training": False,
    "run_test": False,
    "seeds": [17, 42, 73], "epochs": 40, "batch_size": 128, "lr": 0.001,
    "pos_weight": False, "annotated_only": False,
    "annotation_policy": "weak_closed_world",
}
OUT = Path(CFG["output"])

VOCAB = "accordion acousticbassguitar acousticguitar bass beat bell bongo brass cello clarinet classicalguitar computer doublebass drummachine drums electricguitar electricpiano flute guitar harmonica harp horn keyboard oboe orchestra organ pad percussion piano pipeorgan rhodes sampler saxophone strings synthesizer trombone trumpet ukulele viola violin voice".split()
TAGS = ["instrument---" + name for name in VOCAB]
SPLITS = ("train", "validation", "test")
GENRE_TAGS = "classical electronic folk hiphop jazz rock".split()
assert len(VOCAB) == len(set(VOCAB)) == 41

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def song_id(value):
    match = re.fullmatch(r"(?:track_)?(\d{1,7})", str(value).strip())
    if not match:
        raise ValueError(f"Invalid song_id: {value!r}")
    return match.group(1).zfill(7)
''', "setup")

cell("markdown", r'''
## Labels and coverage

This experiment uses the representative six-genre cohort defined by the split
manifest and `genre_labels_csv`. The instrument task uses all 41 tags, including
`ukulele`, from `instrument_labels_csv`. Both label tables are aligned by normalized
track ID. Rows absent from the instrument table receive no instrument supervision.

Default policy: on instrument-annotated rows, omitted instruments are **weak
benchmark negatives**, not verified absence. On missing annotation rows all
41 supervision entries are zero. `positive_only` is available for auditing,
but alone cannot support discriminative BCE training or full precision/AP
evaluation. Verified element-wise labels can instead be passed directly to
the branch and loss with mask 1 for known positives AND known negatives.
''', "label-policy")

cell("code", r'''
def annotation_arrays(ids, records, policy="weak_closed_world"):
    if policy not in {"weak_closed_world", "positive_only"}:
        raise ValueError("Unknown annotation policy")
    y = np.zeros((len(ids), 41), np.float32)
    mask = np.zeros_like(y)
    for i, sid in enumerate(ids):
        if sid in records:
            tags = records[sid]["tags"]
            if not tags <= set(TAGS):
                raise ValueError("Unexpected instrument vocabulary")
            y[i] = [tag in tags for tag in TAGS]
            mask[i] = 1 if policy == "weak_closed_world" else y[i]
    return y, mask

def read_instrument_labels_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"41-tag instrument labels CSV missing: {path}")
    frame = pd.read_csv(path, dtype={"TRACK_ID": str, "track_id": str, "song_id": str})
    id_columns = [name for name in ("TRACK_ID", "track_id", "song_id") if name in frame]
    if len(id_columns) != 1:
        raise ValueError("Instrument CSV needs exactly one ID column")
    names = VOCAB if set(VOCAB) <= set(frame) else TAGS
    if set(frame) != set(names) | set(id_columns):
        raise ValueError("Instrument CSV must contain exactly the 41 ordered vocabulary columns and one ID column")
    ids = frame[id_columns[0]].map(song_id)
    if ids.duplicated().any():
        raise ValueError("Duplicate instrument label IDs")
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Instrument labels must be finite binary values")
    return {sid: {"tags": {TAGS[j] for j, value in enumerate(row) if value == 1}}
            for sid, row in zip(ids, values)}

def read_genre_labels_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Six-genre labels CSV missing: {path}")
    frame = pd.read_csv(path, dtype={"TRACK_ID": str, "track_id": str, "song_id": str})
    id_columns = [name for name in ("TRACK_ID", "track_id", "song_id") if name in frame]
    if len(id_columns) != 1 or set(frame) != set(GENRE_TAGS) | set(id_columns):
        raise ValueError("Genre CSV must contain exactly six named genre columns and one ID column")
    ids = frame[id_columns[0]].map(song_id)
    if ids.duplicated().any():
        raise ValueError("Duplicate genre label IDs")
    values = frame[GENRE_TAGS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Genre labels must be finite binary values")
    if not (values.sum(axis=1) >= 1).all():
        raise ValueError("Representative cohort genre rows need at least one positive genre")
    return set(ids)

def audit_dataset(config):
    out = Path(config["output"])
    manifest = pd.read_csv(config["manifest"], dtype={"TRACK_ID": str, "track_id": str, "song_id": str})
    id_columns = [name for name in ("TRACK_ID", "track_id", "song_id") if name in manifest]
    if len(id_columns) != 1 or "split" not in manifest:
        raise ValueError("Manifest needs exactly one track ID column and a split column")
    manifest["song_id"] = manifest[id_columns[0]].map(song_id)
    if manifest.song_id.duplicated().any() or set(manifest.split) != set(SPLITS):
        raise ValueError("Manifest IDs must be unique and contain train, validation, and test")
    genre_ids = read_genre_labels_csv(config["genre_labels_csv"])
    if not set(manifest.song_id) <= genre_ids:
        raise ValueError("Manifest contains tracks without six-genre labels")
    all_instruments = read_instrument_labels_csv(config["instrument_labels_csv"])
    y, mask = annotation_arrays(manifest.song_id.tolist(), all_instruments, config["annotation_policy"])
    coverage, per_tag = [], []
    for split in SPLITS:
        selected = manifest.loc[manifest.split == split, "song_id"].tolist()
        coverage.append({"split": split, "manifest": len(selected),
                         "instrument_annotated": sum(s in all_instruments for s in selected),
                         "instrument_unannotated": sum(s not in all_instruments for s in selected)})
        for tag in TAGS:
            per_tag.append({"split": split, "tag": tag,
                            "observed": sum(s in all_instruments for s in selected),
                            "positive": sum(tag in all_instruments[s]["tags"] for s in selected if s in all_instruments)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(per_tag).to_csv(out / "coverage_per_tag.csv", index=False)
    pd.DataFrame({"song_id": manifest.song_id, "split": manifest.split,
                  "instrument_annotated": [s in all_instruments for s in manifest.song_id]}).to_csv(out / "observed_rows.csv", index=False)
    write_json(out / "instrument_vocabulary.json", TAGS)
    write_json(out / "genre_vocabulary.json", GENRE_TAGS)
    write_json(out / "annotation_provenance.json", {"manifest_sha256": digest(config["manifest"]),
               "genre_labels_csv_sha256": digest(config["genre_labels_csv"]),
               "instrument_labels_csv_sha256": digest(config["instrument_labels_csv"]),
               "policy": config["annotation_policy"], "genre_order": GENRE_TAGS,
               "instrument_order": TAGS, "order_source": "six-genre representative cohort; 41 instrument CSV columns aligned by name"})
    np.savez_compressed(out / "instrument_labels.npz", song_ids=manifest.song_id.to_numpy(dtype=str), targets=y, supervision_mask=mask)
    return manifest, y, mask, coverage

if CFG["run_audit"]:
    manifest, Y, M, coverage = audit_dataset(CFG)
    display(pd.read_csv(OUT / "coverage.csv"))
''', "audit")

cell("code", r'''
def checked_mask(mask, shape, reference):
    if mask is None:
        return reference.new_zeros(shape)
    mask = torch.as_tensor(mask, device=reference.device, dtype=reference.dtype)
    if tuple(mask.shape) != tuple(shape) or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"Expected binary mask with shape {shape}")
    return mask

BRANCH_VERSION = "song-128-residual-mlp-concepts-41-v3"

class InstrumentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(128)
        self.hidden = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.Dropout(0.1),
        )
        self.output_norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, 41)

    def forward(self, song_repr, window_repr=None, supervision_mask=None, fusion_mask=None):
        if song_repr.ndim != 2 or song_repr.shape[1] != 128 or not torch.isfinite(song_repr).all():
            raise ValueError("Expected finite song_repr (B,128)")
        if window_repr is not None and (
            window_repr.ndim != 3
            or window_repr.shape[0] != len(song_repr)
            or window_repr.shape[1] < 1
            or window_repr.shape[2] != 128
            or not torch.isfinite(window_repr).all()
        ):
            raise ValueError("Expected finite window_repr (B,W,128) with W >= 1")
        normalized = self.input_norm(song_repr)
        hidden = self.output_norm(normalized + self.hidden(normalized))
        logits = self.classifier(hidden)
        probabilities = logits.sigmoid()
        b = len(song_repr)
        sm = checked_mask(supervision_mask, (b, 41), probabilities)
        fm = probabilities.new_ones((b, 1)) if fusion_mask is None else checked_mask(fusion_mask, (b, 1), probabilities)
        return {"concept_values": probabilities, "logits": logits,
                "supervision_mask": sm, "fusion_mask": fm,
                "diagnostics": {"hidden": hidden.detach()}}

def masked_bce(logits, targets, mask, pos_weight=None):
    mask = checked_mask(mask, logits.shape, logits)
    targets = torch.as_tensor(targets, device=logits.device, dtype=logits.dtype)
    if targets.shape != logits.shape:
        raise ValueError("Targets must match logits")
    observed = mask.bool()
    values = targets[observed]
    if not torch.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Observed targets must be in [0,1]")
    # Unknown targets may contain NaN; remove them before BCE, not after it.
    safe_targets = torch.where(observed, targets, torch.zeros_like(targets))
    losses = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none", pos_weight=pos_weight)
    return (losses * mask).sum() / mask.sum().clamp_min(1)

def training_weights(targets, mask):
    positive = (targets * mask).sum(0)
    negative = ((1 - targets) * mask).sum(0)
    return torch.where((positive > 0) & (negative > 0), negative / positive.clamp_min(1), torch.ones_like(positive)).clamp(0.1, 20)

def instrument_objective(output, targets, pos_weight=None):
    return masked_bce(output["logits"], targets, output["supervision_mask"], pos_weight)
''', "branch")

cell("code", r'''
def metric_report(targets, probabilities, mask, thresholds=None):
    targets, probabilities, mask = map(np.asarray, (targets, probabilities, mask))
    if targets.shape != probabilities.shape or targets.shape != mask.shape or targets.ndim != 2 or targets.shape[1] != 41:
        raise ValueError("Metric inputs must be matching (N,41) arrays")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid metric probabilities or mask")
    thresholds = np.full(41, 0.5) if thresholds is None else np.asarray(thresholds)
    if thresholds.shape != (41,) or not np.isfinite(thresholds).all() or (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("Expected 41 thresholds")
    rows = []
    for j, tag in enumerate(TAGS):
        valid = mask[:, j].astype(bool)
        y, p = targets[valid, j], probabilities[valid, j]
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Metrics require binary observed labels")
        n, support = len(y), int(y.sum())
        both = 0 < support < n
        precision, recall, f1 = (None, None, None)
        if n:
            precision, recall, f1, _ = precision_recall_fscore_support(y, p >= thresholds[j], average="binary", zero_division=0)
        rows.append({"tag": tag, "observed": n, "support": support,
                     "AP": float(average_precision_score(y, p)) if both else None,
                     "ROC_AUC": float(roc_auc_score(y, p)) if both else None,
                     "precision": precision, "recall": recall, "F1": f1})
    macro = {"total_tags": 41, "undefined_policy": "AP/AUC require both classes; P/R/F1 use zero_division=0 on observed tags"}
    for name in ("AP", "ROC_AUC", "precision", "recall", "F1"):
        values = [float(r[name]) for r in rows if r[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
        macro[name + "_defined_tags"] = len(values)
    return {"macro": macro, "per_tag": rows}

def calibrate_thresholds(targets, probabilities, mask, split):
    if split != "validation":
        raise ValueError("Thresholds must be selected on validation only")
    thresholds = np.full(41, 0.5, np.float32)
    for j in range(41):
        valid = mask[:, j].astype(bool)
        y, p = targets[valid, j], probabilities[valid, j]
        if len(np.unique(y)) != 2:
            continue
        candidates = np.linspace(0.05, 0.95, 19)
        scores = [precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2] for t in candidates]
        best = np.flatnonzero(np.asarray(scores) == max(scores))
        index = min(best, key=lambda k: abs(candidates[k] - 0.5))
        thresholds[j] = candidates[index]
    return thresholds
''', "metrics")

cell("markdown", r'''
## Training and checkpoints

Dehan supplies `shared_representations.npz` with unique `song_ids` and
`song_repr (N,128)` aligned with the manifest. Optional
`window_repr (N,2,128)` is checked. Export in encoder evaluation mode.
`shared_encoder_provenance.json` must contain `encoder_sha256`,
`normalization_fit_split: "train"`, and `training_split: "train"`.
Frozen exports allow branch pretraining only; joint fine-tuning uses the live
encoder and the class above. The encoder owner remains responsible for the
two-half mel loader and train-only normalization.

Selection: highest validation macro AP over a fixed, reported set of tags with
both classes; test is never consulted. BCE is the default. Capped train-only
`pos_weight` is a configurable experiment, not a claimed improvement.
Threshold tuning changes decisions only, not the probabilities sent to fusion.
Three seeds are configured. Only real completed runs create metrics/checkpoints.
''', "training-notes")

cell("code", r'''
def aligned_npz(path, ids, key, shape):
    with np.load(path, allow_pickle=False) as data:
        source_ids = [song_id(s) for s in data["song_ids"]]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Duplicate representation IDs")
        lookup = {sid: i for i, sid in enumerate(source_ids)}
        if not set(ids) <= set(lookup):
            raise ValueError(f"{path} is missing manifest IDs")
        if len(data[key]) != len(source_ids):
            raise ValueError("Representation row count differs from ID count")
        values = data[key][[lookup[s] for s in ids]]
    if values.shape != (len(ids), *shape) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {key} shape or nonfinite values")
    return values.astype(np.float32)

def run_training(config):
    if config["annotation_policy"] != "weak_closed_world":
        raise ValueError("Positive-only labels cannot support this benchmark validation protocol")
    manifest, y, mask, _ = audit_dataset(config)
    out = Path(config["output"])
    ids = manifest.song_id.tolist()
    x = aligned_npz(config["representations"], ids, "song_repr", (128,))
    with np.load(config["representations"], allow_pickle=False) as archive:
        if "window_repr" in archive:
            aligned_npz(config["representations"], ids, "window_repr", (2, 128))
    provenance = json.loads(Path(config["encoder_provenance"]).read_text())
    if provenance.get("normalization_fit_split") != "train" or provenance.get("training_split") != "train" or not provenance.get("encoder_sha256"):
        raise ValueError("Missing train-only encoder/normalization provenance")
    train = np.flatnonzero(manifest.split.to_numpy() == "train")
    if config["annotated_only"]:
        train = train[mask[train].sum(axis=1) > 0]
    val = np.flatnonzero(manifest.split.to_numpy() == "validation")
    if not len(train) or not len(val):
        raise ValueError("Both train and validation representations are required")
    if not mask[train].any():
        raise ValueError("No observed training labels")
    missing_train = [TAGS[j] for j in range(41) if not (y[train, j] * mask[train, j]).any()]
    if missing_train:
        raise ValueError(f"41-tag training requires positive examples for every tag; missing: {missing_train}")
    x_t, y_t, m_t = [torch.from_numpy(a) for a in (x, y, mask)]
    weights = training_weights(y_t[train], m_t[train]) if config["pos_weight"] else None
    observed_counts = m_t[train].sum(0)
    rates = ((y_t[train] * m_t[train]).sum(0) / observed_counts.clamp_min(1)).tolist()
    prevalence = [rate if count > 0 else None for rate, count in zip(rates, observed_counts.tolist())]
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])) or config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Provide unique seeds, positive epochs and batch size")
    summary = []
    for seed in config["seeds"]:
        seed_all(seed)
        model = InstrumentBranch()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        best_score, best_state, best_epoch = -float("inf"), None, None
        history = []
        for epoch in range(config["epochs"]):
            model.train()
            for batch in np.array_split(np.random.permutation(train), max(1, int(np.ceil(len(train) / config["batch_size"])))):
                if not m_t[batch].any():
                    continue  # No optimizer momentum update for entirely unobserved batches.
                optimizer.zero_grad()
                output = model(x_t[batch], supervision_mask=m_t[batch])
                loss = instrument_objective(output, y_t[batch], weights)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                p = model(x_t[val])["concept_values"].numpy()
            score = metric_report(y[val], p, mask[val])["macro"]["AP"]
            if score is None:
                raise ValueError("Validation has no tags with both observed classes")
            history.append({"epoch": epoch + 1, "validation_macro_AP": score})
            if score > best_score:
                best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
        if best_state is None:
            raise ValueError("No training epochs completed")
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            val_p = model(x_t[val])["concept_values"].numpy()
        thresholds = calibrate_thresholds(y[val], val_p, mask[val], "validation")
        report = metric_report(y[val], val_p, mask[val], thresholds)
        run_dir = out / f"seed_{seed}"
        write_json(run_dir / "validation_metrics.json", report)
        pd.DataFrame(report["per_tag"]).to_csv(run_dir / "validation_per_tag.csv", index=False)
        write_json(run_dir / "history.json", history)
        checkpoint = {"state_dict": best_state, "instrument_vocabulary": TAGS, "thresholds": thresholds.tolist(),
                      "seed": seed, "config": config, "selected_epoch": best_epoch,
                      "target_prevalence": prevalence, "observed_train_counts": m_t[train].sum(0).tolist(),
                      "loss": {"name": "masked_BCE", "pos_weight": weights.tolist() if weights is not None else None},
                      "validation_metrics": report, "encoder": provenance,
                      "annotation": json.loads((out / "annotation_provenance.json").read_text()),
                      "representations_sha256": digest(config["representations"]),
                      "branch_version": BRANCH_VERSION,
                      "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
                      "fusion_projection_owner": "fusion", "training_stage": "instrument_pretraining"}
        torch.save(checkpoint, run_dir / "instrument.pt")
        summary.append({"seed": seed, **report["macro"]})
        print(f"seed={seed}, selected_epoch={best_epoch}, validation AP={best_score:.4f}")
    write_json(out / "seed_metrics.json", summary)
    ap = [r["AP"] for r in summary]
    write_json(out / "seed_summary.json", {"n_seeds": len(ap), "validation_AP_mean": float(np.mean(ap)),
               "validation_AP_std": float(np.std(ap)), "branch_version": BRANCH_VERSION})
    return summary

def load_branch(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("branch_version") != BRANCH_VERSION:
        raise ValueError("Incompatible branch architecture; legacy 64-D-token checkpoints require retraining")
    if checkpoint["instrument_vocabulary"] != TAGS:
        raise ValueError("Checkpoint vocabulary mismatch")
    model = InstrumentBranch()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint

def evaluate_locked_test(config, checkpoint_path):
    model, checkpoint = load_branch(checkpoint_path)
    if checkpoint["annotation"]["manifest_sha256"] != digest(config["manifest"]):
        raise ValueError("Manifest differs from checkpoint")
    if checkpoint["representations_sha256"] != digest(config["representations"]):
        raise ValueError("Encoder representations differ from checkpoint")
    if checkpoint["annotation"]["policy"] != config["annotation_policy"]:
        raise ValueError("Annotation policy differs from checkpoint")
    manifest, y, mask, _ = audit_dataset(config)
    current_annotation = json.loads((Path(config["output"]) / "annotation_provenance.json").read_text())
    if checkpoint["annotation"]["genre_labels_csv_sha256"] != current_annotation["genre_labels_csv_sha256"]:
        raise ValueError("Six-genre cohort labels differ from checkpoint")
    if checkpoint["annotation"]["instrument_labels_csv_sha256"] != current_annotation["instrument_labels_csv_sha256"]:
        raise ValueError("41-tag instrument labels differ from checkpoint")
    x = aligned_npz(config["representations"], manifest.song_id.tolist(), "song_repr", (128,))
    test = np.flatnonzero(manifest.split.to_numpy() == "test")
    if not len(test):
        raise ValueError("No test rows")
    with torch.no_grad():
        probabilities = model(torch.from_numpy(x[test]))["concept_values"].numpy()
    report = metric_report(y[test], probabilities, mask[test], checkpoint["thresholds"])
    write_json(Path(checkpoint_path).parent / "test_metrics.json", report)
    return report

if CFG["run_training"]:
    results = run_training(CFG)
if CFG["run_test"]:
    # Set only after freezing architecture/loss and completing all validation runs.
    for seed in CFG["seeds"]:
        evaluate_locked_test(CFG, OUT / f"seed_{seed}" / "instrument.pt")
''', "training")

cell("code", r'''
def run_contract_tests():
    seed_all(7)
    model = InstrumentBranch().eval()
    x, windows = torch.randn(4, 128), torch.randn(4, 2, 128)
    result = model(x, windows)
    for key, shape in [("concept_values", (4,41)), ("logits", (4,41)), ("supervision_mask", (4,41)), ("fusion_mask", (4,1))]:
        assert result[key].shape == shape and torch.isfinite(result[key]).all()
    assert "fusion_token" not in result
    assert sum(p.numel() for p in model.parameters()) == 38825
    assert torch.equal(result["concept_values"], result["logits"].sigmoid())
    removed = model(x, fusion_mask=torch.zeros(4,1))
    assert torch.equal(removed["concept_values"], result["concept_values"])
    assert (removed["concept_values"] * removed["fusion_mask"]).count_nonzero() == 0
    assert not result["diagnostics"]["hidden"].requires_grad
    assert torch.equal(model(x)["logits"], model(x)["logits"])
    model.train()
    assert not torch.equal(model(x)["logits"], model(x)["logits"])
    model.eval()
    missing_logits = torch.randn(2, 41, requires_grad=True)
    zero_loss = masked_bce(missing_logits, torch.full((2,41), float("nan")), torch.zeros(2,41))
    zero_loss.backward()
    assert zero_loss.item() == 0 and missing_logits.grad.count_nonzero() == 0
    known_logits = torch.zeros(1, 41, requires_grad=True)
    mask = torch.zeros(1, 41); mask[0, 0] = 1
    masked_bce(known_logits, torch.zeros_like(mask), mask).backward()
    assert known_logits.grad[0, 0] > 0 and known_logits.grad[0, 1] == 0
    y, m = annotation_arrays(["0000001", "0000002"], {"0000001": {"tags": {TAGS[0]}}})
    assert m[0].sum() == 41 and m[1].sum() == 0 and y[0,0] == 1
    masked = masked_bce(torch.zeros(2,41), torch.from_numpy(y), torch.from_numpy(m))
    annotated = masked_bce(torch.zeros(1,41), torch.from_numpy(y[:1]), torch.from_numpy(m[:1]))
    assert torch.equal(masked, annotated)
    weights = training_weights(torch.tensor([[1., 0.], [0., 0.], [0., 1.]]), torch.tensor([[1., 1.], [1., 1.], [0., 0.]]))
    assert torch.equal(weights, torch.ones(2))
    mixed_y = np.zeros((3,41)); mixed_y[0,0] = 1; mixed_y[:,1] = 1
    mixed_report = metric_report(mixed_y, np.full((3,41), 0.5), np.ones((3,41)))
    assert mixed_report["macro"]["AP_defined_tags"] == 1
    assert len(mixed_report["per_tag"]) == 41
    report = metric_report(np.zeros((3,41)), np.full((3,41), 0.5), np.ones((3,41)))
    assert report["macro"]["AP"] is None and report["macro"]["AP_defined_tags"] == 0
    try:
        calibrate_thresholds(y, y, m, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Test thresholds were allowed")
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer); buffer.seek(0)
    clone = InstrumentBranch().eval()
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(model(x)["logits"], clone(x)["logits"])
    tiny = InstrumentBranch()
    optimizer = torch.optim.Adam(tiny.parameters(), lr=0.03)
    targets = (torch.randn(4,41) > 0).float()
    for _ in range(100):
        optimizer.zero_grad()
        loss = masked_bce(tiny(x)["logits"], targets, torch.ones_like(targets))
        loss.backward(); optimizer.step()
    tiny.eval()
    final_loss = masked_bce(tiny(x)["logits"], targets, torch.ones_like(targets))
    assert final_loss.item() < 0.02, final_loss.item()
    # A synthetic downstream head verifies genre gradients without instrument labels.
    tiny.zero_grad()
    genre_head = nn.Linear(41, 6)
    live_repr = x.clone().requires_grad_()
    concepts = tiny(live_repr)
    assert concepts["supervision_mask"].count_nonzero() == 0
    genre_logits = genre_head(concepts["concept_values"] * concepts["fusion_mask"])
    F.binary_cross_entropy_with_logits(genre_logits, torch.zeros_like(genre_logits)).backward()
    assert tiny.classifier.weight.grad.abs().sum() > 0
    assert tiny.hidden[0].weight.grad.abs().sum() > 0
    assert live_repr.grad.abs().sum() > 0
    print("Contract checks passed: shapes, masks, bottleneck, metrics, serialization, overfit and gradients.")

run_contract_tests()
''', "contract-tests")

cell("markdown", r'''
## Integration and outstanding experiments

Thevindu receives 41 probabilities in `concept_values` and raw `logits` for stable
BCE. There is no `fusion_token` or learned projection in the branch. This is a
revised interface, incompatible with code expecting the original 64-D token.
Default supervision is unknown (all zero); fusion defaults to present (all one).
Probabilities remain unmasked for instrument supervision and interpretation.
Fusion must apply `fusion_mask` itself, after any biased projection, and mask
absent attention keys when using attention.

```python
window_repr, song_repr = shared_encoder(two_halves)  # team's actual return API may differ
instrument = branch(song_repr, window_repr, supervision_mask=instrument_mask)
# If equal-width tokens are needed, instrument_projection is an nn.Linear(41,64)
# owned by the fusion module and created once in its __init__.
instrument_token = fusion.instrument_projection(instrument['concept_values'])
instrument_token = instrument_token * instrument['fusion_mask']
tokens = torch.stack([instrument_token, rhythm['fusion_token'],
                      timbre['fusion_token'], harmony['fusion_token']], dim=1)
# tokens: (B,4,64); genre_logits: (B,6); genre loss uses BCEWithLogitsLoss.
# Concatenation-based fusion can instead use the masked 41 values directly.
# With the actual joint model's predictions:
# loss = genre_loss + lambda_instrument * instrument_objective(instrument, targets)
#        + other_concept_losses
# loss.backward()  # Optimize live encoder, branches, fusion and genre head together.
```

Any fusion-owned projection must join the genre optimizer. Never detach or
threshold the 41 probabilities during joint training.
For instrument removal, supply a zero fusion mask and mask its fusion key.
For an instrument-only model, train a genre head on its projected token.

Joint training can start directly; standalone pretraining is optional. If using
pretraining, compare frozen-branch fusion training with joint fine-tuning. During
joint training keep concept supervision to reduce drift. The hidden-vector
comparison must be a separately named ablation; the primary class exposes
hidden states detached for diagnostics only. Train its alternative fusion
projection/head separately, controlling encoder initialization and seeds.

No final genre head/shared encoder is implemented here because those belong
to the other branches. Therefore genre ablations, instrument-removal macro-AP
delta and joint-model acceptance remain pending integration. Attention is not
implemented without team approval. Per-half diagnostics are not the
requested learned-attention ablation. Likewise, annotated-only training and
masked full-batch training require matching loss normalization and optimization
steps for a fair comparison.

Before declaring completion, record validation results for unweighted/weighted
BCE, annotation sampling, hidden/concept fusion, frozen/joint training, and at
least three seeds. Run held-out test only after selection. Report mean/std,
per-tag support and the defined macro denominator. Threshold optimization is
not probability calibration; probabilities need not be empirically calibrated.
''', "integration")

cell("code", r'''
def explain_example(model, song_repr, window_repr, thresholds=None):
    if model.training:
        raise ValueError("Use model.eval() for repeatable diagnostic predictions")
    if song_repr.shape != (1,128) or window_repr.shape != (1,2,128):
        raise ValueError("Supply one song and its two real windows")
    with torch.no_grad():
        p = model(song_repr)["concept_values"][0].cpu().numpy()
        w = model(window_repr[0])["concept_values"].cpu().numpy()
    threshold = np.full(41, 0.5) if thresholds is None else np.asarray(thresholds)
    return pd.DataFrame({"instrument": VOCAB, "probability": p, "predicted": p >= threshold,
                         "first_half_probe": w[0], "second_half_probe": w[1],
                         "higher_scoring_probe": np.argmax(w, axis=0) + 1}).sort_values("probability", ascending=False)

# Per-window probes reuse a song-trained head and may be out of distribution.
# They are sensitivity diagnostics, not learned attention or causal attribution.
# model, checkpoint = load_branch(OUT / "seed_17/instrument.pt")
# display(explain_example(model, song_repr[:1], window_repr[:1], checkpoint["thresholds"]))
''', "examples")

cell("markdown", r'''
## Literature and design decisions

The table separates source methods from project-specific decisions. The dataset
paper's full-text endpoint was unavailable during implementation; its official
repository supplies the checked dataset details below.

| Source | Labels/input/architecture | Loss and missing labels | Evaluation and decision |
|---|---|---|---|
| [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset) and [paper](https://mtg.upf.edu/node/3957) | Uploader tags; audio/precomputed mels | Unlisted tags are not human-verified negatives. | Use the project's representative six-genre cohort and 41-instrument label table; audit coverage before training. |
| [OpenMIC-2018](https://brianmcfee.net/papers/ismir2018_openmic.pdf) | 10-second audio; 20 partially annotated classes with confirmed presence/absence. Baseline: independent random forests on mean/std VGGish features. | Baseline is a forest classifier, not neural BCE. Only annotated examples enter each instrument task; unknown labels are not negative examples. | Per-instrument accuracy over 100 sampled splits, with cross-validation for forest settings. Use explicit masks here; Jamendo lacks those confirmed negatives. |
| [Attention for instrument recognition](https://arxiv.org/pdf/1907.04294) | OpenMIC weak/partial labels; each clip is a bag of ten 128-D feature vectors; instance classifiers with label-specific weighted pooling. | Partial BCE excludes missing labels and rescales by observation fraction. | Precision/recall/F1 macro-average both polarities and instruments at 0.5, across ten seeds. Supports testing temporal attention, but does not establish benefit with just two Jamendo windows. |
| [Automatic tagging with CNNs](https://arxiv.org/pdf/1606.00298) | Multi-label MTAT/MSD tags; 29.1-second clips, 96-by-1366 log-mels; convolution/pooling stacks ending in independent sigmoids. | Binary cross-entropy with Adam. Binary tag vectors are used; an element-wise missing-annotation mask is not described in its training section. | ROC-AUC; mel/architecture comparisons. Supports mel-CNN representations and sigmoid/BCE; keep the encoder outside this branch. |
| [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf) | Image concepts and downstream labels in knee radiographs and bird images; explicit concept layer followed by task predictor. | Independent, sequential and joint training; joint objective weights concept and task losses. Requires concept annotations; does not supply a Jamendo missing-label policy. | Task/concept accuracy and interventions. Fusion sees instrument probabilities; compare downstream accuracy and concept fidelity, and retain supervision during joint training. |

Starting configuration: normalized 128-D residual MLP with GELU and dropout 0.1, unweighted
masked BCE and 41 sigmoid probabilities. Fusion owns any projection. Final loss/aggregation selection
remains validation-dependent. Retain the observed-label caveat in reports.
''', "literature")


def main():
    path = ROOT / "notebooks" / "03_instrument_branch.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "cells": CELLS}
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
