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

One notebook for official split-0 auditing, a shared-input PyTorch branch,
masked loss, validation, checkpoints and integration. No Essentia dependency.

```
song_repr (B,128) -> Linear(128,128) -> ReLU -> Dropout(0.1)
                 -> Linear(128,40) -> logits (B,40) -> sigmoid
                 -> concept_values (B,40) -> external fusion
```

Fusion consumes the 40 probabilities directly or owns a projection if equal-width
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
import copy, csv, hashlib, io, json, platform, random, re, urllib.request
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support

BASE = Path("/kaggle/working/MTG_Instrument") if Path("/kaggle").exists() else Path.cwd() / "data/instrument_run"
CFG = {
    "output": str(BASE),
    "manifest": str(BASE / "dataset/song_manifest.csv"),
    "representations": str(BASE / "shared_representations.npz"),
    "encoder_provenance": str(BASE / "shared_encoder_provenance.json"),
    "run_audit": False, "run_training": False,
    "run_test": False,
    "seeds": [17, 42, 73], "epochs": 40, "batch_size": 128, "lr": 0.001,
    "pos_weight": False, "annotated_only": False,
    "annotation_policy": "weak_closed_world",
}
OUT = Path(CFG["output"])

VOCAB = "accordion acousticbassguitar acousticguitar bass beat bell bongo brass cello clarinet classicalguitar computer doublebass drummachine drums electricguitar electricpiano flute guitar harmonica harp horn keyboard oboe orchestra organ pad percussion piano pipeorgan rhodes sampler saxophone strings synthesizer trombone trumpet viola violin voice".split()
TAGS = ["instrument---" + name for name in VOCAB]
SPLITS = ("train", "validation", "test")
ANN_URL = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-0/"
assert len(VOCAB) == len(set(VOCAB)) == 40

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

def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(destination)
    if destination.stat().st_size == 0:
        raise ValueError(f"Empty download: {destination}")
    return destination

def song_id(value):
    match = re.fullmatch(r"(?:track_)?(\d{1,7})", str(value).strip())
    if not match:
        raise ValueError(f"Invalid song_id: {value!r}")
    return match.group(1).zfill(7)
''', "setup")

cell("markdown", r'''
## Labels and coverage

The official dataset contains uploader tags. The full instrument TSV has 41
tags; official splits retain 40. Split TSVs are sparse lists, so they do not
define a column order. We freeze alphabetical instrument order above and verify
the complete official files have exactly that tag set in every partition.

Default policy: on instrument-annotated rows, omitted instruments are **weak
benchmark negatives**, not verified absence. On missing annotation rows all
40 supervision entries are zero. `positive_only` is available for auditing,
but alone cannot support discriminative BCE training or full precision/AP
evaluation. Verified element-wise labels can instead be passed directly to
the branch and loss with mask 1 for known positives AND known negatives.
''', "label-policy")

cell("code", r'''
def read_official(path, prefix):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ValueError(f"Unexpected TSV schema: {header}")
        for row in reader:
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"Malformed annotation row: {row}")
            sid = song_id(row[0])
            tags = set(row[5:])
            if sid in records or not tags or any(not t.startswith(prefix) for t in tags):
                raise ValueError(f"Duplicate ID or unexpected tags: {sid}")
            records[sid] = {"tags": tags, "path": row[3]}
    return records

def annotation_arrays(ids, records, policy="weak_closed_world"):
    if policy not in {"weak_closed_world", "positive_only"}:
        raise ValueError("Unknown annotation policy")
    y = np.zeros((len(ids), 40), np.float32)
    mask = np.zeros_like(y)
    for i, sid in enumerate(ids):
        if sid in records:
            tags = records[sid]["tags"]
            if not tags <= set(TAGS):
                raise ValueError("Unexpected instrument vocabulary")
            y[i] = [tag in tags for tag in TAGS]
            mask[i] = 1 if policy == "weak_closed_world" else y[i]
    return y, mask

def audit_dataset(config):
    out = Path(config["output"])
    official, hashes = {}, {}
    for category, prefix, count in [("instrument", "instrument---", 40), ("genre", "genre---", 87)]:
        official[category] = {}
        for split in SPLITS:
            name = f"autotagging_{category}-{split}.tsv"
            path = fetch(ANN_URL + name, out / "annotations" / name)
            records = read_official(path, prefix)
            official[category][split] = records
            hashes[name] = digest(path)
            vocabulary = set().union(*(r["tags"] for r in records.values()))
            if len(vocabulary) != count:
                raise ValueError(f"Expected {count} official {category} tags")
            if category == "instrument" and vocabulary != set(TAGS):
                raise ValueError("Official instrument vocabulary differs from frozen order")
        groups = [set(official[category][s]) for s in SPLITS]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Official partitions overlap")
    # Check category partitions agree, including instrument-only tracks.
    split_of = {}
    for category in official:
        for split, records in official[category].items():
            for sid in records:
                if sid in split_of and split_of[sid] != split:
                    raise ValueError("Cross-category split conflict")
                split_of[sid] = split
    genre_order = sorted(set().union(*(r["tags"] for r in official["genre"]["train"].values())))
    for split in SPLITS:
        if set().union(*(r["tags"] for r in official["genre"][split].values())) != set(genre_order):
            raise ValueError("Genre tag set varies across partitions")
    manifest = pd.read_csv(config["manifest"], dtype={"song_id": str})
    if not {"song_id", "split"} <= set(manifest):
        raise ValueError("Manifest needs song_id and split columns")
    manifest["song_id"] = manifest.song_id.map(song_id)
    if manifest.song_id.duplicated().any() or not set(manifest.split) <= set(SPLITS):
        raise ValueError("Manifest IDs must be unique with official split names")
    for row in manifest.itertuples():
        if row.song_id not in official["genre"][row.split]:
            raise ValueError(f"Manifest disagrees with official genre split: {row.song_id}")
    all_instruments = {sid: row for split in SPLITS for sid, row in official["instrument"][split].items()}
    y, mask = annotation_arrays(manifest.song_id.tolist(), all_instruments, config["annotation_policy"])
    coverage, per_tag = [], []
    for split in SPLITS:
        inst = official["instrument"][split]
        genre = official["genre"][split]
        selected = manifest.loc[manifest.split == split, "song_id"].tolist()
        coverage.append({"split": split, "official_genre": len(genre), "official_instrument": len(inst),
                         "official_overlap": len(set(genre) & set(inst)), "manifest": len(selected),
                         "manifest_annotated": sum(s in inst for s in selected)})
        for tag in TAGS:
            per_tag.append({"split": split, "tag": tag,
                            "official_positive": sum(tag in r["tags"] for r in inst.values()),
                            "manifest_positive": sum(tag in inst[s]["tags"] for s in selected if s in inst)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(per_tag).to_csv(out / "coverage_per_tag.csv", index=False)
    pd.DataFrame({"song_id": manifest.song_id, "split": manifest.split,
                  "instrument_annotated": [s in all_instruments for s in manifest.song_id]}).to_csv(out / "observed_rows.csv", index=False)
    write_json(out / "instrument_vocabulary.json", TAGS)
    write_json(out / "genre_vocabulary.json", genre_order)
    write_json(out / "annotation_provenance.json", {"sha256": hashes, "manifest_sha256": digest(config["manifest"]),
               "policy": config["annotation_policy"], "order_source": "alphabetical, verified against all complete official split-0 files"})
    np.savez_compressed(out / "instrument_labels.npz", song_ids=manifest.song_id.to_numpy(dtype=str), targets=y, supervision_mask=mask)
    return manifest, y, mask, official

if CFG["run_audit"]:
    manifest, Y, M, official = audit_dataset(CFG)
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

BRANCH_VERSION = "song-128-hidden-128-dropout-0.1-concepts-40-v2"

class InstrumentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(128, 40)

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
        hidden = self.hidden(song_repr)
        logits = self.classifier(hidden)
        probabilities = logits.sigmoid()
        b = len(song_repr)
        sm = checked_mask(supervision_mask, (b, 40), probabilities)
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
    if targets.shape != probabilities.shape or targets.shape != mask.shape or targets.ndim != 2 or targets.shape[1] != 40:
        raise ValueError("Metric inputs must be matching (N,40) arrays")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid metric probabilities or mask")
    thresholds = np.full(40, 0.5) if thresholds is None else np.asarray(thresholds)
    if thresholds.shape != (40,) or not np.isfinite(thresholds).all() or (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("Expected 40 thresholds")
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
    macro = {"total_tags": 40, "undefined_policy": "AP/AUC require both classes; P/R/F1 use zero_division=0 on observed tags"}
    for name in ("AP", "ROC_AUC", "precision", "recall", "F1"):
        values = [float(r[name]) for r in rows if r[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
        macro[name + "_defined_tags"] = len(values)
    return {"macro": macro, "per_tag": rows}

def calibrate_thresholds(targets, probabilities, mask, split):
    if split != "validation":
        raise ValueError("Thresholds must be selected on validation only")
    thresholds = np.full(40, 0.5, np.float32)
    for j in range(40):
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
    if checkpoint["annotation"]["sha256"] != current_annotation["sha256"]:
        raise ValueError("Official annotation files differ from checkpoint")
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
    for key, shape in [("concept_values", (4,40)), ("logits", (4,40)), ("supervision_mask", (4,40)), ("fusion_mask", (4,1))]:
        assert result[key].shape == shape and torch.isfinite(result[key]).all()
    assert "fusion_token" not in result
    assert sum(p.numel() for p in model.parameters()) == 21672
    assert torch.equal(result["concept_values"], result["logits"].sigmoid())
    removed = model(x, fusion_mask=torch.zeros(4,1))
    assert torch.equal(removed["concept_values"], result["concept_values"])
    assert (removed["concept_values"] * removed["fusion_mask"]).count_nonzero() == 0
    assert not result["diagnostics"]["hidden"].requires_grad
    assert torch.equal(model(x)["logits"], model(x)["logits"])
    model.train()
    assert not torch.equal(model(x)["logits"], model(x)["logits"])
    model.eval()
    missing_logits = torch.randn(2, 40, requires_grad=True)
    zero_loss = masked_bce(missing_logits, torch.full((2,40), float("nan")), torch.zeros(2,40))
    zero_loss.backward()
    assert zero_loss.item() == 0 and missing_logits.grad.count_nonzero() == 0
    known_logits = torch.zeros(1, 40, requires_grad=True)
    mask = torch.zeros(1, 40); mask[0, 0] = 1
    masked_bce(known_logits, torch.zeros_like(mask), mask).backward()
    assert known_logits.grad[0, 0] > 0 and known_logits.grad[0, 1] == 0
    y, m = annotation_arrays(["0000001", "0000002"], {"0000001": {"tags": {TAGS[0]}}})
    assert m[0].sum() == 40 and m[1].sum() == 0 and y[0,0] == 1
    masked = masked_bce(torch.zeros(2,40), torch.from_numpy(y), torch.from_numpy(m))
    annotated = masked_bce(torch.zeros(1,40), torch.from_numpy(y[:1]), torch.from_numpy(m[:1]))
    assert torch.equal(masked, annotated)
    weights = training_weights(torch.tensor([[1., 0.], [0., 0.], [0., 1.]]), torch.tensor([[1., 1.], [1., 1.], [0., 0.]]))
    assert torch.equal(weights, torch.ones(2))
    mixed_y = np.zeros((3,40)); mixed_y[0,0] = 1; mixed_y[:,1] = 1
    mixed_report = metric_report(mixed_y, np.full((3,40), 0.5), np.ones((3,40)))
    assert mixed_report["macro"]["AP_defined_tags"] == 1
    assert len(mixed_report["per_tag"]) == 40
    report = metric_report(np.zeros((3,40)), np.full((3,40), 0.5), np.ones((3,40)))
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
    targets = (torch.randn(4,40) > 0).float()
    for _ in range(100):
        optimizer.zero_grad()
        loss = masked_bce(tiny(x)["logits"], targets, torch.ones_like(targets))
        loss.backward(); optimizer.step()
    tiny.eval()
    final_loss = masked_bce(tiny(x)["logits"], targets, torch.ones_like(targets))
    assert final_loss.item() < 0.02, final_loss.item()
    # A synthetic downstream head verifies genre gradients without instrument labels.
    tiny.zero_grad()
    genre_head = nn.Linear(40, 87)
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

Thevindu receives 40 probabilities in `concept_values` and raw `logits` for stable
BCE. There is no `fusion_token` or learned projection in the branch. This is a
revised interface, incompatible with code expecting the original 64-D token.
Default supervision is unknown (all zero); fusion defaults to present (all one).
Probabilities remain unmasked for instrument supervision and interpretation.
Fusion must apply `fusion_mask` itself, after any biased projection, and mask
absent attention keys when using attention.

```python
window_repr, song_repr = shared_encoder(two_halves)  # team's actual return API may differ
instrument = branch(song_repr, window_repr, supervision_mask=instrument_mask)
# If equal-width tokens are needed, instrument_projection is an nn.Linear(40,64)
# owned by the fusion module and created once in its __init__.
instrument_token = fusion.instrument_projection(instrument['concept_values'])
instrument_token = instrument_token * instrument['fusion_mask']
tokens = torch.stack([instrument_token, rhythm['fusion_token'],
                      timbre['fusion_token'], harmony['fusion_token']], dim=1)
# tokens: (B,4,64); genre_logits: (B,87); genre loss uses BCEWithLogitsLoss.
# Concatenation-based fusion can instead use the masked 40 values directly.
# With the actual joint model's predictions:
# loss = genre_loss + lambda_instrument * instrument_objective(instrument, targets)
#        + other_concept_losses
# loss.backward()  # Optimize live encoder, branches, fusion and genre head together.
```

Any fusion-owned projection must join the genre optimizer. Never detach or
threshold the 40 probabilities during joint training.
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
    threshold = np.full(40, 0.5) if thresholds is None else np.asarray(thresholds)
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
| [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset) and [paper](https://mtg.upf.edu/node/3957) | Uploader tags; audio/precomputed mels; official baseline workflow | Unlisted tags are not human-verified negatives; split subsets drop rows without category tags. Paper-specific loss needs full-text verification. | Fixed 40 instruments/87 genres from official split; audit coverage before training. |
| [OpenMIC-2018](https://brianmcfee.net/papers/ismir2018_openmic.pdf) | 10-second audio; 20 partially annotated classes with confirmed presence/absence. Baseline: independent random forests on mean/std VGGish features. | Baseline is a forest classifier, not neural BCE. Only annotated examples enter each instrument task; unknown labels are not negative examples. | Per-instrument accuracy over 100 sampled splits, with cross-validation for forest settings. Use explicit masks here; Jamendo lacks those confirmed negatives. |
| [Attention for instrument recognition](https://arxiv.org/pdf/1907.04294) | OpenMIC weak/partial labels; each clip is a bag of ten 128-D feature vectors; instance classifiers with label-specific weighted pooling. | Partial BCE excludes missing labels and rescales by observation fraction. | Precision/recall/F1 macro-average both polarities and instruments at 0.5, across ten seeds. Supports testing temporal attention, but does not establish benefit with just two Jamendo windows. |
| [Automatic tagging with CNNs](https://arxiv.org/pdf/1606.00298) | Multi-label MTAT/MSD tags; 29.1-second clips, 96-by-1366 log-mels; convolution/pooling stacks ending in independent sigmoids. | Binary cross-entropy with Adam. Binary tag vectors are used; an element-wise missing-annotation mask is not described in its training section. | ROC-AUC; mel/architecture comparisons. Supports mel-CNN representations and sigmoid/BCE; keep the encoder outside this branch. |
| [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf) | Image concepts and downstream labels in knee radiographs and bird images; explicit concept layer followed by task predictor. | Independent, sequential and joint training; joint objective weights concept and task losses. Requires concept annotations; does not supply a Jamendo missing-label policy. | Task/concept accuracy and interventions. Fusion sees instrument probabilities; compare downstream accuracy and concept fidelity, and retain supervision during joint training. |

Starting configuration: 128-D hidden layer with ReLU and dropout 0.1, unweighted
masked BCE and 40 sigmoid probabilities. Fusion owns any projection. Final loss/aggregation selection
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
