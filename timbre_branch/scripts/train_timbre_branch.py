"""Train the strict 128-to-35 timbre branch from saved encoder embeddings.

Embedding NPZ contract:
  track_ids: [N] strings matching TRACK_ID
  embeddings: [N, 128] float array

Split CSV contract:
  TRACK_ID and split, where split is train, validation, or test.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from timbre_branch.constants import (
    FEATURE_COLUMNS,
    V2_EARLY_STOPPING_PATIENCE,
    V2_MAX_EPOCHS,
    V2_TIMBRE_BRANCH_LR,
)
from timbre_branch.data import TimbreEmbeddingDataset
from timbre_branch.model import TimbreBranch, TimbreBranchConfig
from timbre_branch.preprocessing import (
    TimbreStandardizer,
    align_embeddings_and_targets,
    load_timbre_targets,
    normalize_track_id,
)
from timbre_branch.training import (
    build_v2_scheduler,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=REPOSITORY_ROOT / "data/timbre_df.csv")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "checkpoints/timbre_branch/best.pt")
    parser.add_argument("--epochs", type=int, default=V2_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=V2_TIMBRE_BRANCH_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=V2_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    seed_everything(args.seed)

    archive = np.load(args.embeddings, allow_pickle=False)
    if set(("track_ids", "embeddings")).difference(archive.files):
        raise ValueError("Embedding NPZ must contain track_ids and embeddings")
    track_ids = archive["track_ids"].astype(str)
    embeddings = archive["embeddings"].astype(np.float32)
    targets = load_timbre_targets(args.targets)
    embeddings, raw_targets, valid_mask = align_embeddings_and_targets(
        track_ids, embeddings, targets
    )

    splits = pd.read_csv(args.splits, dtype=str)
    split_id_column = "TRACK_ID" if "TRACK_ID" in splits else "track_id" if "track_id" in splits else None
    if split_id_column is None or "split" not in splits:
        raise ValueError("Split CSV must contain TRACK_ID (or track_id) and split")
    splits = splits.rename(columns={split_id_column: "TRACK_ID"})
    splits["TRACK_ID"] = splits["TRACK_ID"].map(normalize_track_id)
    track_ids = np.asarray([normalize_track_id(value) for value in track_ids], dtype=str)
    if splits["TRACK_ID"].duplicated().any():
        raise ValueError("Split CSV contains duplicate TRACK_ID values")
    split_by_id = splits.set_index("TRACK_ID")["split"]
    missing_split = sorted(set(track_ids).difference(split_by_id.index))
    if missing_split:
        raise ValueError(f"Missing split assignment for {len(missing_split)} tracks")
    split_values = split_by_id.loc[track_ids].to_numpy(dtype=str)
    allowed = {"train", "validation", "test"}
    unexpected = sorted(set(split_values).difference(allowed))
    if unexpected:
        raise ValueError(f"Unexpected split names: {unexpected}")

    train_rows = split_values == "train"
    standardizer = TimbreStandardizer().fit(raw_targets[train_rows], valid_mask[train_rows])
    standardized = standardizer.transform(raw_targets)
    # Invalid cells remain harmless placeholders and are excluded by the mask.
    standardized = np.where(valid_mask, standardized, 0.0).astype(np.float32)

    def make_loader(split: str, shuffle: bool = False) -> DataLoader:
        selected = split_values == split
        if not selected.any():
            raise ValueError(f"No rows found for split={split}")
        dataset = TimbreEmbeddingDataset(
            track_ids[selected], embeddings[selected], standardized[selected], valid_mask[selected]
        )
        generator = torch.Generator().manual_seed(args.seed)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
        )

    train_loader = make_loader("train", shuffle=True)
    validation_loader = make_loader("validation")
    test_loader = make_loader("test")

    device = torch.device(args.device)
    model = TimbreBranch(TimbreBranchConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # This branch-only utility has no genre labels, so it uses validation timbre
    # MAE in min mode. The full joint trainer must use build_v2_scheduler()'s
    # default max mode and step it with validation macro average precision.
    scheduler = build_v2_scheduler(optimizer, mode="min")

    best_validation = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation = evaluate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **validation}
        history.append(record)
        score = validation["macro_mae_standardized"]
        scheduler.step(score)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        record["learning_rates"] = learning_rates
        print(f"epoch={epoch:03d} train={train_loss:.5f} val_macro_mae={score:.5f}")
        if score < best_validation:
            best_validation = score
            epochs_without_improvement = 0
            save_checkpoint(
                args.output,
                model,
                standardizer,
                epoch=epoch,
                metrics=validation,
                seed=args.seed,
                optimizer=optimizer,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    from timbre_branch.training import load_checkpoint

    best_model, _, payload = load_checkpoint(args.output, device)
    test_metrics = evaluate(best_model, test_loader, device)
    result_path = args.output.with_suffix(".metrics.json")
    result_path.write_text(
        json.dumps(
            {
                "best_epoch": payload["epoch"],
                "validation": payload["metrics"],
                "test": test_metrics,
                "history": history,
                "feature_names": list(FEATURE_COLUMNS),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"checkpoint={args.output}")
    print(f"metrics={result_path}")


if __name__ == "__main__":
    main()
