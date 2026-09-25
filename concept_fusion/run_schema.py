"""Run / checkpoint / result schema. Tables are generated from these artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from concept_fusion.contract import (
    CONCEPT_DROPOUT_P,
    FUSED_DIM,
    MULTI_SEED_EXPERIMENTS,
    N_GENRE_TAGS,
    N_SEEDS_FINAL,
    N_SEEDS_OTHER,
    TOKEN_DIM,
    FUSION_CONTRACT_VERSION,
    PRIMARY_FUSION_INPUT_MODE,
)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def seeds_for(experiment_id: str) -> tuple[int, ...]:
    if experiment_id in MULTI_SEED_EXPERIMENTS:
        return tuple(range(N_SEEDS_FINAL))
    return (0,) * N_SEEDS_OTHER if N_SEEDS_OTHER == 1 else tuple(range(N_SEEDS_OTHER))[:N_SEEDS_OTHER]


def validate_fusion_checkpoint_metadata(
    payload: dict[str, Any], *, expected_input_mode: str
) -> None:
    """Reject legacy or differently routed checkpoints before loading weights."""
    version = payload.get("fusion_contract_version")
    if version != FUSION_CONTRACT_VERSION:
        raise RuntimeError(
            f"checkpoint fusion contract {version!r} is incompatible with "
            f"{FUSION_CONTRACT_VERSION!r}"
        )
    mode = payload.get("fusion_input_mode")
    if mode != expected_input_mode:
        raise RuntimeError(
            f"checkpoint fusion input mode {mode!r} does not match {expected_input_mode!r}"
        )


@dataclass
class RunConfig:
    experiment_id: str
    fusion: str
    seed: int
    dropout_p: float = CONCEPT_DROPOUT_P
    n_genre_tags: int = N_GENRE_TAGS
    token_dim: int = TOKEN_DIM
    fused_dim: int = FUSED_DIM
    allow_shortcut: bool = False
    lambda_rhythm: float = 1.0
    fusion_contract_version: str = FUSION_CONTRACT_VERSION
    fusion_input_mode: str = PRIMARY_FUSION_INPUT_MODE
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecord:
    config: dict[str, Any]
    config_hash: str
    git_sha: str
    seed: int
    experiment_id: str
    created_utc: str
    split_used_for_selection: str
    test_evaluated: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    n_valid_tags: int | None = None
    checkpoint_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(cls, cfg: RunConfig, **kwargs: Any) -> "RunRecord":
        d = cfg.as_dict()
        return cls(
            config=d,
            config_hash=config_hash(d),
            git_sha=git_sha(),
            seed=cfg.seed,
            experiment_id=cfg.experiment_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            split_used_for_selection="validation",
            test_evaluated=False,
            **kwargs,
        )


def load_records(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(results_dir.glob("*_seed*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if "experiment_id" not in data or "config_hash" not in data:
            continue
        rows.append(data)
    return rows
