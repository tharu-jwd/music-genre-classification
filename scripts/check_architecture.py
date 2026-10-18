"""Validate architecture JSON and show model sizes without audio, training or a GPU."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_encoder.architecture import load_architecture, configure_training
from scripts import train_joint as j
from scripts.train_cnn import GenreCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg = j.TrainConfig(architecture=load_architecture(args.config), device="cpu")
    configure_training(cfg)
    modules = j.build_concept_modules(cfg, "cpu")
    print("Architecture config valid.")
    print(f"Genre fusion: {cfg.fusion}; predictor: {cfg.genre_predictor}")
    total = 0
    for name, module in modules.items():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += count
        print(f"{name}: {count:,} trainable parameters")
    print(f"Full model total: {total:,}")
    baseline = GenreCNN(cfg.architecture)
    print(f"Direct CNN total: {sum(p.numel() for p in baseline.parameters()):,}")


if __name__ == "__main__":
    main()
