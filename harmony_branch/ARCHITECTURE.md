# Harmony branch architecture — current contract

The implemented architecture is **Harmony v4** (`ChromaGroundedHarmonyBranch`, git
branch `harmony-v2`). Its complete account is the
[v4 architecture snapshot](docs/architecture-versions/v4/README.md). The
[version register](docs/architecture-versions/README.md) records v3 and older
versions. This page is a short entry point, not a second specification.

```text
Shared encoder ordered features (B,T,128) + validity/window masks
  → 4 gap-safe dilated residual Conv1d blocks (B,T,96)
  → per-token chroma head → q_t (B,T,12)
      → exact: 12 chroma means = masked mean(q_t); 6 Tonnetz means = Φ·chroma mean
      → 26 token-chroma statistics (std, entropy, max-bin, L2 flux, Tonnetz movement)
  → attention + mean + std pooling → song embedding (B,64)
  → regression head [embedding, statistics] → 27 remaining descriptors
  → 45 transformed, standardized predictions (B,45)
  → fusion-owned Linear(45,64) → gated genre classifier
```

All [45 automatically extracted track descriptors](docs/feature-contract.md) are
targets. Targets get named log/log1m transforms and a training-only z-score; the
fitted parameters live in the model as buffers. Fusion consumes **predictions**,
not target values. The chord head is optional and disabled by default. The 64D
embedding is the embedding-fusion ablation input.

For exact target order, report metrics, checkpoint checks, and unresolved
evidence, read the [integration handoff](docs/integration-handoff.md). The
[temporal chroma/chord proposal](TEMPORAL_CHROMA_PROPOSAL.md) and its
[research plan](docs/temporal-chroma-research-plan.md) are historical/proposed
experiments, **not** instructions for the current v4 training run.
