# Harmony branch architecture — version entry point

The current joint trainer defaults to **Harmony v4**, an implemented 45-target
configuration without a training result. Read the [v4 implementation record](docs/architecture-versions/v4/README.md)
for its exact contract. The completed **Harmony v3** run has a complete, versioned account
(including tensor shapes, diagrams, evidence, and known limits) is the
[v3 architecture snapshot](docs/architecture-versions/v3/README.md). The
[version register](docs/architecture-versions/README.md) records other versions
and proposals. This page is a short entry point, not a second specification.

```text
Shared encoder ordered features (B,T,128) + validity/window masks
  → harmony temporal network → masked song embedding (B,32)
  → descriptor head → 45 standardized song-level predictions (B,45) [v4 default]
  → fusion-owned Linear(45,64) → gated genre classifier
```

The v4 prediction targets are all [45 automatically extracted
track descriptors](docs/feature-contract.md). The v3 comparison option selects
12 of them and uses `Linear(12,64)`. The joint trainer fits target
standardization on training rows and uses masked Smooth L1 supervision.
Fusion consumes **predictions**, not target values or chroma probabilities.
Temporal chroma logits exist in the branch but are not supervised in the
audited v3 run or the v4 configuration; the optional chord head is disabled. The 32D embedding is an
ablation input, not the primary fusion input.

For exact target order, report metrics, checkpoint checks, and unresolved
evidence for v3, read the [integration handoff](docs/integration-handoff.md). The
[temporal chroma/chord proposal](TEMPORAL_CHROMA_PROPOSAL.md) and its
[research plan](docs/temporal-chroma-research-plan.md) are historical/proposed
experiments, **not** instructions for the current descriptor training path.
