# Harmony branch architecture — current contract

The implemented architecture is **Harmony v3**. Its complete, versioned account
(including tensor shapes, diagrams, evidence, and known limits) is the
[v3 architecture snapshot](docs/architecture-versions/v3/README.md). The
[version register](docs/architecture-versions/README.md) records other versions
and proposals. This page is a short entry point, not a second specification.

```text
Shared encoder ordered features (B,T,128) + validity/window masks
  → harmony temporal network → masked song embedding (B,32)
  → descriptor head → 12 standardized song-level predictions (B,12)
  → fusion-owned Linear(12,64) → gated genre classifier
```

The 12 prediction targets are selected from the [45 automatically extracted
track descriptors](docs/feature-contract.md). The joint trainer fits target
standardization on training rows and uses masked Smooth L1 supervision.
Fusion consumes **predictions**, not target values or chroma probabilities.
Temporal chroma logits exist in the branch but are not supervised in the
audited run; the optional chord head is disabled. The 32D embedding is an
ablation input, not the primary fusion input.

For exact target order, report metrics, checkpoint checks, and unresolved
evidence, read the [integration handoff](docs/integration-handoff.md). The
[temporal chroma/chord proposal](TEMPORAL_CHROMA_PROPOSAL.md) and its
[research plan](docs/temporal-chroma-research-plan.md) are historical/proposed
experiments, **not** instructions for the current v3 training run.
