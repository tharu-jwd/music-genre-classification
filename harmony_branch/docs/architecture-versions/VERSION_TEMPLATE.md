# Harmony vN — architecture snapshot

Status: proposed / implemented / retired. Date: YYYY-MM-DD. Owner: name.

## Identity and evidence

- Parent version and exact change:
- Source commit or PR:
- Target-table/extractor version and feature order:
- Checkpoint/report identifier and hash, if trained:
- What is code-verified, artifact-verified, team-decided, or unverified:

## Input and target contract

- Audio region, sample rate, log-Mel or waveform settings, window selection:
- Shared-encoder input/output tensor shapes and temporal resolution:
- Pseudo-label source, alignment, masks, normalization, missing-value policy:
- Exact target names/order and interpretation:

## Branch and fusion contract

- Layer-by-layer branch path, widths, activation, dropout, pooling, mask behavior:
- Active heads versus retained inactive/optional heads:
- Exact tensors handed to fusion and their semantics:
- Fusion projection ownership, mask, genre head, shortcut policy:
- Losses, weights, train-only fitting, optimizer and gradients:

## Diagrams

- Target/provenance path:
- Audio/window/encoder path:
- Harmony branch internals:
- Supervision and fusion:
- Difference from prior version:

Keep each diagram's editable `.mmd` source beside its rendered PNG, and embed
the PNG plus collapsed Mermaid source in this snapshot. Split diagrams by
component when a single one becomes difficult to read.

## Verification and decision

- Schema, shape, mask, and regression tests:
- Branch diagnostics and full-model metrics, with split/seed/budget:
- Matched comparison to parent, including genre macro AP:
- Known limitations and claims explicitly not supported:
- Decision: keep / revise / reject, with owner sign-off:
