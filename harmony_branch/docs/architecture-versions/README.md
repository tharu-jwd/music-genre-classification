# Harmony architecture versions

This directory is the version register for the **harmony branch's architecture**.
The current implementation calls itself **Harmony v4** in
[`concept_fusion/contract.py`](../../../concept_fusion/contract.py) (git branch
`harmony-v2`). The [v4 snapshot](v4/README.md) is the source of truth for the
45-descriptor chroma-grounded model; the [v3 snapshot](v3/README.md) remains the
record of the 12-descriptor run and its evidence. The existing
[`ARCHITECTURE.md`](../../ARCHITECTURE.md) is the short current entry point.
The older [temporal proposal](../../TEMPORAL_CHROMA_PROPOSAL.md) is retained
separately; it is not the v3 specification.

| Record | Status | What it means | Authority |
|---|---|---|---|
| [v4](v4/README.md) | Implemented 2026-09-30; not yet trained on real audio | All 45 descriptors; exact chroma/Tonnetz means from token chroma; statistics pooling; Linear(45,64) to fusion | Code, tests, and the v4 snapshot |
| [v3](v3/README.md) | Superseded by v4; documentation snapshot dated 2026-09-28 | 45 extracted song descriptors; 12 selected standardized descriptor predictions to fusion | Current code contract, trainer, supplied checkpoint and run report |
| Harmony v1 in [ADR 0001](../../../docs/adr/0001-concept-fusion-architecture.md) | Historical contract | Temporal chroma/chords and embedding-oriented fusion | Historical ADR only; do not apply to current run |
| Older 18-value extraction prototype | Historical, not current data provenance | STFT/mel-band-derived summaries in older branches | Git history; not the 45-column CQT table |
| `temporal_chroma_v1` / chord route | Proposed experiment, not v3 training | Aligned per-token chroma and optional chord teacher | [Temporal research plan](../temporal-chroma-research-plan.md) |

Do not invent a Harmony v2 implementation from the gaps between old documents.
If a prior v2 artifact is recovered, add a separate evidence-backed entry with
its actual code and output contract.

## How to record the next change

Use the [snapshot template](VERSION_TEMPLATE.md) and create a new version
folder before changing any of these contracts: target
definition or order; extraction representation/settings; input shape or temporal
resolution; masking or pooling; branch layers/heads; supervised loss; fusion
values/projection/mask; or checkpoint schema. State whether it is implemented,
proposed, or only historical. Record the source commit, dataset/target version,
checkpoint/report identity, tensor shapes, tests, and a specific difference from
the preceding version. Keep the old snapshot intact after it is committed and
reviewed. A documentation correction that does not change behavior can be noted
in the existing snapshot without claiming a new architecture version.

The v3 snapshot distinguishes **code-verified**, **artifact-verified**,
**team-decided**, and **not proven** facts. In particular, the supplied checkpoint
does not reveal the code commit or exact harmony loss weight used for that run.
