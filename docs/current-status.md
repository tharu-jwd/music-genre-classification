# Current status

## Implemented and retained

| Component | Status |
|---|---|
| Dataset download and official split manifest | Implemented |
| Direct CNN genre baseline | Implemented |
| Instrument MIL pretraining | Implemented |
| Rhythm supervision-target extraction | Implemented |
| Timbre supervision-target extraction | Implemented |
| Harmony supervision-target extraction | Implemented |
| Descriptor-fusion baseline | Implemented |
| Baseline metric collection | Implemented |
| Descriptor-fusion attention inspection | Implemented |

Committed notebooks intentionally contain no execution outputs. Therefore, an end-to-end successful run is not proven by the repository alone.

## Proposed model

| Component | Status |
|---|---|
| Shared CNN and masked song pooling | Design agreed; reusable code exists in instrument pretraining |
| Four learned concept branches | Instrument v2 implemented (`instrument_branch/`); rhythm/timbre/harmony not yet |
| Target normalization and missing-label masks | **Implemented on fixtures** (`supervision_mask` vs `fusion_mask`; NaN ≠ zero) |
| Joint multi-task objective | **Implemented on fixtures** (element-level masks; missing labels do not drop tracks) |
| Gated concept fusion | **Implemented on fixtures** (`concept_fusion/`, branch `thevindu-concept-fusion`) |
| Proposed-model training notebook | Not implemented (one-shot runner: `scripts/run_all_fusion.py`) |
| Proposed-model ablations | **Runnable on fixtures** (concat/gated/attn, C-*, leave-one-out, incremental, hidden, shortcut, no-aux, Kendall) |
| Gate and concept-removal evaluation | **Implemented on fixtures** (`concept_fusion/interventions.py`) |

The architecture diagram is a target specification, not evidence of completion.
