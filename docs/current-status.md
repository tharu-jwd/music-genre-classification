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
| Four learned concept branches | Timbre branch implemented; instrument, rhythm, and harmony pending |
| Target normalization and missing-label masks | Implemented for the timbre branch |
| Joint multi-task objective | Not implemented |
| Gated concept fusion | Not implemented |
| Proposed-model training notebook | Not implemented |
| Proposed-model ablations | Not implemented |
| Gate and concept-removal evaluation | Not implemented |

The architecture diagram is a target specification, not evidence of completion.
