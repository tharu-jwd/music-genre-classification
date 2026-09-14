# Current implementation status

This file distinguishes committed code from completed experiments. A notebook existing in Git does not mean it has been executed successfully.

| Area | Status | Notes |
|---|---|---|
| Dataset download and manifest | Implemented | Default subset is shards `00–02`; expansion is manual. |
| Direct CNN baseline | Implemented | Training and split-0 metric export exist. |
| Instrument MIL embedding | Implemented | Includes masked attention and 64-d export. |
| AcousticBrainz rhythm | Implemented | Missing or invalid JSON rows are explicitly excluded and reported. |
| Timbre and harmony | Implemented | Uses audio when available; otherwise marked log-Mel approximations. |
| Linear fusion | Implemented | Trainable from notebook `07`. |
| Attention fusion | Implemented | Trainable from notebook `07`; weights can be exported by `09`. |
| End-to-end verified run | Not recorded | Committed notebooks intentionally have no outputs or checkpoints. |
| Full concept-count ablation | Not implemented | Notebook `08` currently collects existing metrics and writes a sweep plan. |
| Real model compute benchmark | Not implemented | Notebook `08` contains only a small placeholder benchmark. |
| Hyperparameter sweep execution | Not implemented | The grid is documented but not automatically trained. |
| Concept occlusion evaluation | Not implemented | Notebook `09` currently exports attention summaries only. |
| Human qualitative evaluation | Template only | The listening table requires manual annotation. |

## Known methodological risks

- Concept feature columns are not standardized before fusion. Their different numeric scales may affect training.
- Attention weights are model internals, not automatically faithful explanations.
- Timbre/harmony log-Mel approximations are weaker than waveform-derived descriptors.
- The subset contains only the configured shards and is not the complete MTG-Jamendo collection.

These items are the next work, not completed claims.
