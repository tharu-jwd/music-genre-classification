# Colab baseline workflow

Colab persists artifacts under `/content/drive/MyDrive/MTG_Instrument`. Training
notebooks `02`, `03`, and `07` may use a GPU only after their run is approved.

The intended order is:

1. `00_download_to_drive.ipynb`
2. `01_preprocessing.ipynb`
3. `02_direct_cnn_baseline.ipynb`
4. `03_instrument_pretraining.ipynb`
5. `04_rhythm_targets.ipynb`, `05_timbre_targets.ipynb`, and the waveform audit in
   `06_harmony_targets.ipynb`
6. `07_descriptor_fusion_baseline.ipynb` only after a reviewed
   `global_tonal_summary_v1` artifact exists
7. `08_baseline_evaluation.ipynb` and `09_baseline_explainability.ipynb`

Training notebooks `02`, `03`, and `07` use early stopping and a 120-minute wall-time cap
by default. Change the cap through `MAX_GPU_RUN_MINUTES` only for a documented,
approved run. Training stops between batches at 90% of the cap so validation,
export, and final runtime accounting have reserved time. The genre notebooks always save validation predictions; leave
`EVALUATE_TEST` unset during development and set `EVALUATE_TEST=1` only for selected
final evaluations. Run notebook `08` only after those final genre runs.
Window-based notebooks `02` and `03` default to `GPU_BATCH_SIZE=2`; increase it only
after their forward/backward preflight succeeds on the selected runtime.

Before enabling the GPU, reserve its hours with `scripts/manage_gpu_budget.py`, then
review the record with `scripts/validate_gpu_run_request.py`. Upload the approved
JSON, frozen cohort artifact, and matching budget ledger to Drive. In the notebook,
set for example:

```python
os.environ["GPU_RUN_RECORD"] = "/content/drive/MyDrive/MTG_Instrument/run_records/approved.json"
```

Relative cohort paths in the JSON are resolved from the JSON file's directory. A
CPU preflight needs no record; CUDA training refuses to start without a valid record
for that notebook's job. See the [training contract](../../docs/team-standards.md#6-training-contract).

These notebooks contain baseline and target-preparation code, not the proposed
four-branch model. Known upstream blockers are recorded in the
[project plan](../../docs/project-plan.md). Notebook 06 performs only waveform
availability auditing by default. Its optional harmony stages require explicit
`RUN_HARMONY_CHROMA_GATE`, `PREPARE_HARMONY_SCREEN`, or `RUN_HARMONY_SCREEN` flags,
a unique `HARMONY_RUN_NAME`, and an exact 40-character `HARMONY_CODE_COMMIT`. They
force CUDA off and follow the capped ladder in the
[harmony plan](../../docs/harmony-plan.md). The invalid Mel fallback and unvalidated
18-value extractor remain retired.
