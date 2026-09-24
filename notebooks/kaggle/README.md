# Kaggle baseline workflow

Kaggle does not preserve `/kaggle/working` between notebooks. Save every successful
stage with output enabled and attach it to dependent notebooks using **Add Input →
Notebook Output**. Training notebooks `02`, `03`, and `07` may use a GPU only after
their run is approved.

## Intended dependencies

```text
00 → 01
01 → 02 and 03
01 → 04, 05 and 06
01 + 03 + 04 + 05 + 06 → 07
02 + 07 → 08
01 + 03 + 04 + 05 + 06 + 07 → 09
```

1. Run `00_kaggle_data_download.ipynb` with Internet enabled and save its output.
2. Attach that output to `01_preprocessing.ipynb` and save the manifest output.
3. Run `02_direct_cnn_baseline.ipynb` and `03_instrument_pretraining.ipynb` with a
   GPU.
4. Run target notebooks `04` and `05`, then use `06` to audit waveform availability.
5. Run `07_descriptor_fusion_baseline.ipynb` only after the harmony quality gates
   produce a reviewed `global_tonal_summary_v1` artifact.
6. Run notebooks `08` and `09` against the saved baseline artifacts.

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
JSON, frozen cohort artifact, and matching budget ledger as a private Kaggle dataset
or prior notebook output, attach it, and set:

```python
os.environ["GPU_RUN_RECORD"] = "/kaggle/input/<approval-dataset>/approved.json"
```

Relative cohort paths in the JSON are resolved from the JSON file's directory. A
CPU preflight needs no record; CUDA training refuses to start without a valid record
for that notebook's job. See the [training contract](../../docs/team-standards.md#6-training-contract).

This is the intended baseline workflow, not proof that the current pipeline is ready.
Known parsing, vocabulary, windowing, missing-label, and descriptor problems are
tracked in the [project plan](../../docs/project-plan.md). Notebook 06 audits only
waveform availability by default. Its CPU harmony stages require explicit opt-in
flags, a unique run name, an exact Git commit, and—for cached branch screening—the
attached notebook 03 checkpoint path. CUDA is forced off. See the
[harmony plan](../../docs/harmony-plan.md). Joint proposed-model training is not yet
part of this workflow.
