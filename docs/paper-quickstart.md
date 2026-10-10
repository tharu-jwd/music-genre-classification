# Simple guide: run the paper experiments

For a single command per run that includes training, validation and final test
evaluation, see [all 21 individual commands and manual collection instructions](paper-individual-runs.md).

Run these commands in PowerShell from the project folder. The tools are prepared
and tested; no real-data Modal training or pilot has been submitted by preparation.
The existing `mtg-jamendo` audio volume is used. Your Modal account must already be
set up. The first command checks the existing audio volume and creates missing
input/results volumes; it leaves existing volumes intact.

## 1. Upload the two small input tables

```powershell
python scripts/prepare_modal_volumes.py
modal volume put music-genre-data data/vector-dataset.csv /dataset/vector-dataset.csv
modal volume put music-genre-data data/track_split_assignments.csv /dataset/track_split_assignments.csv
```

The first table contains the labels/descriptors for each track. The second says
which tracks belong to training, validation or test. Audio is already in your
`mtg-jamendo/logmel_songs` volume; do not upload it again.

Use `vector-dataset.csv`, not `vector-dataset-normalized.csv`. The normalized file
changes absolute descriptor values through vector normalization. The trainer now
uses the original values and fits scaling from training tracks only.

## 2. Run the small GPU pilot

```powershell
modal run modal_paper.py::paper --action pilot --execute
```

This checks real audio loading and estimates speed/memory using a few training
batches. It prints a report and saves `paper-pilot-v2/pilot.json` in the results
volume. Check `planning.estimated_all_21_training_gpu_hours` before starting the
full study; the estimate excludes validation, startup and retries. This and the
next two steps allocate GPUs and use your Modal budget.

## 3. Train one configuration with one seed

Assign any of the 21 individual runs to your teammates. Each command below starts
exactly one run; it does not launch other configurations or seeds.

Everyone must use the same project version, manifest, input CSVs and audio cache.
If teammates use different Modal workspaces, each workspace needs access to the
audio volume and must complete step 1. Volumes in one workspace are not
automatically available in another. With a shared workspace, upload inputs once.

Choose one run name from this table. Replace `17` with `42` or `73` for the other
seeds. A seed controls training randomness; three repetitions show stability.

| Configuration | Run name for seed 17 |
|---|---|
| Direct CNN | `paper-v2-cnn-s17` |
| All concepts | `paper-v2-full-s17` |
| Without concept supervision | `paper-v2-no-supervision-s17` |
| Without instrument branch | `paper-v2-without-instrument-s17` |
| Without timbre branch | `paper-v2-without-timbre-s17` |
| Without rhythm branch | `paper-v2-without-rhythm-s17` |
| Without harmony branch | `paper-v2-without-harmony-s17` |

Example: train only the full model with seed 17:

```powershell
$experimentRun = "paper-v2-full-s17"
modal run --detach modal_paper.py::paper --action run --phase all --run-name $experimentRun --execute
```

`--phase all` allows selection from either the main comparisons or branch
removals; `--run-name` limits execution to the single named run. Different run
names can run concurrently. Do not launch the same run twice. Rerunning a command
skips completed compatible runs; partial runs stop rather than overwrite files.
Keep the terminal open until the job starts. `--detach` lets the submitted remote
job continue if your terminal disconnects; reconnect to inspect its results.

Each run saves a folder such as `paper-v2-full-s17` in `music-genre-runs`:

- `best.pt`: the saved model selected using validation performance (a checkpoint).
- `results.json`: settings, metrics, parameter counts and training history.
- `validation_predictions.npz`: individual genre and concept predictions.

Training never uses the test split to choose the model.

## 4. Test evaluation is included automatically

The `--action run` command evaluates the validation-selected checkpoint on test
after training finishes. It loads the saved model rather than retraining. Its folder gains:

- `test_predictions.npz`: genre/concept predictions, targets, observation masks
  and track IDs; descriptor predictions are also converted back to source units.
- `evaluation_test.json`: final metrics and measured evaluation time/memory.

## 5. Download this run's ZIP and send it to the analysis owner

```powershell
python scripts/paper_handoff.py download --run-name $experimentRun --out "results/$experimentRun.zip"
```

This downloads only three files for this single run: `results.json`,
`evaluation_test.json` and `test_predictions.npz`. It preserves the run names
and excludes model checkpoints and audio, keeping the handoff small. The command
uses the teammate's currently configured Modal workspace and allocates no GPU.
The teammate sends you the ZIP, for example `paper-v2-full-s17.zip`. They can
train, evaluate and send any other assigned run independently with its own name.

`results.json` alone contains training/validation information. The test report
contains final test metrics; the prediction file enables concept analysis,
matching test tracks across models and optional bootstrap comparisons. Keep all
three files together. Checkpoints remain in Modal if you need to re-evaluate later.

## 6. Combine the ZIPs and analyze locally

Save all received run ZIPs in a local `results/handoffs` folder, then run:

```powershell
$handoffFiles = @(Get-ChildItem -LiteralPath results/handoffs -Filter *.zip | ForEach-Object { $_.FullName })
python scripts/paper_handoff.py import --archives $handoffFiles --out results/modal-runs
python scripts/analyze_paper_runs.py --manifest experiments/paper21-modal.json --runs-root results/modal-runs --split test --out-dir results/paper-analysis
```

The final analysis expects all 21 runs. Use `--allow-incomplete` for an exploratory
report while some runs are outstanding. Import new ZIPs into the same runs folder
by listing only those new ZIPs; existing runs are never overwritten. Use a fresh
analysis output folder each time you generate reports.
Analysis runs locally without a GPU. Its most useful outputs are:

| File | What it tells you |
|---|---|
| `main_table.md` | The seven models' genre performance, mean ± standard deviation across seeds |
| `paired_summary.csv` | How much the full model differs from each comparison model |
| `concept_summary.csv` | Whether the branches predict their intended concepts accurately |
| `genre_performance.pdf` | Main genre-performance chart |
| `branch_removal_heatmap.pdf` | How removing each branch changes performance for each genre |

Higher average precision/F1 is better. Lower descriptor MAE/RMSE is better.
Standard deviation describes variation between training repetitions. A positive
branch-removal difference means the full model performed better than the reduced
model; the difference includes the removed branch's capacity.

## What was changed and why

- Corrected average precision for tied predictions, so scores are consistent.
- Excluded unknown instrument labels from metrics, avoiding incorrect negatives.
- Added checkpoint evaluation, so final testing does not need another training run.
- Exported individual concept predictions and saved scaling, enabling concept
  quality analysis and recovery of descriptor values in their original units.
- Added the fixed manifest and automatic reports, avoiding manual mixing of runs.
- Reduced redundant rhythm computation while retaining its layers, parameters,
  predictions and gradients.

The same-architecture control retains all four heads but uses genre loss only.
Compare it with the supervised full model to study concept supervision at equal
capacity. Compare the full model with direct CNN for overall performance and with
the four removal models for complete-branch removal effects.

For technical details, optional bootstrap intervals, recovery rules and data
metadata requirements, see [the full workflow](paper-experiment-workflow.md).
