# Paper experiment and analysis workflow

The frozen experiment manifest is [paper21-modal.json](../experiments/paper21-modal.json).
It defines seven configurations and seeds 42, 123 and 456: direct CNN, full concept
model, identical architecture with all concept-loss coefficients zero, and four
retrained branch-removal models. There are 21 runs, nine core and twelve ablations.

For one individual run per command, use the [simple run handoff guide](paper-quickstart.md).
The launcher accepts `--phase all --run-name paper-v2-full-s42` for both training
and evaluation. Each teammate downloads that run's ZIP with `scripts/paper_handoff.py`;
the analysis owner imports the run ZIPs into one local runs folder. These
lightweight exports contain reports and test predictions, without checkpoints
or audio. Evaluation reports include prediction hashes so local analysis can
check the downloaded predictions without requiring the saved models.

This workflow prepares and executes experiments; it does not claim that defaults
are optimal. No full training matrix or real-cache Modal pilot is run merely by
preparing these files.

## Data and source requirements

The existing Modal mounts are used:

```text
music-genre-data/dataset/vector-dataset.csv
music-genre-data/dataset/track_split_assignments.csv
mtg-jamendo/logmel_songs/<suffix>/<track-id>.npy
music-genre-runs/<run-id>/
```

The user-declared audio scope is the first four minutes. The loader must use the
cache's actual sample rate, hop length and dimensions. Stacked 3D arrays require
`logmel_config.json` and `logmel_audit.csv` in the dataset directory; 2D arrays use
12 kHz/256-hop defaults only when no metadata is provided. Do not describe 1,366
frames as 15 seconds at those defaults: it is approximately 29.1 seconds.

The paper uses unnormalized descriptor values. The existing
`vector-dataset-normalized.csv` applies per-track L2 normalization to descriptor
groups, changing absolute quantities such as BPM and spectral centroid. That
transform cannot be inverted from a predicted normalized vector alone. The paper
instead fits one training-only standardizer per descriptor, allowing predictions
to be recovered in original units. The older single-run Modal entry point retains
its existing normalized-table default; the new paper entry point explicitly uses
`vector-dataset.csv`.

Upload the existing unnormalized vector table before the pilot:

```powershell
python scripts/prepare_modal_volumes.py
modal volume put music-genre-data data/vector-dataset.csv /dataset/vector-dataset.csv
modal volume put music-genre-data data/track_split_assignments.csv /dataset/track_split_assignments.csv
```

Verify concept extractor scopes against the first-four-minute input separately.
The pilot can validate audio geometry and coverage, not infer which interval an
unversioned target CSV was extracted from. Also check artist overlap/duplicate
recordings before asserting artist-independent generalization.

Freeze the source before paper runs. Checkpoints fingerprint source files and
dataset/preprocessing files; checkpoint-only evaluation rejects a different source
or dataset. Moving an unchanged dataset is supported with evaluator path overrides.
Audio arrays are not individually hashed, so preserve the same immutable cache.

The previous ranking implementation mishandled ties. These paper tools require
the new `sklearn_average_precision_masked_v2` metric version and reject earlier
checkpoints. Retraining is required for paper comparisons because checkpoint
selection under the old metric cannot be retrospectively corrected from one best
checkpoint.

## Preview and pilot

Run from the repository root. Previewing the local manifest does not allocate a GPU:

```powershell
python scripts/paper_experiments.py run --manifest experiments/paper21-modal.json --phase all
```

The printed Python commands describe the container paths; use the Modal entry point
to execute them remotely. The paper entry point is separate so existing
`modal run modal_app.py` usage is preserved.

Start with the bounded real-cache pilot:

```powershell
modal run modal_paper.py::paper --action pilot --execute --gpu A10
```

This allocates a GPU, performs two warm-up plus five measured training batches
for each of the three core models, profiles evaluation on sampled training tracks,
and writes `/paper-pilot-v2/pilot.json` in `music-genre-runs`. It never accesses test
audio, saves model checkpoints or starts full training. Its 30-minute timeout bounds
the job. Reported epoch times are extrapolations and exclude full validation,
startup, cold-cache effects and retries. Repeat with a fresh run name if needed.
The report also estimates GPU-hours for nine core and all 21 training runs, using
full-model timing as an approximation for branch removals. These are planning
estimates, not guaranteed completion times.

Follow the project's GPU budget registration procedure in
[team-standards.md](team-standards.md) before launching GPU jobs. Profile first and
confirm that the projected training budget fits the remaining time and allocation.

## Train the 21 runs

All runs use 30 epochs, batch size 1, the same optimizer settings and validation
macro AP checkpoint selection. Concept configurations use native concatenation and
linear genre prediction; the direct CNN retains its 128 -> 128 -> 6 genre MLP.
Test evaluation is disabled during training.

Start the first seed of the three core comparisons:

```powershell
modal run --detach modal_paper.py::paper --action train --phase core --seed 42 --execute
```

Then complete the core seeds and ablations:

```powershell
modal run --detach modal_paper.py::paper --action train --phase core --execute
modal run --detach modal_paper.py::paper --action train --phase ablation --execute
```

The launcher runs selected jobs sequentially. Completed compatible runs are
checked on CPU and skipped; incompatible or partial runs fail explicitly. This is
run-level continuation, not optimizer/checkpoint resume. Existing run directories
are never silently overwritten. To select one run, use:

```powershell
modal run modal_paper.py::paper --action train --phase core --run-name paper-v2-full-s123 --execute
```

Use only one active launcher for a given phase/run set; concurrent launchers can
race to create the same output directory. A failed partial run requires inspection
and an explicit recovery plan rather than automatic deletion. Changing the common
protocol requires a new manifest and new run IDs, not a selective rerun of the
winning model.

If the pilot shows sequential execution cannot fit the five-day window, submit
non-overlapping `--run-name` or `--seed` selections in separate launchers within
your Modal concurrency and compute budget. Extra GPUs reduce elapsed time but do
not reduce total GPU-hours. Do not submit overlapping selections.

## Final checkpoint evaluation

Once the protocol is locked and all required training runs complete:

```powershell
modal run --detach modal_paper.py::paper --action evaluate --phase all --split test --execute
```

This restores saved weights and training-fitted descriptor scalers; it does not
train, fit scalers or select a checkpoint using test performance. It writes
`evaluation_test.json` and `test_predictions.npz` into each run folder and skips
already completed, matching evaluations. Source and dataset hashes, split IDs,
feature order and audio metadata are checked on reload. Completed training outputs
contain validation predictions and detailed validation metrics already.

To evaluate a checkpoint locally after downloading it, provide the relocated data:

```powershell
python scripts/evaluate_checkpoint.py --checkpoint results/modal-runs/paper-v2-full-s42/best.pt --data-dir data --dataset-csv data/vector-dataset.csv --split-csv data/track_split_assignments.csv --logmel-root PATH_TO_CACHE --split test --out-dir results/local-evaluation
```

Dataset files must have identical contents. Outputs are protected against
overwriting; `--overwrite` is an explicit local evaluator option for recovering an
interrupted evaluation. Keep the original checkpoints unchanged.

## Download and analyze

Download the run volume into a fresh local directory:

```powershell
modal volume get music-genre-runs / results/modal-runs
python scripts/analyze_paper_runs.py --manifest experiments/paper21-modal.json --runs-root results/modal-runs --split test --out-dir results/paper-analysis --n-resamples 2000
```

Use a different output directory for each analysis. For preliminary validation
analysis before all runs finish, use `--split validation --allow-incomplete`; the
report explicitly marks incomplete outputs exploratory. `--no-plots` produces
tables without matplotlib. Bootstrap is optional (`--n-resamples 0`); it runs on CPU.

Analysis checks the common track/target cohort, seed/config metadata, source and
dataset fingerprints, equal capacity for full versus no-supervision, and consistent
parameter counts across seeds. Predictions are aligned by track ID, never row order.

Outputs include:

- `main_table.md`, `runs.csv`, `summary.csv`: genre AP/F1, parameter counts and seed SD.
- `per_genre.csv`, `per_genre_summary.csv`: genre-specific AP/F1 and coverage.
- `concepts.csv`, `concept_features.csv`, `concept_summary.csv`: masked instrument
  metrics, standardized descriptor MAE/RMSE/R2 and observed-cell counts, for learned
  predictions and training-only reference predictors.
- `paired_differences.csv`, `paired_summary.csv`: full versus CNN, full versus
  no-supervision and each full-versus-removal comparison, paired by seed.
- `per_genre_differences.csv`: genre-specific branch-removal differences.
- `genre_performance.pdf/png`, `branch_removal_heatmap.pdf/png`: shareable figures.
- `analysis.json`: missing runs, bootstrap intervals and interpretation notes.

The prediction NPZ contains genre probabilities/targets, track IDs and genre
names, plus `<branch>_predictions`, `_targets`, `_mask`, `_names` and
`_baseline_predictions` for enabled heads. Checkpoint-only evaluation also exports
raw-unit descriptor predictions and targets. Unknown raw targets are NaN and must
remain masked. Instrument reference probabilities use observed training prevalence;
descriptor references predict the training mean (zero in standardized space).

The bootstrap reader accepts both the old `song_ids/label_names/scores` contract
and trainer `track_ids/genre_tags/probabilities` fields. Its legacy CLI name
`macro_pr_auc` computes **average precision**, not trapezoidal PR area.

## Interpretation limits

Compute metrics for each trained seed and then summarize mean and sample SD.
The bootstrap averages per-seed metric differences after paired track resampling;
it does not evaluate an ensemble of averaged predictions. Its intervals describe
track-sampling uncertainty conditional on the fitted models, not training-seed
uncertainty. Tracks are assumed independent; artist-cluster bootstrap is not
implemented. The report makes no multiple-comparison significance claims.

F1 uses a fixed threshold of 0.5. AP excludes labels with no observed positives;
F1 excludes wholly unobserved labels. Unknown instrument cells are excluded from
all ranking/threshold metrics. Continuous errors are computed only on observed
targets. Coverage and per-feature metrics should accompany aggregate results.

Branch-removal comparisons include capacity changes. Automatically extracted
descriptors are reference targets, not human ground truth. Unsupervised branch
outputs should not be assumed to represent their named concepts. Raw classifier
weights are not causal importance estimates.
Total training losses are also not comparable across different auxiliary loss
weights; compare genre metrics or the separate genre-loss term instead.
