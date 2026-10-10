# Individual paper runs: train, validate and test

Run each command from the project folder in its assigned Modal workspace. Each
command starts exactly one configuration/seed run, trains for 30 epochs,
uses validation genre macro average precision to select the best checkpoint,
then evaluates that checkpoint on test. Test results do not select checkpoints.
Training and test evaluation execute within the same remote GPU job.
Keep the terminal open until Modal confirms the GPU job has started; `--detach`
lets that remote job continue if the terminal disconnects afterward.

Every workspace must use the same project snapshot and manifest, the same
`vector-dataset.csv` and `track_split_assignments.csv` uploaded to
`music-genre-data/dataset`, and the same audio under `mtg-jamendo/logmel_songs`.
Follow steps 1–2 of [the setup guide](paper-quickstart.md) before training.
Volumes are specific to their workspace; identical names do not share data.

The three preselected seeds are 42, 123 and 456. Report all three scores and their
mean and sample standard deviation. Three repetitions provide a limited estimate
of training variability; they do not establish robustness across data splits or
strong statistical evidence for small differences. Keep the same seeds for all
configurations and avoid selecting seeds or extending experiments based on test scores.

## Seed 42

```powershell
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-cnn-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-full-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-no-supervision-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-instrument-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-timbre-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-rhythm-s42 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-harmony-s42 --execute
```

## Seed 123

```powershell
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-cnn-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-full-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-no-supervision-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-instrument-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-timbre-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-rhythm-s123 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-harmony-s123 --execute
```

## Seed 456

```powershell
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-cnn-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-full-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-no-supervision-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-instrument-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-timbre-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-rhythm-s456 --execute
modal run --detach modal_paper.py::paper --action run --phase all --run-name paper-v2-without-harmony-s456 --execute
```

## Manually collect results from every workspace

Download each completed run's folder from its workspace's `music-genre-runs`
volume. Put all 21 folders directly inside `results/modal-runs` on the machine
where you will analyze results. Preserve their names. For example:

```text
results/modal-runs/
  paper-v2-cnn-s42/
    results.json
    evaluation_test.json
    test_predictions.npz
  paper-v2-full-s42/
    results.json
    evaluation_test.json
    test_predictions.npz
  ... remaining 19 run folders ...
```

These three files per run are required; downloading the whole folder is also
fine. `results.json` contains training/validation history and settings;
`evaluation_test.json` contains final test metrics and timing;
`test_predictions.npz` contains per-track genre/concept predictions and targets.
Checkpoints and audio are not needed for local analysis. Do not flatten the files
into a single folder or add extra workspace directories between the runs root
and run folders. You may assign any commands to any teammates/workspaces.

## Next command: analyze locally

Once all 21 folders are collected, run this from the project folder:

```powershell
python scripts/analyze_paper_runs.py --manifest experiments/paper21-modal.json --runs-root results/modal-runs --split test --out-dir results/paper-analysis
```

No Modal login, GPU or audio cache is required for this step. The local Python
environment needs numpy, pandas, scikit-learn, torch and matplotlib. If missing:

```powershell
python -m pip install numpy pandas scikit-learn torch matplotlib
```

The output folder contains `main_table.md` (mean and standard deviation across
three seeds), `paired_summary.csv` (model differences), `concept_summary.csv`
(concept prediction quality), per-genre/per-feature CSVs and PDF/PNG figures.
The analyzer checks matching code/data, run settings and prediction files.
Use a new analysis output directory if `results/paper-analysis` already contains
an earlier report. Do not revise settings based on test results.

