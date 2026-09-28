# Instrument branch

Open [the standalone notebook](notebooks/03_instrument_branch.ipynb). It has no
Essentia dependency and does not import project modules. It contains the design,
annotation audit, branch, loss, training, metrics, checks, literature notes and
integration guide. See [ARCHITECTURE.md](ARCHITECTURE.md) for the concise model,
tensor, masking, loss, and fusion contracts.

All mergeable instrument-branch files live in this directory:

```text
instrument_branch/
    ARCHITECTURE.md
    README.md
    notebooks/03_instrument_branch.ipynb
    scripts/generate_instrument_branch_notebook.py
    scripts/validate_instrument_branch_notebook.py
    docs/instrument-vocabulary.json
    docs/instrument-annotation-audit.json
```

The notebook is self-contained. The scripts are development tools, and the JSON
files record the fixed vocabulary and representative-cohort annotation audit.
Downloaded data and training artifacts remain outside the mergeable files.

The primary path is `song_repr (B,128) -> Linear(128,128) -> ReLU ->
Dropout(0.1) -> Linear(128,41) -> sigmoid probabilities (B,41)`.
The branch has 21,801 trainable parameters and no fusion projection.
`window_repr (B,W,128)` is accepted for contract checking but not used by the
song-level head. A learned temporal-attention extension remains pending approval.

## Run

1. Run the notebook with its default switches to execute the synthetic contract checks.
2. Set `CFG['manifest']` to the representative cohort's `track_id`/`song_id`
   and `split` CSV. Set `CFG['genre_labels_csv']` to its six-genre label CSV,
   and `CFG['instrument_labels_csv']` to its 41-instrument label CSV. Set
   `CFG['output']` to a writable artifact directory and enable `run_audit`.
3. The instrument CSV needs all 41 named binary columns, including `ukulele`.
   Obtain Dehan's
   `shared_representations.npz`, containing string `song_ids` and
   `song_repr (N,128)`. Optional `window_repr` must be `(N,W,128)`. Supply the
   encoder provenance JSON described in the notebook, then enable `run_training`.
4. Compare validation runs in separate output directories. Freeze the experiment
   configuration before enabling `run_test`. Three seeds are configured by default.

Input representations must come from the agreed two real temporal halves, using
normalization fitted on training data. The notebook owns neither the mel loader
nor the shared encoder. Old Stage 1 64-D MIL embeddings are not compatible inputs.

## Annotation Audit

The task is the representative six-genre cohort, with 41 instrument tags.
The notebook aligns the split table, six-genre labels, and instrument labels
by normalized track ID. [The saved vocabulary](docs/instrument-vocabulary.json)
fixes output order. [The audit](docs/instrument-annotation-audit.json) records
per-split label coverage and tag support for the local cohort.

| Split | Cohort tracks | Instrument-annotated | Unknown rows |
|---|---:|---:|---:|
| Train | 5,127 | 5,127 | 0 |
| Validation | 1,099 | 1,099 | 0 |
| Test | 1,098 | 1,098 | 0 |

The notebook re-audits the actual supplied manifest and exports observed IDs.
Missing annotation rows have all-zero supervision masks. On annotated rows,
omitted tags are weak benchmark negatives under the explicit `weak_closed_world`
policy; they are not verified instrument absence. A `positive_only` audit policy
is provided, but it cannot by itself support this BCE/validation protocol.
Element-wise masks support verified positives and negatives if supplied later.

Omitted tags in a populated row remain weak negatives under the selected policy;
they are not human-verified instrument absence.

## Integration Status

`concept_values` contains 41 probabilities; `logits` is provided for BCE.
Diagnostic hidden states are detached. The branch no longer returns `fusion_token`.
Fusion may concatenate the 41 values directly, or own `Linear(41,64)` to preserve
the instrument, rhythm, timbre, harmony token stack. Fusion applies `fusion_mask`
after any projection and masks absent attention keys. The branch returns unmasked
probabilities so concept supervision is independent of instrument removal.

Joint training uses genre loss plus masked instrument loss and the other concept
losses. Gradients flow through probabilities to the branch and live encoder.
Standalone instrument pretraining remains optional. Any fusion-owned projection
is trained with the genre objective. Thresholds affect reported binary predictions, not
the fusion bottleneck. Undefined per-tag AP/AUC are excluded with explicit
denominators; macro metrics never silently substitute zero.

Checkpoints record the v2 architecture; old 64-D-token branch checkpoints cannot
be loaded into this head. Evaluation disables dropout and preserves exact logits
on save/load.

Real shared-encoder representations are not present in this checkout. Therefore
there is no trained research checkpoint, real validation/test metric report or
genre-ablation result yet. Synthetic tests are discarded, not reported as model
performance. Joint integration, loss selection, temporal attention, concept vs.
hidden fusion, removal/instrument-only ablations and final three-seed evidence
remain research work requiring the corresponding inputs and team components.

## Development

The generator is the source of truth. Run these commands from the repository root:

```powershell
python instrument_branch/scripts/generate_instrument_branch_notebook.py
python instrument_branch/scripts/validate_instrument_branch_notebook.py
python instrument_branch/scripts/validate_instrument_branch_notebook.py --cohort-audit
```

The validator additionally needs `nbformat`. Offline checks cover the whole
synthetic training/serialization/evaluation path, three seeds, malformed split
rejection, masks, the concept bottleneck and test-input independence. The cohort
audit command reads the local split and label CSVs, writes diagnostics under
the ignored `data/instrument_audit/`, and refreshes the tracked audit JSON.
