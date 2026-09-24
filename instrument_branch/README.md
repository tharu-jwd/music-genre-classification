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
files record the fixed vocabulary and completed official annotation audit.
Downloaded data and training artifacts remain outside the mergeable files.

The primary path is `song_repr (B,128) -> Linear(128,128) -> ReLU ->
Dropout(0.1) -> Linear(128,40) -> sigmoid probabilities (B,40)`.
The branch has 21,672 trainable parameters and no fusion projection.
`window_repr (B,W,128)` is accepted for contract checking but not used by the
song-level head. A learned temporal-attention extension remains pending approval.

## Run

1. Run the notebook with its default switches to execute the synthetic contract checks.
2. Set `CFG['manifest']` to the canonical CSV containing unique normalized
   `song_id` and official `split` columns. Set `CFG['output']` to a writable artifact
   directory, then rerun the configuration cell so `OUT` agrees. Enable `run_audit`.
3. Obtain Dehan's `shared_representations.npz`, containing string `song_ids` and
   `song_repr (N,128)`. Optional `window_repr` must be `(N,W,128)`. Supply the
   encoder provenance JSON described in the notebook, then enable `run_training`.
4. Compare validation runs in separate output directories. Freeze the experiment
   configuration before enabling `run_test`. Three seeds are configured by default.

Input representations must come from the agreed two real temporal halves, using
normalization fitted on training data. The notebook owns neither the mel loader
nor the shared encoder. Old Stage 1 64-D MIL embeddings are not compatible inputs.

## Annotation Audit

The complete official split-0 files were downloaded and checked. All instrument
partitions contain the same 40 tags. Sparse TSVs do not define a column order:
[the saved vocabulary](docs/instrument-vocabulary.json) freezes alphabetical order.
[The audit](docs/instrument-annotation-audit.json) includes per-tag counts and source
hashes. This covers the complete official genre manifest, not local audio coverage.

| Split | Genre Tracks | Instrument-Annotated Genre Tracks | Unknown Rows |
|---|---:|---:|---:|
| Train | 32,572 | 14,218 | 18,354 |
| Validation | 11,043 | 5,428 | 5,615 |
| Test | 11,479 | 5,063 | 6,416 |

The notebook re-audits the actual supplied manifest and exports observed IDs.
Missing annotation rows have all-zero supervision masks. On annotated rows,
omitted tags are weak benchmark negatives under the explicit `weak_closed_world`
policy; they are not verified instrument absence. A `positive_only` audit policy
is provided, but it cannot by itself support this BCE/validation protocol.
Element-wise masks support verified positives and negatives if supplied later.

The [official dataset documentation](https://github.com/MTG/mtg-jamendo-dataset)
describes uploader tags and category filtering; it does not establish that every
omitted tag is a verified negative.

## Integration Status

`concept_values` contains 40 probabilities; `logits` is provided for BCE.
Diagnostic hidden states are detached. The branch no longer returns `fusion_token`.
Fusion may concatenate the 40 values directly, or own `Linear(40,64)` to preserve
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
python instrument_branch/scripts/validate_instrument_branch_notebook.py --official-audit
```

The validator additionally needs `nbformat`. Offline checks cover the whole
synthetic training/serialization/evaluation path, three seeds, malformed split
rejection, masks, the concept bottleneck and test-input independence. The official
audit command downloads annotations only, caches them under the repository's ignored `data/instrument_audit/`, and
refreshes the two tracked JSON audit artifacts.
