# Complete model architecture and training specification

This is the system-level specification for the implemented concept-guided,
multi-label music-genre classifier. It covers the shared encoder, all four concept
branches, fusion, genre prediction, target provenance, masking, losses, and the
intended training procedure.

The word **target** is deliberate. Not every target is human ground truth: genre
and instrument labels are weak uploader annotations; rhythm and timbre targets are
automatically extracted descriptors; chroma is an audio-derived reference; and
optional chord labels are teacher-generated pseudo-labels.

## 1. End-to-end system

![Predicted-concept fusion architecture](diagrams/proposed-concept-guided-architecture.svg)

```text
sampled log-Mel windows: (B, W, 1, 96, F)
                         │
                         ▼
              shared CNN audio encoder
            ┌────────────┴─────────────┐
            │                          │
  pooled song representation     ordered token sequence
        (B, 128)                    (B, T, 128)
       ┌────┴────┐                  ┌───┴────┐
       ▼         ▼                  ▼        ▼
 instrument   timbre             rhythm   harmony
 40 probs     35 values        10 values  12 pooled probs
       │         │               │          │
       └────┬────┴───────────────┬──────────┘
            ▼                    ▼
       branch-specific projections to four 64D tokens
                         │
              masked gated fusion (primary)
                         │
                  fused vector (B, 128)
                         │
                 genre head: 87 logits
```

All branches consume outputs from the **same encoder forward pass**. Instrument
and timbre need the pooled 128D representation. Rhythm and harmony additionally
need the ordered 128D sequence, validity mask, and source-window identity. Harmony
also uses token interval metadata to align offline targets. Therefore, “the CNN
output” is not one tensor: it includes pooled and temporal representations plus
layout metadata.

### Default dimensions

| Component | Input | Output used downstream | Parameters |
|---|---|---|---:|
| Shared CNN | `(B,W,1,96,F)` | sequence `(B,T,128)`, song `(B,128)` | 86,913 |
| Instrument | song `(B,128)` | 40 probabilities | 21,672 |
| Rhythm | sequence `(B,T,128)` | 10 standardized predictions | 87,883 |
| Timbre | song `(B,128)` | 35 standardized predictions | 27,299 |
| Harmony, chroma only | sequence `(B,T,128)` | 12 masked-pooled predicted chroma probabilities | 35,436 |
| Optional harmony chord head | 32D token embeddings | 25 logits/token | +825 |
| Fusion | four projected 64D tokens | 128D fused representation | variant-dependent |
| Genre head | 128D fused representation | 87 logits | 11,223 |

Counts use current defaults and exclude optional components unless stated.

## 2. Data and tensor conventions

### 2.1 Dataset split and labels

The project uses the official MTG-Jamendo split-0 train, validation, and test
partitions. Split membership must not be recomputed randomly. Extractors,
normalizers, class weights, early stopping, threshold calibration, and final
evaluation must respect this partition.

Genre prediction has 87 outputs. It is multi-label: outputs are independent logits,
not one softmax distribution.

### 2.2 Audio representation

The shared input is:

```text
x                    float tensor (B, W, 1, 96, F)
window_mask          bool/0-1 tensor (B, W)
window_valid_frames  integer tensor (B, W)
window_start_seconds float tensor (B, W)
```

The canonical preprocessing contract is:

- sample rate: 12,000 Hz;
- hop length: 256 samples, or 21.333 ms per Mel frame;
- Mel bins: 96;
- canonical window: 1,366 frames, approximately 29.14 seconds;
- maximum selected windows per track: 12;
- stored values: log-Mel magnitudes in dB;
- no second dataset-wide Mel normalization inside the encoder.

Short windows are padded, while `window_valid_frames` records the real extent.
Invalid frames are zeroed before convolution. Discontinuous sampled windows are
never convolved as though they were adjacent in the song.

### 2.3 Three masks with different meanings

| Mask | Meaning | Effect |
|---|---|---|
| Sequence/window mask | Real audio exists at this position | excludes padding from temporal modeling and pooling |
| Supervision mask | A trustworthy target exists for this item/output | includes that element in its auxiliary loss |
| Fusion mask | A branch prediction is available for the track | includes that branch token in genre fusion |

A missing descriptor target does **not** imply missing audio. Its auxiliary loss is
masked, but the branch may still participate in fusion.

## 3. Shared CNN audio encoder

Implementation: [`shared_encoder/model.py`](../shared_encoder/model.py). Detailed
geometry: [`shared-cnn-encoder-architecture.md`](shared-cnn-encoder-architecture.md).

### 3.1 Internal architecture

Each selected window is encoded independently after folding `B × W` into the
batch axis:

| Stage | Operation | Channels | Kernel/stride |
|---|---|---:|---|
| 1 | Conv2d + GroupNorm(8) + GELU | 1 → 32 | `3×3`, pad 1 |
| 2 | MaxPool2d, ceil mode | 32 | `2×2`, stride `2×2` |
| 3 | Conv2d + GroupNorm(8) + GELU | 32 → 64 | `3×3`, pad 1 |
| 4 | MaxPool2d, ceil mode | 64 | `2×1`, stride `2×1` |
| 5 | Conv2d + GroupNorm(8) + GELU | 64 → 96 | `3×3`, pad 1 |
| 6 | Conv2d frequency scorer + softmax | 96 → 1 | `1×1` |
| 7 | weighted frequency sum | 96 | frequency axis |
| 8 | Linear projection | 96 → 128 | per time token |

GroupNorm avoids dataset-fitted running statistics. Frequency attention produces a
distribution over frequency bins at every time position, removes frequency by a
weighted sum, and preserves time.

Temporal downsampling is two Mel frames, so tokens are 42.667 ms apart. The CNN's
temporal receptive field is 12 input frames, approximately 256 ms. For `F` frames:

```text
tokens_per_window = ceil(F / 2)
T                 = W * tokens_per_window
```

Invalid tail tokens are zeroed. Each valid token receives its source-window index
and start/end/centre times so later branches can distinguish adjacency from gaps.

### 3.2 Output contract

One metadata-aware encoder call returns:

```text
encoded_sequence       (B, T, 128)  temporal branch input
sequence_mask          (B, T)       valid token positions
sequence_window_index  (B, T)       source window; -1 if invalid
sequence_start_times   (B, T)       token interval starts
sequence_end_times     (B, T)       token interval ends
sequence_times         (B, T)       token centres
window_repr            (B, W, 128)  masked mean per window
pooled_song/song_repr  (B, 128)     masked mean over valid tokens
availability           (B,)         at least one valid audio token
```

The pooled representations are masked reductions of the same sequence, not another
CNN. All-masked tracks produce zeros. The encoder can be frozen for branch
screening or unfrozen for joint training. Only complete v2 encoder checkpoints are
accepted; legacy two-convolution checkpoints are shape-incompatible.

## 4. Instrument branch

Detailed document:
[`instrument_branch/ARCHITECTURE.md`](../instrument_branch/ARCHITECTURE.md).

### 4.1 Internal architecture

```text
pooled_song (B,128)
  → Linear(128,128) → ReLU → Dropout(0.10)
  → Linear(128,40) → logits → sigmoid → probabilities
```

The 40 probabilities form the primary concept bottleneck. Fusion owns
`Linear(40,64)`. The detached 128D hidden state is diagnostic and is allowed only
in the named `F-Hidden` ablation. A supplied `(B,W,128)` tensor is validated by the
standalone implementation but is not used in its current prediction path.

### 4.2 Targets, loss, and metrics

Targets are the official 40 MTG-Jamendo instrument tags in the frozen alphabetical
order stored in
[`instrument-vocabulary.json`](../instrument_branch/docs/instrument-vocabulary.json).

They are **weak human/uploader annotations**, not exhaustive acoustic-presence
labels. On a row known to have instrument annotation, an omitted tag is treated as
negative by the explicit `weak_closed_world` policy; this does not prove acoustic
absence. A row without instrument annotation gets an all-zero supervision mask.

| Split | Genre cohort | Instrument-annotated | Unknown instrument labels |
|---|---:|---:|---:|
| Train | 32,572 | 14,218 | 18,354 |
| Validation | 11,043 | 5,428 | 5,615 |
| Test | 11,479 | 5,063 | 6,416 |

Training uses element-wise BCE with logits, reduced only over the supervision mask.
Optional positive weights come from observed training rows only. Per-tag thresholds
are fitted on validation only for reporting; fusion receives continuous
probabilities, never thresholded values. Report macro/micro average precision and
ROC-AUC, excluding undefined labels from macro averages and reporting valid counts.

## 5. Rhythm branch

Detailed document: [`rhythm_branch/ARCHITECTURE.md`](../rhythm_branch/ARCHITECTURE.md).

### 5.1 Internal architecture

```text
encoded_sequence (B,T,128)
  → Linear(128,64) → LayerNorm(64) → GELU
  → residual temporal blocks with dilation 1, 2, 4
  → masked learned attention pooling
  → Linear(64,64) → LayerNorm(64) → GELU
  ├→ 64D internal rhythm embedding
  └→ Linear(64,10) descriptor predictions → Linear(10,64) in fusion
```

Each residual block is:

```text
Conv1d(64,64,kernel=5,dilation=d,padding=2d)
→ GELU → Dropout(0.15)
→ Conv1d(64,64,kernel=1) → Dropout(0.15)
→ residual add → LayerNorm(64) → token mask
```

With window identities, every selected window is convolved separately, preventing
gap crossing. The three dilations create a 29-token branch receptive field, about
1.24 seconds at encoder stride, in addition to context inside each encoder token.
Masked attention learns song-level pooling. All-masked rows return zero embeddings
and predictions without softmax NaNs. The primary model sends the ten predictions,
not the 64D internal embedding, to fusion. The embedding remains available only in
the configured `embedding_fusion` ablation.

### 5.2 Targets, loss, and metrics

The fixed `acousticbrainz_rhythm_v1` order is:

1. `bpm`
2. `beats_count`
3. `beats_loudness_mean`
4. `bpm_histogram_first_peak_bpm`
5. `bpm_histogram_first_peak_spread`
6. `bpm_histogram_first_peak_weight`
7. `onset_rate`
8. `danceability`
9. `beat_interval_mean`
10. `beat_interval_std`

These are **AcousticBrainz/Essentia-derived descriptors**, not human ground truth
and never model inputs. They supervise whether Mel-derived CNN features learn a
rhythm-aware embedding.

Each feature is standardized from observed **training-split cells only**. Missing
or invalid cells have mask 0 and are not imputed into the loss. `beats_count`
describes a whole recording and must be masked when the input is sampled windows or
an excerpt rather than the same full recording.

The loss is masked Smooth L1/Huber in standardized space. No observed cells gives
a differentiable zero. Report per-descriptor and macro MAE/RMSE; when inverse
scaling is available, also report original-unit metrics.

## 6. Timbre branch

Detailed document:
[`timbre-branch-implementation.md`](../timbre_branch/docs/timbre-branch-implementation.md).

### 6.1 Internal architecture

```text
pooled_song (B,128)
  → Linear(128,128) → LayerNorm(128) → GELU → Dropout(0.20)
  → Linear(128,64) → GELU
  → Linear(64,35) → standardized descriptor predictions
```

There is no final activation because targets are standardized real values. This is
a strict bottleneck: the 35 predictions are the auxiliary output and the only
primary timbre values exposed to fusion. Fusion owns `Linear(35,64)`; no unrestricted
`pooled_song` bypass exists.

### 6.2 Targets, loss, and metrics

The 35 fixed targets are:

- spectral shape: centroid mean/std, bandwidth mean/std, contrast mean, flatness
  mean, and rolloff mean (7);
- harmonic/noise: mean harmonic-to-noise ratio in dB and mean inharmonicity (2);
- spectral envelope: MFCC 1–13 means and MFCC 1–13 standard deviations (26).

Exact order: [`constants.py`](../timbre_branch/src/timbre_branch/constants.py).
These are **deterministically extracted signal descriptors**, not human labels.
The checked extraction artifact used up to the first four minutes of full-quality
audio and contains 7,324 successful tracks, 35 finite columns, no duplicate IDs,
and no recorded failures. Those are artifact facts, not model results.

Standardization uses training-split statistics only. Training uses masked Smooth
L1 over finite available cells. Standalone selection uses validation macro
descriptor MAE, restores the best checkpoint, and evaluates test once. Report
per-descriptor and macro MAE/RMSE plus observed-cell counts.

## 7. Harmony branch

Detailed document:
[`harmony_branch/ARCHITECTURE.md`](../harmony_branch/ARCHITECTURE.md).

### 7.1 Internal architecture

```text
encoded_sequence (B,T,128)
  → Linear(128,64)
  → reshape to (B×W,64,tokens_per_window)
  → residual Conv1d(64,64,kernel=3), GELU, Dropout(0.10)
  → residual Conv1d(64,64,kernel=3), GELU, Dropout(0.10)
  → reshape to (B,T,64)
  → Linear(64,32) per token
  ├→ Linear(32,12) chroma logits/token → softmax over pitch class
  ├→ optional Linear(32,25) chord logits/token
  └→ uniform masked mean → 32D song embedding for the embedding-fusion ablation
```

There is no activation immediately after input projection. A temporal update is
`context + dropout(GELU(conv(context)))`, followed by masking. Window-wise reshaping
prevents gap crossing. Two kernel-3 layers see five branch tokens, approximately
213 ms of token positions, in addition to encoder context.

For primary fusion, softmax is applied independently to each valid token's 12
logits, and those probabilities are averaged over valid audio tokens only. Padded
tokens never enter the mean. The resulting `(B,12)` predicted chroma vector is sent
through fusion-owned `Linear(12,64)`. An all-masked song produces a zero vector and
an unavailable fusion mask. Temporal chroma loss still consumes the unpooled logits.

The separate 32D song embedding is a uniform masked mean of token embeddings. It
is retained only for the configured `embedding_fusion` ablation, where fusion owns
`Linear(32,64)`. Optional chord predictions do not enter either primary fusion or
the predicted chroma vector.

### 7.2 Chroma reference targets

The primary target is a 12-bin pitch-class distribution per encoder token:

```text
C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

It is computed from waveform audio by the versioned CQT/HPCP extractor and aligned
to encoder token intervals. A valid row is finite, non-negative, and sums to one.
Alignment error must remain below 100 ms; the 42.667 ms token step supports this.

Chroma is an **algorithmic reference**, not human ground truth. It uses masked
soft-target cross-entropy:

```text
L_chroma(t) = -sum_c target[t,c] * log_softmax(logits[t,:])[c]
```

Only positions enabled by both prediction and target masks are reduced.

### 7.3 Optional chord pseudo-labels

MTG-Jamendo has no aligned human chord timelines. The optional chord vocabulary is
12 major roots, 12 minor roots, and `N` (no chord). An automatic teacher may supply
targets only after it is benchmarked on an existing human-annotated chord dataset
and passes the agreed quality gate. Confidence and failure masks must be preserved.

These are **pseudo-labels**, never human ground truth. Chords use masked categorical
cross-entropy. If no teacher passes, construct the branch without the chord head
and train chroma only. No key/mode head is currently implemented. The harmony term
is chroma plus the optional chord term; either becomes differentiable zero when it
has no valid positions.

## 8. Fusion and genre classifier

Implementation: [`concept_fusion/`](../concept_fusion/).

### 8.1 Token assembly

The frozen order is `[instrument, rhythm, timbre, harmony]`:

| Branch value | Fusion-owned adapter | Token |
|---|---|---|
| 40 instrument probabilities | `Linear(40,64)` | `(B,64)` |
| 10 standardized rhythm predictions | `Linear(10,64)` | `(B,64)` |
| 35 standardized timbre predictions | `Linear(35,64)` | `(B,64)` |
| 12 masked-pooled chroma probabilities | `Linear(12,64)` | `(B,64)` |

Every value is a branch prediction, making the primary route a
**predicted-concept fusion** model. This structural bottleneck improves auditability
but does not, by itself, prove perfect interpretability, causal faithfulness, or
complete concepts. Every value is multiplied by its fusion mask. Runtime validation rejects wrong
dimensions/order, NaN/Inf, non-binary masks, and attempts by instrument, timbre, or
harmony to supply a private 64D token.

The versioned `embedding_fusion` ablation preserves the former route: instrument
and timbre still use their predicted values, rhythm uses its internal 64D embedding,
and harmony uses its projected 32D song embedding. Configuration records the mode;
checkpoints record both `fusion_input_mode` and
`fusion_contract_version=predicted_concept_fusion_v1`. Missing or mismatched metadata
is rejected by the checkpointed pipeline rather than silently reinterpreted.

### 8.2 Primary masked gated fusion

Each token receives LayerNorm and a scalar `Linear(64,1)` score. Disabled scores
are masked before softmax. Gates sum to one across enabled branches; disabled gates
are exactly zero. The weighted 64D sum passes through:

```text
Linear(64,128) → ReLU → Dropout(0.10)
```

If every concept is unavailable, a learned 64D null token is used. Gates show model
routing but are not, alone, causal feature-importance evidence.

### 8.3 Alternatives and concept dropout

- **Concat:** LayerNorm tokens, zero disabled tokens, concatenate to 256D, then
  `Linear(256,128) → ReLU → Dropout(0.10)`. All-disabled output is reset to zero.
- **Attention:** single-head self-attention over four LayerNorm tokens with disabled
  keys masked, masked pooling, and the same 64→128 output block. All-disabled rows
  use a learned null token.

Training-time concept dropout independently removes each enabled branch with
probability 0.15. If this would remove all originally available branches, one is
restored. Naturally all-unavailable rows remain so. Dropout is off for validation
and test.

The primary model forbids a direct `pooled_song → genre` route. That route belongs
only to `F-Shortcut`; private hidden vectors belong only to `F-Hidden`.

### 8.4 Genre head and target

```text
fused (B,128) → Dropout(0.10) → Linear(128,87) → logits
```

Training uses BCE with logits against the official 87D multi-hot genre annotation.
Sigmoid is applied once for metrics/inference, never before BCE. Genre tags are
also weak uploader labels: zero means “not annotated under the benchmark policy,”
not guaranteed acoustic absence. Report macro/micro average precision, macro
ROC-AUC where defined, per-tag support, and the exact cohort. Threshold-dependent
metrics use validation-selected thresholds only.

## 9. Supervision taxonomy

| Output | Shape/granularity | Source | Status | Loss |
|---|---|---|---|---|
| Genre | 87/song | official genre tags | weak human/uploader annotation | BCE with logits |
| Instrument | 40/song | official instrument tags | weak human/uploader annotation | masked BCE with logits |
| Rhythm | 10/song | AcousticBrainz/Essentia | extracted target | masked Smooth L1 |
| Timbre | 35/song | deterministic audio extractor | extracted target | masked Smooth L1 |
| Chroma | 12/token | CQT/HPCP extractor | algorithmic reference | masked soft-target CE |
| Chord, optional | 25/token | validated automatic teacher | pseudo-label | masked hard-label CE |

Only the first two originate in human-provided dataset tags, and those are weak
and incomplete. The others must not be called human ground truth in reports.

## 10. Joint objective and gradients

```text
L_total = λgenre      Lgenre
        + λinstrument Linstrument
        + λrhythm     Lrhythm
        + λtimbre     Ltimbre
        + λharmony    (Lchroma + Lchord_if_enabled)
```

Defaults are 1.0, but lambdas are experiment configuration rather than scientific
constants. Optional Kendall weighting learns one log-variance for each of the five
top-level terms. Each term averages over its own observed elements, so target
density does not silently scale it. No observations yields zero for that term and
does not drop the track.

In primary end-to-end mode, genre gradients flow through the genre head, gated
fusion, all four fusion-owned projections, the instrument sigmoid, rhythm
regression head, harmony per-token chroma head and softmax/pooling operation, the
branch encoders, and the shared CNN. Concept predictions are not detached. Each
auxiliary gradient separately flows through its prediction head and branch into the
same CNN. The encoder is optimized by all unmasked active objectives unless frozen.

## 11. Training procedure

### Stage 0 — contracts and artifacts

1. Use official split-0 IDs and fixed 87-genre/40-instrument orders.
2. Produce log-Mel windows and complete window metadata.
3. Join targets by normalized track ID without changing splits.
4. Generate availability masks; never silently impute missing targets.
5. Fit rhythm/timbre scalers and class weights on training only and serialize them.
6. Fit no parameters, thresholds, or quality gates on test.

### Stage 1 — optional encoder initialization

Train the shared CNN and instrument head on observed instrument labels, or load a
complete compatible v2 instrument-pretraining checkpoint. This is initialization,
not the final genre model. Save the architecture identifier and full encoder state.

### Stage 2 — standalone branch screening

Freeze the shared encoder and cache pooled features for instrument/timbre and
ordered features plus masks/layout for rhythm/harmony. Train each branch on its own
masked objective. Optimize on train, select/stop on validation, and evaluate test
once after selection. Cached features cannot update the CNN, so they do not prove
end-to-end improvement.

### Stage 3 — joint concept-model training

For each batch:

1. run one live shared-encoder pass;
2. route `pooled_song` to instrument and timbre;
3. route the sequence, mask, and window indices to rhythm and harmony;
4. adapt outputs to the common branch contract;
5. use instrument probabilities, rhythm predictions, timbre predictions, and
   masked-pooled harmony probabilities, then project four ordered 64D tokens;
6. apply training-only concept dropout;
7. fuse tokens and produce 87 genre logits;
8. compute genre and independently masked auxiliary losses;
9. backpropagate their weighted total.

The intended final model jointly fine-tunes an unfrozen CNN. A short frozen warm-up
is allowed if recorded as training policy.

### Stage 4 — selection and evaluation

Choose epochs, loss weights, fusion, and thresholds using validation only. Restore
the selected checkpoint, then evaluate held-out test once. Comparisons must share
split, vocabularies, input-window policy, and song cohort. Save the seed, commit,
config hash, scalers, feature orders, target versions, mask policy, checkpoint, and
valid-label counts with every result.

## 12. Required comparisons and ablations

- direct CNN genre classifier;
- descriptor/concept fusion baseline;
- concat fusion;
- primary masked gated fusion;
- `F-Embedding`, the former rhythm/harmony embedding-fusion route under identical
  split, cohort, windows, budget, and evaluation procedure;
- optional attention fusion;
- no-harmony control;
- chroma-only versus chroma-plus-chords, only after teacher acceptance;
- `F-Hidden` private-representation diagnostic;
- `F-Shortcut` direct-audio diagnostic.

The last two are not the primary concept bottleneck. Finalists should use the
declared multiple seeds; fixed-subset, single-seed screening is for development.

## 13. Reproducibility contract

Every reproducible bundle should contain:

- shared encoder version/weights and every branch configuration/weight set;
- fixed tag/feature orders and schema versions;
- rhythm/timbre training-only scaler statistics and optional class weights;
- fusion type, concept-dropout rate, loss weights, and Kendall parameters;
- optimizer/scheduler state when resumable;
- epoch, seed, git commit, and configuration hash;
- validation selection metric and thresholds;
- preprocessing/window policy and split identifier;
- observed-target counts and evaluated cohort IDs.

Shape-valid synthetic tests prove interface correctness only. Synthetic losses,
random predictions, extraction coverage, and successful notebook execution are
not evidence of trained model quality.

## 14. Source-of-truth map

| Concern | Source |
|---|---|
| Shared encoder | [`shared_encoder/`](../shared_encoder/) |
| Encoder geometry | [`shared-cnn-encoder-architecture.md`](shared-cnn-encoder-architecture.md) |
| Instrument | [`instrument_branch/ARCHITECTURE.md`](../instrument_branch/ARCHITECTURE.md) |
| Rhythm | [`rhythm_branch/ARCHITECTURE.md`](../rhythm_branch/ARCHITECTURE.md) |
| Timbre | [`timbre-branch-implementation.md`](../timbre_branch/docs/timbre-branch-implementation.md) |
| Harmony | [`harmony_branch/ARCHITECTURE.md`](../harmony_branch/ARCHITECTURE.md) |
| Fusion | [`concept_fusion/`](../concept_fusion/) |
| Decisions | [`docs/adr/`](adr/) |

When code and prose diverge, tested runtime contracts are the immediate
implementation truth. Update this document in the same change that modifies a
shape, target order, mask rule, or objective.
