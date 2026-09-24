# Harmony branch architecture

## Purpose

The harmony branch learns tonal content and change over time from the shared CNN's
ordered audio features. It predicts a 12-bin chroma distribution at every valid
encoder token and produces a configurable song-level harmony embedding for genre
fusion. An optional temporal chord-classification head can be enabled only when an
accepted chord teacher is available.

The project uses the name **harmony branch**. “Harmonic branch” refers to the same
component, but `harmony` is the canonical name in code, checkpoints, fusion, and
experiment reports.

## Position in the complete model

```text
waveform -----------------> chroma/chord teacher
   |                                |
   |                                v
   |                     timestamp-aligned targets + masks
   v
log-Mel windows
   |
   v
shared CNN encoder
   |
   | encoded_sequence (B,T,128)
   | sequence_mask (B,T)
   | sequence_window_index (B,T)
   v
+------------------------------------------------------------------+
| TemporalHarmonyBranch                                            |
|                                                                  |
| Linear(128,64)                                                    |
|   -> 2 gap-safe residual Conv1d(64,64,kernel=3) layers           |
|   -> per-token Linear(64,32)                                     |
|        |                              |                           |
|        |                              +-> chord logits (B,T,25)   |
|        +-> chroma logits (B,T,12)         optional               |
|        |                                                         |
|        +-> masked mean over time -> song embedding (B,32)        |
+------------------------------------------------------------------+
   |
   | embedding (B,32)
   v
fusion-owned Linear(32,64)
   |
   v
masked concept fusion with instrument, rhythm, and timbre
```

The chroma and chord predictions are used for temporal auxiliary losses. The
song-level embedding—not a pooled target or thresholded chord sequence—is the
harmony representation sent toward fusion.

## Reference configuration

The current implementation is a correctness-oriented reference baseline rather
than a claim that the architecture is optimal.

| Setting | Initial value |
|---|---:|
| Shared-encoder input width | 128 |
| Temporal hidden width | 64 |
| Temporal layers | 2 |
| Conv1d kernel size | 3 |
| Dropout | 0.1 |
| Song embedding width | 32 |
| Chroma classes | 12 |
| Optional chord classes | 25 |

The Python class accepts a configurable input and embedding width, but the common
model uses 128-dimensional shared features and begins screening with a
32-dimensional harmony embedding.

For those dimensions, the parameter counts are:

| Component | Parameters |
|---|---:|
| Input projection `Linear(128,64)` | 8,256 |
| Two temporal `Conv1d(64,64,3)` layers | 24,704 |
| Token embedding `Linear(64,32)` | 2,080 |
| Chroma head `Linear(32,12)` | 396 |
| **Chroma-only total** | **35,436** |
| Optional chord head `Linear(32,25)` | 825 |
| **Chroma + chord total** | **36,261** |

## Shared-encoder input contract

The forward pass receives:

```python
encoded_sequence:      FloatTensor  # (B, T, D), normally D=128
sequence_mask:         Tensor       # (B, T)
sequence_window_index: Tensor       # (B, T)
windows:               int
tokens_per_window:     int
```

The flattened temporal length must satisfy:

```text
T = windows * tokens_per_window
```

Valid positions in `sequence_window_index` must match the flattened window layout:

```text
[0, 0, ..., 0, 1, 1, ..., 1, ..., W-1]
```

Masked positions must use window index `-1`. This makes padding explicit and
prevents a caller from silently presenting a malformed temporal layout.

The branch consumes within-window encoder tokens, not one vector per approximately
29-second model window. Under the frozen log-Mel/shared-encoder contract, each
token spans approximately 42.7 ms.

## Timestamp alignment before the model

The branch forward method does not receive timestamps directly, but its targets
must be aligned using the exact encoder-token intervals before training:

```text
sequence_start_times  (B,T)
sequence_end_times    (B,T)
sequence_mask         (B,T)
```

`align_chroma_to_intervals` averages valid waveform-derived chroma frames inside
each encoder interval and renormalizes the resulting 12-bin distribution. Tokens
with no valid tonal frames remain zero and receive a false target mask.

The registered `temporal_chroma_v1` screening contract rejects encoder tokens
longer than 100 ms. This is an anti-coarsening validity check: a 29-second pooled
window cannot be relabeled as a temporal harmony token.

## Temporal encoder

### Input projection

Each shared-encoder token is mapped to the hidden width:

```text
(B,T,128) -> Linear(128,64) -> (B,T,64)
```

There is no activation immediately after this input projection. Invalid positions
are zeroed before temporal processing.

### Window-separated residual convolutions

The sequence is reshaped to treat every source window as an independent item:

```text
(B,T,64)
  -> (B * windows, tokens_per_window, 64)
  -> transpose for Conv1d
  -> (B * windows, 64, tokens_per_window)
```

Each of the two default temporal layers performs:

```text
update = GELU(Conv1d(64,64,kernel=3,padding=1)(context))
update = Dropout(0.1)(update)
context = (context + update) * structured_mask
```

Because each sampled window is convolved independently, a kernel cannot cross the
unobserved gap between distant windows selected from a long song. Two kernel-3
layers provide a five-token local receptive field, approximately 213 ms with the
current shared-encoder timing.

## Token bottleneck and prediction heads

The contextual hidden sequence is projected token-by-token:

```text
(B,T,64) -> Linear(64,32) -> token_embeddings (B,T,32)
```

The 32-dimensional token embedding is the common bottleneck for all harmony
outputs:

```text
token_embeddings -> Linear(32,12) -> chroma_logits (B,T,12)
token_embeddings -> Linear(32,25) -> chord_logits  (B,T,25), optional
```

No softmax is applied inside the model. The loss applies `log_softmax` to chroma
logits, and chord classification uses cross-entropy. Inference code may apply
softmax when probabilities are required.

Masked positions in both temporal heads are set to exactly zero.

## Song-level harmony embedding

The fusion embedding is a deterministic masked mean of the same per-token
embeddings used by the prediction heads:

```text
weight_t = mask_t / number_of_valid_tokens
embedding = sum(weight_t * token_embedding_t)
```

This is intentionally not an untrained or separate attention pool. Chroma and
chord supervision therefore update the representation that is pooled and sent to
fusion.

For a song with no valid encoder tokens:

- `availability` is false;
- pooling weights are all zero;
- the song embedding is exactly zero; and
- padded temporal logits are exactly zero.

## Output contract

`TemporalHarmonyBranch.forward` returns `HarmonyBranchOutput`:

| Field | Shape | Meaning |
|---|---:|---|
| `embedding` | `(B,D_harmony)` | Song-level representation sent toward fusion; initially 32D |
| `chroma_logits` | `(B,T,12)` | Temporal pitch-class logits |
| `chord_logits` | `(B,T,25)` or `None` | Optional major/minor/no-chord logits |
| `availability` | `(B,)` | Whether each song contains any valid audio token |
| `prediction_mask` | `(B,T)` | Valid encoder positions |
| `pooling_weights` | `(B,T)` | Deterministic masked-mean weights |

The 25-class chord vocabulary consists of 12 major chords, 12 minor chords, and
one no-chord class. Extensions and inversions are intentionally outside the
initial vocabulary.

## Chroma supervision

Chroma targets are waveform-derived 12-bin probability distributions in this
fixed C-first order:

```text
C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

They are reference features or pseudo-supervision, not human ground truth. A valid
target must be finite, non-negative, and sum to one.

The branch uses masked soft-target cross-entropy:

```text
valid = chroma_target_mask AND sequence_mask
L_chroma = mean(-sum(target * log_softmax(chroma_logits))) over valid tokens
```

If no valid chroma targets exist in a batch, the loss returns a differentiable
zero.

## Optional chord supervision

The chord head is enabled only after the pseudo-label teacher passes its external
benchmark and coverage gates. Hard chord labels use a separate target mask and
masked cross-entropy:

```text
valid = chord_target_mask AND sequence_mask
L_chord = CrossEntropy(chord_logits[valid], chord_labels[valid])
```

Chroma and chord masks are independent. A token may contribute to either, both, or
neither auxiliary loss.

## Fusion contract

The branch does **not** own a 64-dimensional `fusion_token`. Concept fusion owns
the projection from the configurable harmony width:

```python
harmony_token = harmony_projection(output.embedding)  # Linear(D_harmony, 64)
harmony_token = harmony_token * fusion_mask
```

The fusion adapter also computes a pooled 12-bin chroma probability vector from
the temporal logits. That value is diagnostic and satisfies the common branch
container; it is not a replacement for temporal chroma supervision and is not the
harmony fusion input.

By default, `fusion_mask` comes from audio `availability`. Missing chroma or chord
pseudo-labels mask only their matching auxiliary losses. They do not disable an
otherwise available harmony embedding in genre fusion.

## Joint objective and gradient flow

The harmony contribution to joint training is:

```text
L_total = L_genre
        + lambda_chroma * L_chroma
        + lambda_chord * L_chord       # only when teacher is accepted
        + other branch losses
```

Unless deliberately frozen, genre and auxiliary gradients propagate through the
harmony branch and shared encoder:

```text
genre loss  -> fusion projection -> pooled harmony embedding --+
                                                            |
chroma loss -> chroma head ----------------------------------+-> token bottleneck
                                                            |   -> temporal CNN
chord loss  -> optional chord head --------------------------+   -> shared CNN
```

## Architectural and scientific boundaries

The following violate or change the current reference architecture and require a
new version with matching evaluation:

- replacing the temporal sequence with one pooled 29-second window vector;
- allowing a temporal convolution to cross gaps between sampled windows;
- using adjacent Mel bands as if they were pitch classes;
- describing automatic chroma or chord estimates as human ground truth;
- replacing the temporal loss with loss on a pooled 12-bin diagnostic;
- sending pooled chroma, chord decisions, or a branch-owned 64D token to fusion;
- enabling the chord head before the teacher passes the registered quality gate;
- using one shared validity mask when chroma and chord supervision differ;
- treating missing pseudo-labels as zero-valued targets;
- adding a key/mode head without a separately defined target and loss contract; or
- changing the embedding width, temporal depth, vocabulary, or pooling mechanism
  without versioning the experiment.

## Implementation source

- Model: [`src/harmony_branch/model.py`](src/harmony_branch/model.py)
- Masked losses: [`src/harmony_branch/losses.py`](src/harmony_branch/losses.py)
- Target alignment: [`src/harmony_branch/alignment.py`](src/harmony_branch/alignment.py)
- Chroma/HPCP extraction: [`src/harmony_branch/features.py`](src/harmony_branch/features.py)
- Fusion adapter: [`../concept_fusion/harmony_adapter.py`](../concept_fusion/harmony_adapter.py)
- Frozen screening runner:
  [`scripts/screen_temporal_harmony_branch.py`](scripts/screen_temporal_harmony_branch.py)
- Integration contract tests:
  [`tests/test_harmony_fusion_contract.py`](tests/test_harmony_fusion_contract.py)
- Detailed research plan: [`docs/plan.md`](docs/plan.md)
