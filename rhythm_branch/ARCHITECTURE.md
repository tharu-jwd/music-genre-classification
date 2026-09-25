# Rhythm branch architecture

## Purpose

The rhythm branch learns a song-level rhythm representation from the shared CNN's
ordered audio features. It has two outputs with different responsibilities:

- a learned internal 64-dimensional embedding; and
- ten standardized rhythm predictions used for auxiliary supervision and primary
  predicted-concept fusion.

The AcousticBrainz rhythm descriptors are training targets only. They are never
provided to the branch as model inputs.

## Position in the complete model

```text
log-Mel windows
      |
      v
shared CNN encoder
      |
      | encoded_sequence (B, T, 128)
      | sequence_mask (B, T)
      | sequence_window_index (B, T)
      v
+----------------------------------------------------------------+
| RhythmBranch                                                   |
|                                                                |
| Linear(128,64) -> LayerNorm -> GELU                            |
|      -> 3 gap-aware residual temporal Conv1d blocks            |
|      -> masked attention pooling                               |
|      -> Linear(64,64) -> LayerNorm -> GELU                     |
+----------------------------------------------------------------+
      |                                      |
      | embedding (B, 64)                    | Linear(64,10)
      | embedding-fusion ablation            v
      |                            standardized predictions (B,10)
      |                                      |             |
      v                                      v             v
embedding-fusion only                 masked Huber   fusion Linear(10,64)
```

The primary model sends the ten predicted descriptors to fusion-owned
`Linear(10,64)`. The branch's 64D embedding remains available only for the named,
versioned `embedding_fusion` ablation.

## Default configuration

The checkpoint architecture identifier is `temporal_rhythm_branch_v1`.

| Setting | Value |
|---|---:|
| Input width | 128 |
| Temporal hidden width | 64 |
| Fusion embedding width | 64 |
| Regression outputs | 10 |
| Residual temporal blocks | 3 |
| Conv1d kernel size | 5 |
| Block dilations | 1, 2, 4 |
| Dropout | 0.15 |

With the default configuration, the branch has **87,883 trainable parameters**:

| Component | Parameters |
|---|---:|
| Input projection and LayerNorm | 8,384 |
| Three residual temporal blocks | 74,496 |
| Attention scorer | 65 |
| Embedding head and LayerNorm | 4,288 |
| Regression head | 650 |
| **Total** | **87,883** |

## Input contract

```python
encoded_sequence:      FloatTensor  # (B, T, 128), finite values
sequence_mask:         Tensor       # (B, T), valid-token mask
sequence_window_index: Tensor | None  # (B, T)
```

The shared encoder supplies ordered, within-window features rather than one pooled
vector per song. `T` may vary between batches, while the final feature width must
remain 128.

For valid positions, `sequence_window_index` contains the zero-based source-window
index. Padded positions use `-1`. Supplying it is strongly recommended for the
project's full-song sampling scheme because selected windows may have large gaps
between them.

The rhythm branch does not require the shared encoder's absolute timestamps. It
requires the token order, validity mask, and window identity.

## Input projection

Every 128-dimensional shared-encoder token is projected to the temporal hidden
width:

```text
(B,T,128) -> Linear(128,64) -> LayerNorm(64) -> GELU -> (B,T,64)
```

Invalid token positions are then set to zero using `sequence_mask`.

## Gap-aware temporal encoder

The temporal encoder contains three residual blocks. Block `l` uses dilation
`2^l`, giving the default dilations 1, 2, and 4.

Each block performs:

```text
input (B,T,64)
  -> Conv1d(64,64,kernel=5,dilation=d,padding=2d)
  -> GELU
  -> Dropout(0.15)
  -> Conv1d(64,64,kernel=1)
  -> Dropout(0.15)
  -> residual addition
  -> LayerNorm(64)
  -> apply sequence mask
```

The three default blocks have a 29-token temporal receptive field. Global context
is subsequently obtained through attention pooling.

When `sequence_window_index` is supplied, each source window is encoded as an
independent masked segment. A convolution can therefore use neighboring tokens
inside a real window but cannot treat the end of one sampled window and the start
of a distant window as adjacent audio.

Removing this separation would create artificial rhythmic transitions across
unobserved gaps.

## Masked attention pooling

After temporal encoding, a scalar score is learned for each token:

```text
score_t = Linear(64,1)(hidden_t)
weight_t = softmax(score_t over valid tokens)
pooled = sum(weight_t * hidden_t)
```

Padded tokens are excluded from the softmax. For an all-masked example, the
implementation uses a temporary safe position to avoid an undefined softmax and
then zeros the resulting embedding and predictions.

Audio availability is derived only from the encoder mask:

```python
availability = sequence_mask.any(dim=1)  # (B,)
```

Target availability does not determine whether the predicted rhythm concepts are
available for fusion.

## Embedding and regression heads

The attention-pooled 64-dimensional state passes through:

```text
Linear(64,64) -> LayerNorm(64) -> GELU
```

This produces `embedding (B,64)`. A final `Linear(64,10)` produces the rhythm
predictions used by both the masked auxiliary loss and primary fusion. The
embedding itself is used by fusion only in the embedding-fusion ablation.

The predictions use no final activation because they estimate standardized
continuous values.

## Output contract

`RhythmBranch.forward` returns `RhythmBranchOutput`:

| Field | Shape | Meaning |
|---|---:|---|
| `embedding` | `(B,64)` | Internal representation; embedding-fusion ablation input |
| `predictions` | `(B,10)` | Standardized estimates; primary fusion and auxiliary-loss input |
| `availability` | `(B,)` | Whether each song has at least one valid audio token |

Unavailable examples receive zero embeddings and zero predictions.

The fusion adapter adds two external masks:

| Field | Shape | Meaning |
|---|---:|---|
| `supervision_mask` | `(B,10)` | Which target cells may enter rhythm loss |
| `fusion_mask` | `(B,1)` | Whether rhythm predictions participate in fusion |

When no explicit `fusion_mask` is supplied, it is derived from `availability`.

## Rhythm target contract

The target schema is `acousticbrainz_rhythm_v1`, with this fixed order:

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

The order is part of the checkpoint and fusion contract and must not be changed
without a new schema version.

Each target is standardized independently using means and scales fitted only on
observed cells in the official training partition. The standardizer, feature
order, schema version, input scope, and exclusions are stored in every checkpoint.

If the model sees sampled windows rather than the complete recording,
`beats_count` is masked because it is a whole-recording count and is not
interval-compatible with partial input.

## Auxiliary loss

The branch uses element-wise masked Huber/Smooth-L1 loss:

```text
L_rhythm = sum(mask * SmoothL1(predictions, standardized_targets)) / sum(mask)
```

Only observed, finite, interval-compatible targets contribute. Missing targets do
not remove a song from genre training and do not disable its rhythm predictions in
fusion. If a batch contains no observed targets, the loss returns a differentiable
zero.

## Fusion and joint training

Primary fusion projects the ten standardized predictions to the common width:

```python
rhythm_token = rhythm_projection(output.predictions)  # Linear(10,64)
rhythm_token = rhythm_token * fusion_mask
```

Fusion applies shared token normalization, concept dropout, and the selected
gating/attention mechanism. The former `output.embedding` route remains selectable
only as the versioned `embedding_fusion` ablation.

During joint training, the relevant objective is conceptually:

```text
L_total = L_genre + lambda_rhythm * L_rhythm + other branch losses
```

Unless deliberately frozen, both genre and rhythm gradients propagate through the
rhythm branch into the shared CNN:

```text
genre loss  -> fusion -> predictions -> regression head --+
rhythm loss ----------------------------> regression head --+-> temporal encoder
                                                             -> shared CNN
```

## Architectural boundaries

The following are changes to `temporal_rhythm_branch_v1` and require a new version
and matching evaluation:

- passing AcousticBrainz descriptors into the branch as inputs;
- removing either versioned fusion mode or changing the 10→64 primary projection;
- changing the fixed target order;
- fitting normalization statistics on validation or test data;
- treating separated sampled windows as continuous audio;
- allowing padding to affect convolutions or attention pooling;
- supervising `beats_count` from partial-window inputs; or
- changing the temporal depth, kernel size, dilation schedule, or embedding width.

## Implementation source

- Model: [`src/rhythm_branch/model.py`](src/rhythm_branch/model.py)
- Masked loss: [`src/rhythm_branch/losses.py`](src/rhythm_branch/losses.py)
- Target alignment and standardization:
  [`src/rhythm_branch/preprocessing.py`](src/rhythm_branch/preprocessing.py)
- Checkpoint contract: [`src/rhythm_branch/training.py`](src/rhythm_branch/training.py)
- Fusion adapter: [`../concept_fusion/rhythm_adapter.py`](../concept_fusion/rhythm_adapter.py)
- Contract tests: [`../tests/test_rhythm_branch.py`](../tests/test_rhythm_branch.py)
