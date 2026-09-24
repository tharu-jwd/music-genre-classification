# Shared CNN audio encoder architecture

## Role and boundary

`SharedAudioEncoder` is the only learned audio front end for the four concept
branches. Its model input is sampled log-Mel audio; labels and extracted concepts
are supervision targets, never encoder inputs.

One metadata-aware forward pass produces all branch inputs from the same sampled
audio and keeps them in one autograd graph:

```text
log-Mel windows (B,W,1,96,F)
            |
            v
shared_cnn_audio_encoder_v2
       +----+------------------------+
       |                             |
       v                             v
pooled_song (B,128)        encoded_sequence (B,T,128)
       |                             |
       +-> instrument                +-> rhythm
       +-> timbre                    +-> harmony
```

Fusion consumes only branch bottlenecks. The primary model has no unrestricted
shared-encoder-to-genre shortcut.

## Canonical input

```text
x:                    (B,W,1,96,F) floating-point log-Mel dB values
window_mask:          (B,W)        true exactly for windows with real frames
window_valid_frames:  (B,W)        number of real leading frames in each window
window_start_seconds: (B,W)        song-relative start of each sampled window
```

The frozen data schema uses 12 kHz mono audio, a 512-sample analysis frame, a
256-sample hop (approximately 21.333 ms), 96 Slaney Mel bands, 1,366 frames per
window, and at most 12 ordered, evenly selected windows.

The v1 data policy applies no additional normalization after the stored log-Mel dB
representation. Encoder v2 uses GroupNorm, whose statistics are computed per
example and are not fitted from validation/test data.

Padded mel frames are set to zero inside the encoder before normalization or
convolution. A nonzero value in a padded input position therefore cannot affect a
valid token or a pooled representation.

## CNN

Every window is reshaped onto the batch axis and processed independently:

| Stage | Operation | Channels | Frequency stride | Time stride |
|---|---|---:|---:|---:|
| 1 | `Conv2d(1,32,3,pad=1)` + `GroupNorm` + `GELU` | 32 | 1 | 1 |
| 2 | `MaxPool2d(2,2,ceil_mode=True)` | 32 | 2 | 2 |
| 3 | `Conv2d(32,64,3,pad=1)` + `GroupNorm` + `GELU` | 64 | 1 | 1 |
| 4 | `MaxPool2d((2,1),ceil_mode=True)` | 64 | 2 | 1 |
| 5 | `Conv2d(64,96,3,pad=1)` + `GroupNorm` + `GELU` | 96 | 1 | 1 |

No convolution sees two different source windows. Distant sampled windows are
concatenated only after the CNN and keep explicit window identities for downstream
gap-aware temporal branches.

## Learned frequency aggregation

The CNN does not discard the frequency axis with an early arithmetic mean. After
three convolution stages, a learned `Conv2d(96,1,1)` produces a content-dependent
score for every remaining frequency/time position. Softmax over frequency creates
weights used to aggregate the 96 feature channels at each time step:

```text
feature map (96,F',T')
    -> learned frequency scores (1,F',T')
    -> softmax over F'
    -> weighted spectral aggregation (T',96)
    -> Linear(96,128)
    -> temporal tokens (T',128)
```

This gives the harmony branch a learned spectral representation while retaining a
compact 128-dimensional temporal interface.

## Temporal geometry

There is one time downsampling operation with stride two:

```text
token stride = 2 * 256 / 12000 = 0.042666... seconds
             = approximately 42.7 ms
```

For a window with `V` valid input frames, the valid encoder-token count is
`ceil(V / 2)`.

The three kernel-3 convolutions and the time pool give a theoretical receptive
field of 12 input mel frames, approximately 256 ms. The receptive field is broader
than the token's alignment interval.

The timestamp convention is based on output stride bins, not the full receptive
field:

```text
token_start = window_start + (2 * token_index) * hop / sample_rate
token_end   = window_start + min(2 * (token_index + 1), valid_frames)
                              * hop / sample_rate
token_time  = (token_start + token_end) / 2
```

Thus normal token intervals are approximately 42.7 ms and remain below harmony's
registered 100 ms limit. Chroma alignment uses `[token_start, token_end)`.

## Named output contract

Metadata-aware `forward` and `encode_temporal` return `SharedEncoderOutput`:

| Field | Shape | Meaning |
|---|---:|---|
| `encoded_sequence` | `(B,T,128)` | Window-major, then within-window time-major tokens |
| `sequence_mask` | `(B,T)` | Valid token positions |
| `sequence_window_index` | `(B,T)` | Source-window index; `-1` at padding |
| `sequence_start_times` | `(B,T)` | Song-relative token interval starts |
| `sequence_end_times` | `(B,T)` | Song-relative token interval ends |
| `sequence_times` | `(B,T)` | Interval midpoints |
| `window_repr` | `(B,W,128)` | Masked mean of valid tokens per sampled window |
| `pooled_song` | `(B,128)` | Masked mean across every valid audio token |
| `song_repr` | `(B,128)` | Alias of `pooled_song` |
| `availability` | `(B,)` | At least one valid audio token exists |

`T = W * ceil(F / 2)` is the padded token-slot count. Actual valid-token counts are
derived separately for every window.

For an all-masked song, temporal tokens, `window_repr`, and `pooled_song` are exact
zeros and `availability` is false. Pooling denominators are clamped only to prevent
division by zero; padded values never enter a numerator.

Calling `forward(x)` without metadata remains a compatibility convenience that
assumes every frame is valid and returns only `(B,W,D)`. It must not be used for
joint training with padded windows.

## Branch wiring

| Branch | Encoder fields |
|---|---|
| Instrument | `pooled_song`; optional `window_repr` |
| Timbre | `pooled_song` |
| Rhythm | `encoded_sequence`, `sequence_mask`, `sequence_window_index` |
| Harmony | Same ordered fields; start/end times align temporal targets |

The instrument optional-window validator accepts `(B,W,128)`; it no longer assumes
exactly two windows. Its current predictions still use only `pooled_song`.

The rhythm and harmony branches separately prevent their own temporal convolutions
from crossing window boundaries. `sequence_window_index` is therefore part of the
required real integration contract even though rhythm retains an optional fallback.

## Freezing and cached features

The encoder is trainable by default. Controlled experiments can call:

```python
encoder.freeze()    # every parameter requires_grad = False
encoder.unfreeze()  # restore gradient tracking
```

Standalone branch experiments may cache `.npz` outputs. Cached tensors are
detached from the CNN and cannot update it. Results trained from caches are frozen-
encoder branch screens, not end-to-end joint training.

## Checkpoint compatibility

Encoder v2 checkpoints must contain the complete `shared_cnn_audio_encoder_v2`
state, either directly or below `enc.`/`encoder.`. The old two-convolution
`cnn.*`/`proj.*` architecture is incompatible because it lacks GroupNorm, the
third convolution, learned frequency aggregation, and the new projection input
width. Partial loading is rejected.

A real instrument-pretraining checkpoint for encoder v2 is therefore required
before the frozen real-audio harmony cache or real joint training can run.

## Verification

Automated tests cover shapes for different window/frame counts, partial and
all-masked inputs, nonzero padded-frame invariance, masked pooling, window identity,
timestamp intervals, separation across sampled-window gaps, checkpoint restore,
freeze/unfreeze, and gradients from all four branches and genre fusion into the
shared CNN.

Implementation package: [`../shared_encoder/`](../shared_encoder/)

Compatibility import: [`../scripts/shared_audio_encoder.py`](../scripts/shared_audio_encoder.py)
