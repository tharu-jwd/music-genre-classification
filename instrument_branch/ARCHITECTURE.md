# Instrument branch architecture

## Purpose

The instrument branch is the instrument-concept bottleneck in the proposed music
genre classifier. It receives a song-level representation from the shared audio
encoder and predicts the 40 instruments in the frozen MTG-Jamendo split-0
instrument vocabulary.

The predicted instrument probabilities are both:

- supervised using the available instrument annotations; and
- passed to concept fusion as the instrument representation.

The branch does not contain the shared CNN, a genre classifier, or a fusion
projection.

## Position in the complete model

```text
log-Mel windows
      |
      v
shared CNN encoder
      |
      | pooled_song / song_repr (B, 128)
      v
+--------------------------------------------------------+
| InstrumentBranch                                       |
|                                                        |
| Linear(128, 128) -> ReLU -> Dropout(0.1)              |
|                  -> Linear(128, 40) -> logits          |
|                  -> Sigmoid -> instrument probabilities|
+--------------------------------------------------------+
      |
      | concept_values (B, 40)
      v
fusion-owned Linear(40, 64)
      |
      v
masked concept fusion with rhythm, timbre, and harmony
      |
      v
87 genre logits
```

There is deliberately no direct path from the unrestricted 128-dimensional audio
representation to fusion through this branch. Fusion receives the 40 named
instrument concepts, preserving the concept bottleneck.

## Network definition

The current version is:

```text
song-128-hidden-128-dropout-0.1-concepts-40-v2
```

| Layer | Input shape | Output shape | Operation |
|---|---:|---:|---|
| Shared representation | `(B, 128)` | `(B, 128)` | Input from shared encoder |
| Hidden layer | `(B, 128)` | `(B, 128)` | `Linear(128,128)` |
| Activation | `(B, 128)` | `(B, 128)` | `ReLU` |
| Regularization | `(B, 128)` | `(B, 128)` | `Dropout(p=0.1)` |
| Classifier | `(B, 128)` | `(B, 40)` | `Linear(128,40)` |
| Concept activation | `(B, 40)` | `(B, 40)` | Independent sigmoid |

The branch has **21,672 trainable parameters**:

```text
Linear(128,128): 128 * 128 + 128 = 16,512
Linear(128,40):  128 * 40  + 40  =  5,160
Total:                                21,672
```

The 40 outputs are independent because a song can contain multiple instruments.
Consequently, the branch uses sigmoids rather than a softmax.

## Input contract

### Required input

```python
song_repr: FloatTensor  # shape (B, 128), finite values
```

`song_repr` is the masked, pooled song representation produced by the shared CNN
encoder. Its preprocessing, normalization, checkpoint identity, and feature order
must be identical during training and inference.

### Optional compatibility input

```python
window_repr: FloatTensor | None  # shape (B, W, 128), W >= 1
```

The current v2 branch validates `window_repr` when it is supplied but does not use
it to make predictions. It is retained for compatibility and diagnostics. Adding
temporal attention over window representations would be a new, separately tested
architecture version.

### Masks

```python
supervision_mask: Tensor | None  # shape (B, 40)
fusion_mask:      Tensor | None  # shape (B, 1)
```

- `supervision_mask[b, c]` determines whether instrument target `c` is known for
  song `b` and may contribute to the instrument loss.
- `fusion_mask[b, 0]` determines whether the already-predicted instrument concepts
  are enabled in fusion, for example during branch dropout or an ablation.
- A missing supervision label must not automatically disable the branch in fusion.
- Applying `fusion_mask` must not change `concept_values`; masking is performed at
  the fusion boundary.

## Output contract

The branch returns a dictionary with the following fields:

| Field | Shape | Meaning |
|---|---:|---|
| `concept_values` | `(B, 40)` | Sigmoid instrument probabilities sent to fusion |
| `logits` | `(B, 40)` | Raw values used for numerically stable BCE |
| `supervision_mask` | `(B, 40)` | Observed-target mask |
| `fusion_mask` | `(B, 1)` | Branch availability/dropout mask for fusion |
| `diagnostics.hidden` | `(B, 128)` | Detached hidden state for diagnostics only |

The following invariant must hold:

```python
concept_values == sigmoid(logits)
```

The branch must **not** return a `fusion_token`. In the current integration
contract, concept fusion owns `Linear(40,64)`. The detached hidden representation
is not a valid fusion input.

## Supervision and loss

Instrument recognition is a partially observed multi-label task. The branch uses
element-wise masked binary cross-entropy with logits:

```text
L_instrument = sum(mask * BCEWithLogits(logits, targets)) / sum(mask)
```

Only observed targets enter the loss. Unknown annotations must not be converted to
negative labels. If a batch contains no observed targets, the implementation
returns a differentiable zero loss.

Optional positive-class weights may be calculated from the training partition
only. They must not be derived from validation or test data.

## Fusion contract

Concept fusion transforms the instrument probabilities to the common 64-dimensional
token width:

```python
instrument_token = instrument_projection(concept_values)  # Linear(40, 64)
instrument_token = instrument_token * fusion_mask
```

The token is then normalized and combined with the rhythm, timbre, and harmony
tokens by the selected fusion mechanism. Thresholded binary instrument decisions
must not be sent to fusion; fusion consumes the continuous probabilities so genre
gradients can reach the instrument head and shared encoder.

During joint training, the relevant objective is conceptually:

```text
L_total = L_genre + lambda_instrument * L_instrument + other branch losses
```

Unless the encoder or instrument branch is deliberately frozen, genre gradients
flow through:

```text
genre head -> fusion -> instrument probabilities
           -> instrument classifier -> shared encoder
```

Instrument supervision should remain active during joint fine-tuning to reduce
concept drift.

## Training and evaluation behavior

- Training mode enables dropout in the hidden layer.
- Evaluation mode disables dropout and must produce deterministic logits for the
  same input and checkpoint.
- Thresholds are selected using validation data only and are used for reporting
  discrete instrument metrics.
- Threshold selection does not modify the continuous probabilities sent to fusion.
- The fixed 40-label order is stored in
  [`docs/instrument-vocabulary.json`](docs/instrument-vocabulary.json) and must be
  preserved by target loading, outputs, metrics, and checkpoints.

## Minimal forward-pass pseudocode

```python
def forward(song_repr, supervision_mask=None, fusion_mask=None):
    hidden = dropout(relu(hidden_linear(song_repr)))
    logits = classifier(hidden)
    probabilities = sigmoid(logits)

    return {
        "concept_values": probabilities,
        "logits": logits,
        "supervision_mask": checked_supervision_mask,
        "fusion_mask": checked_fusion_mask,
        "diagnostics": {"hidden": hidden.detach()},
    }
```

## Architectural boundaries

The following changes are outside the v2 architecture and require a new version and
matching ablation:

- sending the 128-dimensional audio representation or hidden state directly to
  fusion;
- adding a branch-owned 64-dimensional fusion token;
- replacing the sigmoid outputs with softmax;
- treating unobserved annotations as negative labels;
- thresholding probabilities before fusion;
- adding window attention or another temporal aggregation mechanism; or
- changing the frozen 40-instrument vocabulary or its order.

## Implementation source

The executable reference implementation, loss, training loop, contract checks, and
evaluation code are generated by
[`scripts/generate_instrument_branch_notebook.py`](scripts/generate_instrument_branch_notebook.py)
into [`notebooks/03_instrument_branch.ipynb`](notebooks/03_instrument_branch.ipynb).
The generator is the maintainable source when notebook code and documentation need
to be synchronized.
