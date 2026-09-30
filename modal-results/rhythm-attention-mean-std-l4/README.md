---
library_name: pytorch
tags:
  - audio-classification
  - music-information-retrieval
  - multi-label-classification
  - concept-bottleneck-model
metrics:
  - average_precision
---

# Music Genre Concept-Bottleneck Model — Rhythm Attention + Mean/Std Pooling

This PyTorch checkpoint performs multilabel music-genre classification with a
shared log-Mel CNN and supervised instrument, rhythm, timbre, and harmony concept
branches. It is the 30-epoch `attention_mean_std` rhythm-pooling experiment.

## Rhythm architecture change

The original rhythm branch pooled its `(B,T,64)` temporal representation using
learned scalar attention only. This variant summarizes the same masked temporal
features in three ways:

1. learned attention-weighted mean `(B,64)`;
2. masked arithmetic mean `(B,64)`;
3. masked population standard deviation `(B,64)`.

The three summaries are concatenated to `(B,192)` and projected through
`Linear(192,64)` before the ten rhythm concepts are predicted. Padding and
all-masked examples remain excluded by the existing masks. The output and fusion
contracts are unchanged: the rhythm branch still predicts ten standardized
AcousticBrainz descriptors and supplies a 64-dimensional representation.

## Results

| Split | Tracks | Genre macro AP |
|---|---:|---:|
| Validation | 1,099 | 0.8081 |
| Test | 1,098 | 0.7769 |

The checkpoint was selected at epoch 28 by validation genre macro average
precision. Test genre micro AP is 0.7777, macro F1 is 0.6956, and micro F1 is
0.6968. The test rhythm branch achieved standardized MAE 0.4515, RMSE 0.6714,
and macro R² 0.5447. Complete per-tag, per-feature, and epoch metrics are in
`results.json`.

These scores should not be treated as a controlled comparison against the older
20-epoch, batch-size-4 attention-only run. A matched 30-epoch, batch-size-1
attention-only run is required for that comparison.

## Inputs and outputs

The model consumes 128-band log-Mel features from 16 kHz audio, with a hop length
of 512, 15-second windows, and at most 12 sampled windows per track. The genre
order is:

```text
classical, electronic, folk, hiphop, jazz, rock
```

The checkpoint contains the shared encoder, all four concept heads, gated fusion
model, target standardizers, label vocabularies, optimizer state, rhythm model
configuration, and preprocessing metadata.

## Loading

Use source revision `a9f5a44` of the
[project repository](https://github.com/tharu-jwd/music-genre-classification/tree/a9f5a44)
for the implementation.

```python
import torch

checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
assert checkpoint["rhythm_model_config"]["pooling_mode"] == "attention_mean_std"

# Component state dictionaries include:
# checkpoint["encoder"]
# checkpoint["instrument_head"]
# checkpoint["rhythm_head"]
# checkpoint["timbre_head"]
# checkpoint["harmony_head"]
# checkpoint["fusion_model"]
```

Instantiate the modules as in `scripts/train_joint.py`, restore each state
dictionary, call `eval()`, and apply sigmoid to the fusion logits for genre
probabilities. Because the checkpoint includes optimizer state and non-tensor
metadata, load it only from a trusted source.

## Training configuration

- Run: `rhythm-attention-mean-std-l4`
- GPU: NVIDIA L4
- Epochs: 30
- Batch size: 1
- Seed: 42
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Weight decay: 0.0001
- Scheduler: cosine annealing
- Rhythm pooling: `attention_mean_std`
- Selection metric: validation genre macro average precision
- Training/validation/test tracks: 5,127 / 1,099 / 1,098

## Intended use and limitations

This checkpoint is intended for coursework, research, reproducibility, and
rhythm-pooling ablations. It is not calibrated for production decisions. Results
come from one split and one random seed. Genre and instrument targets are
multilabel and imbalanced; rhythm, timbre, and harmony targets are derived
descriptors rather than human explanations.

## Files

- `best.pt`: complete validation-selected checkpoint
- `results.json`: epoch history and validation/test metrics
- `config.json`: machine-readable architecture and training summary
