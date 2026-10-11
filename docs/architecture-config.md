# Architecture settings

Edit [configs/architecture.json](../configs/architecture.json), then check it:

```powershell
python scripts/check_architecture.py
```

The checker validates the settings and prints trainable parameter counts. It
does not allocate a GPU, read audio or train a model. The supplied values reproduce
the current paper architecture. The existing 21 Modal commands automatically use
this file; no command changes are needed. All teammates must use the same file.

| Section | What you can change |
|---|---|
| `shared_encoder` | Three channel widths, convolution kernel, convolution bias, GroupNorm group count, activation |
| `instrument`, `timbre` | Hidden width, activation and dropout |
| `rhythm`, `harmony` | Hidden width, number of temporal blocks, kernel size, dilation base, activation and dropout |
| `harmony` | Song embedding width |
| `genre_predictor` | Linear/MLP predictor, MLP hidden width/activation/dropout and whole-concept dropout |
| `direct_cnn` | Baseline hidden width, activation and dropout |

For example, change `instrument.dropout` from `0.15` to `0.10`, or set
`genre_predictor.predictor` to `mlp` to use a nonlinear genre head. Its hidden
width, activation and dropout then apply; those settings have no effect with
the linear predictor. Temporal dilation is `dilation_base ** layer_index`:
base 2 and two blocks produce dilations 1 and 2.

Activations supported by this config are `gelu`, `relu` and `silu`. Encoder
channels must be divisible by the GroupNorm group count. Temporal kernels must
be odd and at least 3. A rhythm hidden width different from 64 adds a projection
back to the existing 64-dimensional diagnostic embedding. An instrument hidden
width different from 64 omits the optional legacy hidden-feature diagnostic;
its instrument predictions and their fusion are unaffected.

The file also records fixed contracts. Shared output/input width 128, temporal
pooling geometry, masked mean pooling and concept output counts 41/12/10/12 must
remain valid. Six genre outputs are fixed by the labels. The chroma head stays
frozen and the chord head stays disabled because these targets are unavailable.
Changing these fields requires changes to the corresponding data/model contracts,
so the validator rejects them rather than ignoring them. PyTorch normalization
epsilon, bias/initialization defaults and residual block ordering remain code
defaults; this file controls the exposed architecture settings, not arbitrary
network rewrites.

Epochs, learning rate, batch size, weight decay, seeds, loss weights and window
sampling remain training/protocol settings in `experiments/paper21-modal.json`.
Loss choices (BCE and masked Smooth L1) remain fixed implementations.

For an alternative local architecture file:

```powershell
python scripts/check_architecture.py --config configs/my-architecture.json
python scripts/train_joint.py --architecture-config configs/my-architecture.json --out-dir results/my-run
```

Explicit fusion/genre CLI flags override that part of the local JSON. To use an
alternative file for the Modal paper commands, change the manifest's
`architecture_config` path. Configured files are shipped from `configs` to Modal.

Every checkpoint/result saves the resolved architecture. Evaluation reconstructs
from this saved snapshot, so changing a JSON file later does not alter a trained
model. Paper runs use the same resolved config; the analyzer rejects mismatches
with the current manifest/config. Keep the config used for your completed study
when analyzing it. Existing completed run folders are not overwritten: use new
run names/a separate manifest for a different study rather than mixing settings.
