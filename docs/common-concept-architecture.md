# Common concept architecture v2

This is the current joint-training specification. Defaults are fixed reference
choices, not validation-selected or claimed optimal. Existing v1 checkpoints need
retraining; their hidden layers and state dictionaries differ.

## Branches

All heads receive shared 128D encoder features. Instrument and timbre use pooled
features: Linear(128,64), LayerNorm(64), GELU, Dropout(0.15), Linear(64,N).
Instrument emits 41 logits and sends sigmoid probabilities to fusion. Timbre
emits 12 standardized continuous predictions.

Rhythm and harmony process ordered tokens at hidden width 64, using two residual
Conv1d blocks with kernel 3 and dilations 1 and 2. Each block applies convolution,
GELU, dropout 0.15, residual addition and LayerNorm. Padding is excluded and
convolutions never cross sampled-window boundaries. Masked mean pooling precedes
a linear concept head. Rhythm emits 10 standardized predictions; the current
joint-training harmony schema emits 12 standardized descriptor predictions.
Both temporal branches normalize and activate their input projection with
LayerNorm and GELU, matching the pooled heads.

The harmony class retains temporal chroma/chord compatibility outputs and a
configurable embedding for separate legacy experiments. Joint descriptor training
uses a 64D embedding, a direct descriptor head, and freezes the unsupervised chroma
head. The 45D extracted harmony table is a different target schema and must not be
substituted for the 12D joint vector without updating its target contract.

| Component | Trainable parameters, excluding shared encoder |
|---|---:|
| Instrument | 11,049 |
| Timbre | 9,164 |
| Rhythm | 33,994 |
| Harmony descriptor branch | 34,124 |
| Linear genre predictor, all branches | 456 |

Within each family, these count differences come from the required output widths.
The harmony count excludes the frozen compatibility chroma head.

## Fusion and genre prediction

Concatenate the enabled branches' predicted concept values, preserving native
widths. With all current branches, C = 41 + 10 + 12 + 12 = 75.
The default predictor is Linear(C,6): six genre logits, independent sigmoid
probabilities for reporting. No raw encoder feature bypass is used.

The optional MLP is Linear(C,H), GELU, Dropout(p), Linear(H,6). It applies hidden
dropout once. H defaults to 64 and p to 0; neither is tuned automatically.
The native reference uses no concept dropout. Optional whole-branch dropout is
separate from hidden dropout, retains at least one enabled branch, and is disabled
at evaluation. The older gated fusion remains available as a separate experiment.

## Loss and selection

Genre and instrument use BCE with logits. Descriptor branches use Smooth L1 with
beta 1 on targets standardized using training-only statistics. Unknown concept
labels are masked, never treated as negatives; each branch loss is averaged over
observed cells. An entirely unobserved branch contributes zero auxiliary loss,
while its predictions can still participate in genre fusion.

The reference objective is genre loss + lambda_aux * sum(branch losses), with
lambda_aux = 1. Per-branch flags remain explicit overrides for ablations.
Equal coefficients do not guarantee equal loss scales or gradient magnitudes.

| Decision | Reason or required evidence |
|---|---|
| Concept and genre widths | Fixed by the frozen target vocabularies |
| Native concatenation | Preserves concepts without projection or gate hyperparameters |
| Linear reference predictor | Tests additive concept contributions with no hidden width |
| Width 64 and branch dropout 0.15 | Shared reference settings; require sensitivity experiments |
| Kernel 3, two layers, dilations 1/2 | Compact seven-token receptive field; compare temporal heads with pooled MLPs |
| Masked mean pooling | Parameter-free aggregation; compare attention only if justified |
| LayerNorm and GELU | Fixed normalization/activation choices; not claimed optimal |
| Smooth L1 beta 1 | Transition at one training-standard-deviation error; compare MSE if claiming robustness |
| Shared lambda_aux = 1 | Equal coefficient reference; select tradeoff using validation genre and concept metrics |
| MLP width/dropout | Compare H in 32/64/128 and p in 0/0.1/0.2, retaining the smallest competitive candidate |

Keep data splits, preprocessing, training budgets and seed sets consistent. Reserve
the test set for final evaluation. Existing optimizer settings (AdamW, learning
rate 3e-4, weight decay 1e-4, cosine schedule) are reference settings too; record
them and validate sensitivity before claiming they are optimal. Checkpoints store
the architecture version, branch configs, resolved loss weights, per-module
trainable parameter counts, run configuration and split track IDs.

## Commands

```powershell
python scripts/train_joint.py --genre-predictor linear --lambda-aux 1 --skip-test --out-dir results/v2-linear
python scripts/train_joint.py --genre-predictor mlp --genre-hidden-dim 64 --genre-dropout 0.1 --lambda-aux 0.3 --skip-test --out-dir results/v2-mlp-h64-p01-l03
```

These commands start training when invoked. No full training or hyperparameter
search is implied by implementation of the architecture.
