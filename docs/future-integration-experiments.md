# Future Full-Architecture Integration Experiments

**Status:** Proposed experimental protocol  
**Scope:** Shared encoder, concept branches, joint loss, optimization, and fusion integration  
**Public contracts:** instrument 41, rhythm 10, timbre 35, harmony 12, genre 6

## 1. Motivation

The completed 30-epoch runs show that improving final genre prediction does not
automatically improve every concept branch. The observed test results are:

| Model | Genre macro AP | Genre macro F1 | Rhythm macro R² | Timbre macro R² | Harmony macro R² | Instrument macro AP |
|---|---:|---:|---:|---:|---:|---:|
| Full Architecture V1 with Timbre V1, 30 epochs | 0.7424 | 0.6432 | **0.5806** | 0.7043 | 0.4818 | 0.2472 |
| Improved rhythm pooling | **0.7769** | 0.6956 | 0.5447 | **0.7428** | 0.5644 | 0.2646 |
| Improved timbre branch | 0.7745 | **0.7036** | 0.4670 | 0.6597 | **0.6240** | **0.2698** |

The genre objective can improve while rhythm or timbre fidelity declines. This
is evidence of multi-task interference at the shared encoder and joint-training
boundaries. The experiments below test integration changes incrementally rather
than combining several unmeasured changes in one run.

## 2. Invariants for every experiment

Unless an experiment explicitly declares otherwise, all runs must preserve:

- the same 5,127/1,099/1,098 train, validation, and test assignments;
- the same selected 7,324 tracks and log-Mel inputs;
- the same seed and deterministic settings;
- the shared encoder's 128-dimensional feature width;
- instrument, rhythm, timbre, and harmony public output widths of 41, 10, 35,
  and 12;
- predicted concepts as the primary fusion inputs, without a direct shared-audio
  shortcut;
- the current 30-epoch reference budget for promoted experiments;
- test evaluation exactly once, after model selection from validation data.

Five-epoch screens must use the learning-rate trajectory of the first five
epochs of the 30-epoch reference run. If cosine annealing is used, retain
`T_max=30` and stop after epoch 5; do not compress a complete cosine cycle into
the screen.

Every run needs a unique experiment ID, output directory, checkpoint, results
JSON, source commit, configuration snapshot, seed, wall time, and parameter
count.

## 3. Common evaluation

Every experiment reports:

1. genre macro and micro AP, macro and micro F1, and binary accuracy;
2. per-genre AP and F1;
3. instrument macro/micro AP and F1 plus per-label results;
4. rhythm, timbre, and harmony standardized MAE, RMSE, macro R², and per-feature
   results;
5. timbre metrics grouped into spectral, harmonic/noise, MFCC mean, and MFCC
   standard-deviation families;
6. train and validation loss terms and shared-encoder gradient diagnostics;
7. inference parameters, checkpoint size, and training time.

A five-epoch result is a screening result, not a final performance claim. A
candidate is promoted when it improves its intended metric materially without a
material genre regression or non-finite/unstable training. As an initial rule,
use `0.005` absolute genre macro AP and `0.02` absolute branch macro R² as
material changes. Report confidence intervals or repeated seeds before making a
final research claim.

## 4. I1 — Integration adapters

### Hypothesis

Small branch-private adapters will allow specialization while reducing pressure
on the shared 128-dimensional representation.

### Required baseline

I1 starts from the complete 30-epoch Full Architecture V1 configuration. The
timbre branch must also be Timbre V1: the original single output head, raw-then-
z-score targets, and the V1 loss/integration behavior. I1 must not use the V2
log-flatness preprocessor or any V2 timbre implementation. This isolates the
effect of integration adapters from the later timbre experiment.

### Design

Add a zero-initialized residual adapter before each branch:

```text
h_audio or temporal representation
        -> LayerNorm
        -> Linear(128, 32)
        -> GELU
        -> Dropout(0.10)
        -> Linear(32, 128)
        -> scaled residual addition
```

For instrument and timbre, adapt `pooled_song [B,128]`. For rhythm and harmony,
apply an equivalent adapter token-wise to `encoded_sequence [B,T,128]` while
preserving temporal masks and indices. Initialize the residual scale to zero so
the initial network is functionally equivalent to the current model.

### Controlled variables

- Keep all current losses, loss weights, optimizer settings, scheduler, fusion,
  and branch heads unchanged.
- Train I1 for the complete 30-epoch budget and compare it with the matched
  30-epoch Full Architecture V1 control.

### Implemented 30-epoch entry point

I1 is implemented as an explicit opt-in mode so it cannot silently alter other
training runs:

```powershell
python scripts/train_joint.py --experiment-i1-control --epochs 30 `
  --out-dir results/i1-v1-control-30epoch

python scripts/train_joint.py --experiment-i1 --epochs 30 `
  --out-dir results/i1-adapter-30epoch
```

The flag jointly enforces the four documented adapters and Timbre V1
raw-then-z-score preprocessing. Adapter parameters and configuration are stored
in the checkpoint, while the result JSON records `experiment: I1` and
`timbre_preprocessing_version: timbre_v1_raw_zscore_v1`. Omitting the flag
preserves the existing default path.

Both I1 modes use the complete 30-epoch budget with `scheduler_t_max: 30`.
The control uses the same V1 preprocessing and training policy but does not
instantiate adapters. The adapter run is compared with the existing 30-epoch
Full Architecture V1 result; rerunning the control is optional when the same
data split, seed, source revision, and training configuration are retained.

### Promotion criteria

- improved target-branch validation metric;
- no genre macro AP reduction greater than 0.005;
- no material regression in two or more other branches;
- stable adapter scale and finite gradients.

## 5. I2 — Loss correction

### Hypothesis

Losses appropriate to each target type will improve rare binary concepts and
continuous descriptor fidelity without changing tensor interfaces.

### Instrument loss

Compare the current binary loss with asymmetric focal loss to reduce domination
by abundant negative labels. Record the exact positive/negative focusing factors
and probability margin. Select per-label classification thresholds using only
the validation split; AP remains the threshold-independent selection metric.

### Continuous branch losses

For rhythm, timbre, and harmony, compute masked Smooth L1 independently per
feature and average features rather than raw observed cells. Add a small
concordance term:

```text
L_branch = mean_feature(SmoothL1) + lambda_ccc * mean_feature(1 - CCC)
```

Screen `lambda_ccc` at 0.10 before considering 0.20. CCC is calculated only
where a feature has sufficient valid samples and non-zero variance.

For timbre, also compare descriptor-family balancing:

```text
L_timbre = 1/4 * (L_spectral + L_harmonic + L_mfcc_mean + L_mfcc_std)
```

This prevents the 26 MFCC values from determining most of the timbre loss.

### Controlled variables

- Do not add adapters, PCGrad, new pooling, or optimizer changes in the first I2
  comparison.
- Screen each loss amendment independently before combining them.

### Promotion criteria

- instrument: higher macro AP or macro F1 without lower genre macro AP;
- continuous branches: lower RMSE and higher macro R², including improvement in
  the intended feature family;
- no branch improves only because missing targets were included incorrectly.

## 6. I3 — Shared-gradient conflict

### Hypothesis

Some task gradients point in conflicting directions at the shared encoder.
Projecting conflicting gradients will reduce negative transfer while leaving
branch-private optimization unchanged.

### Design

Apply PCGrad only to shared-encoder parameters. Branch heads, branch adapters,
fusion projections, and the genre classifier retain their ordinary gradients.
Log, per batch or at a fixed sampling interval:

- pairwise task-gradient cosine similarities;
- percentage of conflicting task pairs;
- gradient norms before and after projection;
- effective task-loss weights.

I3 must be compared against the exact promoted I2 configuration. A second
experiment may test GradNorm for adaptive task magnitudes, but PCGrad and
GradNorm must first be evaluated separately.

### Promotion criteria

- reduced frequency or magnitude of destructive shared gradients;
- recovery of rhythm/timbre R² without losing more than 0.005 genre macro AP;
- no substantial increase in instability or training time.

## 7. I4 — Staged training

### Hypothesis

Establishing concept prediction before unrestricted genre optimization will
preserve bottleneck fidelity.

### Stages

1. **Concept warm-up:** train the encoder, adapters, and concept heads for 3–5
   epochs; freeze fusion and disable genre loss.
2. **Fusion fitting:** freeze the encoder and concept branches for 2–3 epochs;
   train fusion and the genre classifier on predicted concepts.
3. **Joint fine-tuning:** unfreeze the full model, use a lower shared-encoder
   learning rate, and continue with all losses and the promoted gradient-balancing
   method.

The total training budget should remain 30 epochs for comparison. Stage lengths,
learning rates, and optimizer-state handling must be stored in the run
configuration.

### Promotion criteria

- concept metrics at the final checkpoint remain close to or exceed their
  warm-up values;
- genre macro AP matches or exceeds the non-staged control;
- results are not explained solely by a larger number of optimizer steps.

## 8. I5 — Fusion robustness

### Hypothesis

Normalizing and regularizing branch integration will prevent fusion from relying
on one high-variance or temporarily strong branch.

### Design

Apply branch-specific normalization before the existing 64-dimensional fusion
projections:

```text
instrument [41] -> LayerNorm -> Linear(41,64)
rhythm     [10] -> LayerNorm -> Linear(10,64)
timbre     [35] -> LayerNorm -> Linear(35,64)
harmony    [12] -> LayerNorm -> Linear(12,64)
```

Then evaluate, separately:

1. branch dropout with probability 0.05, followed by 0.10 if stable;
2. learned branch gates constrained to be finite and inspectable;
3. a no-gate normalization-only control.

Branch dropout is active only during training. Ground-truth concepts must never
replace predictions in the primary fusion route.

### Promotion criteria

- improved or unchanged genre macro AP;
- smaller performance loss in no-branch ablations;
- gates do not collapse permanently to one branch;
- concept predictions and public branch shapes remain unchanged.

## 9. Checkpoint selection

Continue saving the best genre checkpoint, but also save diagnostic checkpoints
for the best instrument, rhythm, timbre, and harmony validation metrics. Add a
Pareto report showing genre performance against concept fidelity.

For a research model advertised as explainable, consider constrained selection:

```text
maximize validation genre macro AP
subject to rhythm/timbre/harmony macro R² not falling more than 0.03
below the declared reference
```

The constraint and reference run must be selected before test evaluation.

## 10. Execution order

| Order | Experiment | Initial screen | Full run condition |
|---:|---|---:|---|
| 0 | Full Architecture V1 with Timbre V1 control | Existing 30-epoch reference | Existing V1 30-epoch result |
| 1 | I1 residual adapters | 30 epochs | Primary I1 comparison |
| 2 | I2 branch-appropriate losses | 5 epochs each | Best isolated loss configuration |
| 3 | I3 PCGrad on shared encoder | 5 epochs | Improves negative-transfer metrics |
| 4 | I4 staged training | 30 epochs | Run only with promoted I1–I3 choices |
| 5 | I5 fusion normalization/dropout/gates | 5 epochs each | Promote one fusion change at a time |

Do not interpret five-epoch ranks as final ranks. Retrain every promoted candidate
for the complete budget, repeat with multiple seeds where feasible, and perform
the held-out test evaluation only after validation-based selection is complete.

## 11. Ownership and decoupling requirements

Each branch should expose its feature names, output dimension, preprocessing
state, loss terms, metrics, and optional optimizer-group requests through a
common adapter interface. The full trainer remains responsible for the shared
optimizer, scheduling, epoch loop, gradient-conflict method, checkpoint assembly,
and test evaluation. Fusion owns projections and gates. No branch may terminate
the joint training loop or silently change another branch's gradients.

This ownership boundary permits branch experimentation while preserving a
reproducible full-architecture integration contract.
