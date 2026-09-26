# Concept fusion stack — how to run everything

**Branch:** `harmony` integration candidate

**Owners:** Thevindu (fusion) and harmony branch owner

**Status:** Fusion, genre head, temporal harmony adapter/losses, metrics, and the full experiment matrix run in **one command** on fixtures.

These numbers are **not** paper results until real branch tokens replace the fixture generator.

**Instrument v2 (on `main`):** no 64-D token. Fusion owns `Linear(40, 64)` over 40 probabilities.

**Timbre v2 (on `main`):** no 64-D token and no `h_audio` shortcut. Fusion owns `Linear(35, 64)` over Senindu's standardized `z_timbre`. Feature order is `timbre_branch/src/timbre_branch/constants.py` (`FEATURE_COLUMNS`, 35 names).

**Predicted-concept fusion v1:** instrument 40→64, rhythm 10→64, timbre 35→64,
and masked-pooled predicted chroma 12→64 are fusion-owned projections. Temporal
chroma logits and optional 25-class chords retain their masked auxiliary losses.
The former rhythm/harmony embedding route is `F-Embedding`.

---

## Run all experiments once

From the repo root:

```powershell
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_all_fusion.py --quick
```

Full seed policy (3 seeds for B1 / F-Concat / F-Gated, 1 seed elsewhere):

```powershell
python scripts/run_all_fusion.py
```

One experiment:

```powershell
python scripts/run_all_fusion.py --only F-Gated --seed 0
python scripts/run_concept_fusion_mock.py --experiment-id F-Concat --seed 0 --steps 40
```

`--quick` uses a tiny cohort (16/8/8) and few steps so the whole matrix finishes locally. The default run uses 48/24/24 tracks and 30 steps.

Writes under `results/proposed/mock/` (gitignored):

| File | What |
|---|---|
| `{ID}_seed{k}.pt` | Checkpoint + val-fitted thresholds + test song IDs |
| `{ID}_seed{k}.json` | `RunRecord` (config hash, git SHA, val/test metrics) |
| `comparison.csv` / `summary.csv` | Tables generated from JSON, not typed by hand |
| `REPORT.md` | Mean ± std plus every seed |
| `faithfulness.json` | F-Gated occlusion vs gates (dropout required) |
| `cohort_ids.json` | Same disjoint train/val/test IDs for every model |

---

## What one command trains

Same frozen fixture test IDs for every row. Thresholds are fit on **validation only**.

| ID | What |
|---|---|
| B1 | Fixture `song_repr` MLP (stand-in for Dehan's compact CNN) |
| C-I / C-R / C-T / C-H | One concept (`fusion_mask` on that slot only) |
| F-Concat | Concatenation baseline |
| F-Gated | **Primary** masked gated fusion |
| F-Embedding | Previous rhythm/harmony embedding-fusion ablation |
| F-Attn | Self-attention |
| F-Hidden | Hidden embeddings instead of concept tokens |
| F-Shortcut | Fusion + `song_repr` path |
| F-Gated-NoAux | No auxiliary concept losses |
| F-Gated-Kendall | Uncertainty weighting |
| F-Gated-NoDropout | Dropout off — **no occlusion claims** |
| F-Gated-no-* | Leave-one-out |
| F-Inc-IR / F-Inc-IRT | Incremental add |

B0 (legacy notebook CNN) is not in this runner.

---

## Code map

| Path | Role |
|---|---|
| `concept_fusion/pipeline.py` | Train / val-select / test-eval / write artifacts |
| `concept_fusion/experiments.py` | Frozen matrix |
| `concept_fusion/fixtures.py` | Cohort with disjoint seven-digit IDs |
| `concept_fusion/fusion.py` | Concat, gated, attention |
| `concept_fusion/model.py` | Bottleneck; shortcut and hidden are named flags |
| `concept_fusion/harmony_adapter.py` | Temporal harmony output to shared branch contract |
| `concept_fusion/joint_loss.py` | Genre/fixed-concept losses plus temporal chroma/chord losses |
| `docs/adr/0001-concept-fusion-architecture.md` | The 11 frozen decisions |

---

## Real joint training (fusion is already the mixer)

Predicted-concept fusion is the only mixer: official 40 / 10 / 35 / 12 → 64,
masked gated fusion, 6 scoped genre logits. The live path is:

```powershell
python scripts/train_joint.py --quick --batch-size 1
```

If `data/full_dataset.csv` exists it is used automatically. Extra `ukulele` is
ignored. Missing chroma columns mask the harmony **loss** only; predicted chroma
still enters fusion. Hosted GPU: [modal-training.md](modal-training.md).

1. Do not implement a second fusion in the trainer.
2. Do not report scoped-6 AP as official 87-tag results.
3. Do not disable dropout on F-Gated if you will report occlusion.

---

## What not to do

- Do not report fixture `macro_ap` as a result.
- Do not fit thresholds on test.
- Do not add `song_repr` to the primary model.
- Do not turn off concept dropout to chase validation AP.
