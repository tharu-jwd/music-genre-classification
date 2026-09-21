# Concept fusion stack — how to run and what is next

**Branch:** `thevindu-concept-fusion` (created from latest `main` after pull)  
**Owner:** Thevindu  
**Status:** Steps 1–2 complete on **fixtures**. No paper numbers until real branches are wired.

This is the integration layer after the four concept branches: concat + gated fusion, 87-label genre head, joint loss, metrics, thresholds, run schema, occlusion tests.

---

## 0. What was done before coding

1. `git fetch origin`
2. `git checkout main`
3. `git pull origin main` (now at `d8b3276` — restructured proposed architecture)
4. `git checkout -b thevindu-concept-fusion`

Do not mix this with Colab notebooks `07_descriptor_fusion_baseline` / `09_baseline_explainability`. Those remain **baselines**. This package is the **proposed** model.

---

## 1. Install and test (today)

From the repo root:

```bash
pip install -r requirements.txt pytest
```

Windows / PowerShell:

```powershell
cd "d:\Semester 5\Deep Neural Networks\Sem Project\dnn-project"
python -m pytest tests -q
```

Linux / macOS / Colab:

```bash
cd /path/to/dnn-project
python -m pytest tests -q
```

(`pytest.ini` sets `pythonpath = .`. The mock CLI also inserts the repo root on `sys.path`.)

Expected: all tests pass (token order, masked gates = 0, all-masked null token, sigmoid once, overfit tiny batch, AP ≠ trapezoidal PR-AUC, missing concept labels do not drop genre tracks, occlusion refuses to run if dropout was off).

---

## 2. Train the mock stack (still no real audio)

```bash
python scripts/run_concept_fusion_mock.py --fusion gated --experiment-id F-Gated --seed 0 --steps 40
python scripts/run_concept_fusion_mock.py --fusion concat --experiment-id F-Concat --seed 0 --steps 40
```

Writes (gitignored):

- `results/proposed/mock/F-Gated_seed0.pt`
- `results/proposed/mock/F-Gated_seed0.json`  (`RunRecord`: config hash, git SHA, metrics)

These JSON files are **fixture runs**. Do not put them in the paper.

Build a table from stored JSON (empty if you have not run the CLI):

```python
from pathlib import Path
from concept_fusion.tables import comparison_table, mean_std_table
print(comparison_table(Path("results/proposed/mock")))
```

---

## 3. What the code is

| Path | Role |
|---|---|
| `concept_fusion/contract.py` | 87 genres, 40 instruments, token order, dropout p, seed policy |
| `concept_fusion/types.py` | `BranchOutput` / `BranchBundle` — **rejects** bad shapes/NaNs |
| `concept_fusion/fixtures.py` | Random tensors of exact contract shapes |
| `concept_fusion/fusion.py` | Concat, **masked gated** (primary), attention |
| `concept_fusion/genre_head.py` | 128-D → 87 logits (no sigmoid) |
| `concept_fusion/dropout.py` | p=0.15, never all four |
| `concept_fusion/model.py` | Bottleneck model; shortcut only if `allow_shortcut=True` |
| `concept_fusion/joint_loss.py` | Genre BCE + masked aux losses; optional Kendall |
| `concept_fusion/metrics.py` | Macro AP vs trapezoidal PR-AUC, valid-tag count |
| `concept_fusion/thresholds.py` | Val-only per-tag F1 thresholds |
| `concept_fusion/interventions.py` | Occlusion via the same `fusion_mask` |
| `concept_fusion/run_schema.py` | Run JSON + 3-seed vs 1-seed policy |
| `docs/adr/0001-concept-fusion-architecture.md` | The 11 frozen decisions |

---

## 4. Next — step by step (your calendar)

### Step 0 (today / tomorrow) — ratify the ADR

1. Send `docs/adr/0001-concept-fusion-architecture.md` to Dehan, Anupama, Senidu, Tharupahan.
2. 90-minute meeting. Do not wait to “form opinions” — the ADR already recommends:
   - probabilities into branch projections
   - LayerNorm tokens
   - global gates
   - dropout p=0.15, learned null token
   - λ=1 then val sweep + one Kendall run
   - per-tag val F1 thresholds, support floor 10
3. Each branch owner’s **first** deliverable: a mock `BranchOutput` fixture (not a trained model).
4. Freeze 87-tag order and 40-tag instrument order with Dehan (official split-0 files).

### Step 3 — integrate as branches land

1. **Instrument first (C-I).** Wire Anupama/instrument tokens into `BranchBundle`, leave rhythm/timbre/harmony on fixtures **or** zeros with `fusion_mask=0` for unused branches.
2. Run C-I end-to-end on real split-0 IDs. That validates the whole pipeline on real data and is a required experiment.
3. Missing concept **labels** must **not** drop the track from genre loss (`tests/test_joint_loss.py` already encodes this).
4. Shared encoder + joint backprop: coordinate with Dehan. This package does not own the CNN; it consumes `(B,4,64)` tokens.

### Step 4 — experiment matrix (after all four tokens are real)

Train (Colab GPU):

| ID | Seeds | Notes |
|---|---|---|
| B1 | 0,1,2 | Compact direct CNN (Dehan protocol) |
| F-Concat | 0,1,2 | Concat fusion |
| F-Gated | 0,1,2 | **Primary** |
| C-I, C-R, C-T, C-H | 0 | Single concept (`fusion_mask` on that slot only) |
| F-Attn | 0 | Optional |
| F-Hidden, F-Shortcut | 0 | **Do not cut these** — interpretability tax |
| Leave-one-out / incremental | 0 | After gated works |

Then: occlusion faithfulness report, real compute (warmup + `torch.cuda.synchronize`), writing.

---

## 5. Collaboration checklist

- [ ] ADR ratified; written objections only
- [ ] Mock fixtures from all four branches
- [ ] Official 87 / 40 vocab files in fixed order
- [ ] Same frozen test IDs for every comparison
- [ ] Concept dropout stays on for any run that will be occluded
- [ ] No inner join on missing concept labels
- [ ] No `nan_to_num(0)` in fusion

---

## 6. What you should not do yet

- Do not report mock `macro_ap` as a result.
- Do not use notebook 09 placeholder attention.
- Do not disable dropout to improve val AP.
- Do not add `song_repr` into the primary `ConceptBottleneckModel`.
