# Phase 2 — Pre-Work Verification Checklist (§0)

**Owner for sign-off:** Member 3 (pipeline integrity) + Member 5 (canonical notebook / paper sources)  
**Do not start Stage 2 feature wiring until this is signed.**

---

## A. Stage 1 notebook — three bug fixes

Search the **canonical** Stage 1 notebook / `patched_baseline_code` on Drive.

### A1. `best_macro_map` is updated on checkpoint save

- [ ] Find the branch that saves the best checkpoint (e.g. `if macro_map > best_macro_map`).
- [ ] Confirm `best_macro_map = macro_map` (or equivalent) runs **inside** that branch.
- [ ] Confirm it is **not** stuck comparing against a frozen `0.0` with no assignment.

**Pass criterion:** After epoch *k* with a new best val score, `best_macro_map` equals that score and the saved file is that epoch’s weights.

### A2. No test→validation split leakage

- [ ] Locate train / validation / test mask construction for `split-0`.
- [ ] Confirm validation mask uses **only** validation track IDs.
- [ ] Confirm test track IDs are **never** OR’d / unioned into the validation mask.

**Pass criterion:** Intersection of val IDs and test IDs is empty; reported “val” metrics cannot include test tracks.

### A3. `resolve_stacked_mel_path` fallback under `DRIVE_MEL_ROOT`

- [ ] Find `resolve_stacked_mel_path` (or equivalent path helper).
- [ ] Primary path resolves under the configured mel root.
- [ ] Fallback path also stays inside `DRIVE_MEL_ROOT` / `dataset/logmel_songs` — not a stale `/content/...` or wrong Drive folder.

**Pass criterion:** Missing primary file falls back to a path that still exists under the Drive mel root used in this project.

---

## B. Proposal `.tex` / `.bib`

- [ ] Matches last **verified-compiling** version.
- [ ] Body ≤ **2 pages** excluding references.
- [ ] All **7** citations resolve; no broken keys.
- [ ] Canonical copy owned by Member 5 (Overleaf or repo path agreed by team).

---

## C. Canonical ownership (nominate once)

| Role | Name | Artifact |
|---|---|---|
| Member 5 | ________________ | Stage 1/2 notebooks + `.tex`/`.bib` |
| Member 3 | ________________ | Signs §A after re-check |

Date verified: ____________  
Notes / retrain required? ____________

---

## D. If any §A item fails

1. Patch `patched_baseline_code` on Drive (not only a throwaway Colab clone).
2. Re-copy into `/content/drive/MyDrive/MTG_Instrument/patched_baseline_code`.
3. Re-train Stage 1; save best checkpoint under `checkpoints/stage1/`.
4. Re-run this checklist before Member 1 joins real instrument embeddings.
