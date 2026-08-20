# Thevindu Fernando — Contributor Log

## Earlier (proposal / baseline)

- Ran baseline model for the project proposal — CNN baseline experiment with the MTG-Jamendo dataset using **split-0**.
- Drafted project proposal sections: problem statement, baseline, and proposed contribution.

## 15 Aug 2026 — Phase 2 workflow kickoff (`thevindu-branch`)

- Created GitHub branch **`thevindu-branch`** for Phase 2 work through **23 Aug**.
- Authored team workflow docs:
  - `docs/phase2-workflow.md` — full plan (verify §0 → Drive setup → architecture → member tasks → timeline → DoD).
  - `docs/phase2-verification-checklist.md` — Stage 1 three-fix + proposal `.tex`/`.bib` sign-off.
  - `docs/drive-layout.md` — shared `MTG_Instrument` Drive folder conventions.
- Added Colab / pipeline scaffolds:
  - `scripts/colab/01_one_time_drive_setup.py` — Drive dirs, mel shard download hooks, baseline clone/copy.
  - `scripts/colab/02_session_bootstrap.py` — per-session local mel sync + checkpoint push helper.
  - `scripts/features/extract_rhythm_timbre.py` — Member 2 rhythm + timbre extractors (librosa).
  - `scripts/features/extract_harmony.py` — Member 3 harmony (chroma / Tonnetz).
  - `scripts/stage2/fusion_and_genre_classifier.py` — Member 1 linear + attention fusion and 87-tag genre head.
- Role going forward (Member 5): own canonical notebooks + `.tex`/`.bib`, explainability eval, Phase 2 short paper assembly as results land.

## 19 Aug 2026 — Full Kaggle staged pipeline notebooks

- Wrote `docs/kaggle-pipeline-plan.md` (download strategy, cache-as-dataset workflow, stage map).
- Generated complete `notebooks/` pipeline **00 → 09**:
  - data download (**mel shards 00–09**), preprocessing with kernel-output attach, CNN baseline, Stage 1 instrument MIL,
  - rhythm/timbre/harmony features, Stage 2 fusion+genre, ablations, explainability.
- Documented Kaggle Save Version → Add Input chain in `docs/kaggle-how-to.md`.
- Updated root `README.md` repository structure to match real folders + Kaggle plan link.
- Added `notebooks/README.md` run-order index.

## Next (toward 23 Aug)

- [ ] Upload/run `00` on Kaggle (Internet on); Save output as dataset cache.
- [ ] Run `01`→`03`, then `04`–`06` in parallel, then `07`→`09`.
- [ ] Team completes §0 verification against `03_instrument_embedding` bug-fix cells.
- [ ] Track Stage 2 metrics / ablations for the 4-page Phase 2 paper.
- [ ] Fill `09_qualitative_listening.csv` once fusion checkpoint exists.
