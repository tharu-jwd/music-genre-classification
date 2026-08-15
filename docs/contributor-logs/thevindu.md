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

## Next (toward 23 Aug)

- [ ] Team completes §0 verification on canonical Stage 1 notebook / patched code on Drive.
- [ ] Confirm one-time Drive setup (mels + `patched_baseline_code`) is done by any member.
- [ ] Track incoming Stage 2 metrics / ablations for the 4-page Phase 2 paper.
- [ ] Qualitative explainability pass on held-out tracks once fusion checkpoint exists.
