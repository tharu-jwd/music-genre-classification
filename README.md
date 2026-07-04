# DNN Project — Wearable Sensor Fusion for Human Activity Recognition

Semester DNN research project. See `docs/project-guidelines.md` for the full course
brief and `docs/research-ideas.pdf` for the original 10 candidate directions this
was chosen from (idea #4: wearable sensor fusion).

## 1. Dataset

**PAMAP2** (Physical Activity Monitoring), downloaded from the link in
`docs/initial-datasets.pdf`. Lives at `data/pamap2+physical+activity+monitoring/` locally
(gitignored — 1.8GB, not pushed to GitHub; only an empty `data/.gitkeep` is tracked
so the folder structure survives a fresh clone).

- 9 subjects, 3 wearable IMU sensors (wrist, chest, ankle) at 100Hz + a heart-rate
  monitor at ~9Hz
- 18 activities (walking, running, cycling, ironing, vacuuming, rope jumping, etc.)
- `Protocol/` = required 12-activity protocol, all 9 subjects. `Optional/` = 6 extra
  activities, only some subjects did them — **not** a clean subject × activity grid
- 54 columns per row: timestamp, activityID, heart rate, then 3×17 columns for
  hand/chest/ankle IMU blocks (temperature, accel ±16g, accel ±6g, gyro, magnetometer,
  orientation — orientation columns are invalid/unused per the dataset's own readme)
- Free/open, CC BY 4.0, no credentialing needed
- No physical equipment needed on our end — data was already collected by the
  original researchers (DFKI); we only work with the downloaded files

## 2. Missing-data findings (from exploring the raw files)

Across all 14 files (3,850,505 rows total):

- **90.87% of rows are missing a heart-rate value** — this is *not* real data loss.
  It's a sample-rate mismatch: HR is logged at ~9Hz but every row is timestamped on
  the 100Hz IMU clock, so ~91/100 rows simply fall between two real HR readings.
  Fixed trivially with forward-fill/resampling — not a genuine problem.
- **Excluding heart rate, real IMU sensor dropout is rare**: ~0.86% of rows on a
  representative subject file. When a sensor does drop a reading, the *whole*
  packet for that sensor goes NaN at once (all 17 columns together), not individual
  channels independently.
- Dropout isn't evenly spread across sensors: **hand ≈0.37%, ankle ≈0.35%, chest
  ≈0.11%** (of all rows, dataset-wide) — chest is the most reliable sensor.
- **No naturally occurring case where all 3 IMUs drop out at the same timestamp** —
  so testing "what if a sensor fails" requires synthetically zeroing out a sensor
  stream ourselves; it doesn't happen organically in this data.

## 3. Research plan

Course requirement (`docs/project-guidelines.md`): the contribution must be on the
**DNN side** (architecture / training objective / representation learning /
efficiency), not the data science side.

**Core contribution — placement-aware sensor fusion:**
Baseline = one small CNN per sensor location (hand/chest/ankle) → naive
concatenation → classifier. Proposed improvement = a fusion layer that *learns*
how much to weight each sensor per activity (e.g. trust ankle more for walking,
wrist more for ironing), instead of combining them equally.

**Planned experiments:**

1. **Missing-sensor robustness** — synthetically zero out one IMU stream at test
   time (justified by the finding above that this doesn't happen naturally) and
   compare how much the baseline vs. the proposed fusion model degrades.
2. **Class imbalance check** — rare activities (rope jumping, soccer) have far
   fewer samples than common ones (sitting, walking). Report per-activity F1, not
   just overall accuracy, and try a class-weighted/focal loss as the "new training
   objective" angle.
3. **Subject-independent generalization (LOSO)** — Leave-One-Subject-Out
   cross-validation instead of a random split, so the model is tested on people
   it's never seen rather than memorizing individual movement styles.
4. **(Bonus) Efficiency for on-device deployment** — since this is wearable data,
   measure model size/inference speed as a proxy for real smartwatch feasibility
   (no physical device needed — this is standard practice, not a real deployment
   test).

## 4. Repo structure

```
docs/
  project-guidelines.md      course brief / requirements
  research-ideas.pdf         original 10 candidate project directions
  initial-datasets.pdf       dataset reference for all 10 ideas (sources, licenses, access notes)
  contributor-logs/          per-member work logs
data/
  pamap2+.../                 the actual dataset (gitignored, local only)
```

## 5. Kaggle access

Not needed for PAMAP2 (direct UCI download), but set up for other ideas' datasets
that are Kaggle-hosted. Token lives at `~/.kaggle/kaggle.json` (chmod 600, not in
this repo). Setup steps are in `docs/initial-datasets.pdf`.
