# Comparable six-genre CNN baseline

The direct baseline uses the same SharedAudioEncoder as joint training:
log-mel windows -> Conv 1/32 -> Conv 32/64 -> Conv 64/96 (GroupNorm,
GELU, and the existing pooling) -> frequency attention -> 128D projection
-> masked song pooling -> Linear 128/128 -> ReLU -> Dropout 0.15
-> Linear 128/6. Outputs are independent genre logits, trained with BCE.
The encoder is trained end to end. No concept predictions, targets, or losses
enter this model's forward pass or objective.

Both trainers share dataset construction, six-label vocabulary, track splits,
window selection, padding masks, and macro AP implementation. Both use AdamW,
cosine decay to 5% of initial LR, gradient clipping at 5, and checkpoint selection
by validation macro AP. Baseline loading intentionally still uses the combined
concept dataset to preserve the joint run's cohort. Auxiliary tables are loaded
but ignored by the baseline model. Parameter counts differ; this is an encoder-
matched comparison, not a parameter-count-matched experiment.

From the repository root, for a 10-epoch, batch-4 L4 run:

```powershell
modal run --detach modal_app.py --model cnn --gpu L4 --epochs 10 --batch-size 4 --learning-rate 0.0003 --max-windows 12 --run-name cnn-l4-b4-10ep-v1
```

Use the **actual** max-windows, initial learning rate, dataset CSV, split manifest,
and audio extraction configuration of joint-l4-b4-10ep-v2. The console excerpt
alone does not establish all those settings; 12 windows is the runner default,
not a verified property of that run. Do not change the data Volume between runs.
The existing joint checkpoint records max_windows, window_frames, and mel_config.
Use the original launch arguments for the remaining training settings.

Local equivalent (add the same data path options used for joint training):

```powershell
python scripts/train_joint.py --model cnn --epochs 10 --batch-size 4 --lr 0.0003 --max-windows 12 --out-dir results/cnn
```

Use --quick for a three-epoch, 32-track-per-split smoke test only; --skip-test
reserves the test evaluation. Modal supports both flags. Use a distinct run name
for every experiment. This implementation does not launch paid training itself.

Outputs: best.pt, results.json, validation_predictions.npz, and (unless skipped)
test_predictions.npz. Results include split track IDs, configuration, parameter
count, per-genre AP, training history, and the same test_macro_ap field as joint
training. AP excludes genres with no positive examples; their per-genre score is
null. Loss uses the same mean-over-batches reporting convention as joint training.
Compare test macro AP and test genre loss, not joint total training loss against
baseline genre-only training loss. Compare the baseline score with 0.6781 only
after matching the protocol. One run is a point estimate; repeated runs and
paired prediction comparisons are needed to establish a reliable advantage.
