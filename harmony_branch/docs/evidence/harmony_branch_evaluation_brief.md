# Tharupahan — harmony branch

## Purpose and evidence
The second run enabled harmony supervision and reported **12 track-level tonal descriptors**, test macro R² **0.4704**, and standardized MAE **0.4791**. Some descriptors are strong (Tonnetz movement mean R² 0.811); individual Tonnetz components are weak (about 0.072, 0.044, and -0.004). Our more recent proposed branch instead predicts **12 pitch-class chroma probabilities per valid token**, masked-averages them, and passes the 12-value mean to fusion. These are different prediction tasks, even though both have 12 outputs.

## Read and question first
Read `project.md.txt`, especially Sections 5.3, 7.7, 10.1, 11.4, and 13. It warns that dividing mel bins into 12 adjacent bands is not chroma. Verify which extraction path and architecture the current run actually uses, and identify where the document reflects an older snapshot. Evaluate the proposal below critically; select the scientifically defensible target rather than following this file mechanically.

## Checks before changing architecture
1. Trace the 12 current labels to raw audio or a mathematically valid pitch-class representation. Document extraction settings, harmonic separation, alignment, missing/low-tonality handling, and target normalization.
2. Inspect the branch input `(B,T,128)`, token validity mask, temporal resolution, and target resolution. For per-token chroma, confirm frame/window alignment and loss on valid tokens only.
3. Check what is sent to Thevindu: per-token softmax then masked mean `(B,12)`, 12 standardized tonal regressions, or a hidden embedding. Write the actual computation and tensor shapes.
4. Examine weak Tonnetz components by feature distribution and musical identifiability before assuming they need a deeper model.

## Candidate changes to evaluate
- Decide explicitly whether the research model uses **per-token chroma probabilities** or **track-level tonal descriptors**; give a musical and measurement rationale. If comparing both, mark them as separate experiments.
- For chroma: use a pitch-class teacher computed from suitable audio/STFT, apply valid-token masking, and assess chroma prediction quality at matching time resolution. Averaging chroma discards ordering; consider an additional temporal summary only if ablation supports it.
- For descriptor regression: keep its standardized masked regression contract and test loss/feature quality fixes before architecture expansion.

## Experiments and handoff
Report branch metrics appropriate to the selected target, per-feature or per-pitch-class diagnostics, genre macro AP, and matched ablations. Agree with Thevindu on whether fusion receives probabilities or standardized regressions; never silently substitute one for the other. Deliver a note distinguishing implemented, proposed, and rejected approaches, with evidence. Do not blindly implement this file.
