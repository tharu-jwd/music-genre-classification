# Proposed-model implementation roadmap

## 1. Freeze the data contracts

- Verify log-Mel shapes and valid-window masks.
- Produce one aligned table of genre, instrument, rhythm, timbre, and harmony targets.
- Fit continuous-target normalization on training rows only.
- Record missing-value masks for every target group.

## 2. Extract reusable model code

- Move the window CNN and masked attention pooling out of notebook `03` into an importable module.
- Add unit tests for tensor shapes, padding masks, and empty/partial target masks.
- Load instrument-pretraining weights into the shared encoder.

## 3. Implement the proposed network

- Add the 64-D instrument branch.
- Add 32-D rhythm, timbre, and harmony branches.
- Add concept prediction heads used only for supervision and evaluation.
- Add gated fusion and the 128-D fused representation.
- Add the genre head and joint masked loss.

## 4. Establish the training sequence

1. Run the direct CNN baseline.
2. Run instrument pretraining.
3. Run the descriptor-fusion baseline.
4. Train the proposed model with fixed initial loss weights.
5. Tune only with validation metrics.
6. Evaluate the selected model once on test.

## 5. Required experiments

- Direct CNN versus descriptor fusion versus proposed model.
- Proposed model without concept supervision.
- Remove each concept branch individually.
- Concatenation versus gated fusion.
- Random initialization versus instrument-pretrained encoder.
- Loss-weight sensitivity.
- Parameters, training time, and inference latency.
- Gate analysis and concept-removal analysis.

## Definition of done

- A clean run succeeds in both the selected hosted workflow and a small smoke-test configuration.
- Best checkpoints are chosen only by validation performance.
- All metrics, target normalization, tag order, configuration, and random seed are saved.
- Undefined macro metrics are excluded, not zero-filled.
- Results include all required baselines and ablations.
- Documentation clearly separates measured results from interpretation.
