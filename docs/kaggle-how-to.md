# Run the baseline pipeline on Kaggle

1. Run `00_kaggle_data_download.ipynb` with Internet enabled and save its output.
2. Attach that output to `01_preprocessing.ipynb` and save the manifest output.
3. Run `02_direct_cnn_baseline.ipynb` and `03_instrument_pretraining.ipynb` with a GPU.
4. Run `04_rhythm_targets.ipynb`, `05_timbre_targets.ipynb`, and `06_harmony_targets.ipynb` on CPU.
5. Attach the required earlier outputs to `07_descriptor_fusion_baseline.ipynb` and run it with a GPU.
6. Run `08_baseline_evaluation.ipynb` and `09_baseline_explainability.ipynb` against the saved baseline artifacts.

```text
00 → 01
01 → 02 and 03
01 → 04, 05 and 06
01 + 03 + 04 + 05 + 06 → 07
02 + 07 → 08
01 + 03 + 04 + 05 + 06 + 07 → 09
```

Kaggle does not preserve `/kaggle/working` between notebooks. Use **Add Input → Notebook Output** for every dependency and save every successful stage with output enabled.

The proposed-model workflow will be documented here after its implementation exists.
