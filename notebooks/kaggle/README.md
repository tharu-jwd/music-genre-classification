# Kaggle baseline workflow

Run the numbered notebooks by attaching saved outputs from their dependencies. Start with `00_kaggle_data_download.ipynb`; use a GPU for `02_direct_cnn_baseline.ipynb`, `03_instrument_pretraining.ipynb`, and `07_descriptor_fusion_baseline.ipynb`.

The `04–06` notebooks create both descriptor-baseline inputs and supervision targets for the future proposed model. See the [Kaggle run guide](../../docs/kaggle-how-to.md) for the complete dependency graph.
