# Run the three-shard experiment locally on Windows

This repository includes generated local copies of the Colab workflow under
`notebooks/local/`. They write all large artifacts under
`data/MTG_Instrument/`, which Git ignores.

## One-time setup

1. Install Python 3.11 and JupyterLab.
2. Install a CUDA-enabled PyTorch build using the command selected for Windows
   and pip on the official PyTorch installation page. Then install the other
   project packages:

   ```powershell
   python -m pip install jupyterlab numpy pandas scikit-learn tqdm librosa matplotlib
   ```

3. In PowerShell, from the repository root, confirm that PyTorch can see the
   GPU:

   ```powershell
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

   It should print `True` and `NVIDIA GeForce RTX 2050`.

4. Start JupyterLab:

   ```powershell
   python -m jupyter lab
   ```

## Run order

Open `notebooks/local/` and run notebooks in this order:

1. `00_download_local.ipynb` downloads only shards `00`, `01`, and `02`.
2. `01_preprocessing.ipynb` creates the official split-0 song manifest.
3. Run `02_cnn_baseline.ipynb` for the comparison baseline.
4. Run `03_instrument_embedding.ipynb` to generate 64-dimensional instrument embeddings.
5. Run `04`, `05`, and `06` to create rhythm, timbre, and harmony features.
6. Run `07_fusion_genre_classifier.ipynb`, then `08` and `09`.

The generated notebooks use `data/MTG_Instrument` by default. To place the
data on a different disk, set `MTG_ROOT` before starting JupyterLab:

```powershell
$env:MTG_ROOT = 'D:\MTG_Instrument'
python -m jupyter lab
```

## RTX 2050 settings

The local notebooks reduce the CNN baseline batch size to 4 and the Stage 1
instrument batch size to 2. If CUDA reports an out-of-memory error, reduce
those values to 2 and 1 respectively, restart the kernel, and run the notebook
again from the top.
