# Google Colab workflow

Use these notebooks when artifacts should persist in Google Drive under:

```text
/content/drive/MyDrive/MTG_Instrument
```

Run `00` and `01` first. Then run `02` and `03`; extract concepts with `04`–`06`; train fusion with `07`; finish with `08` and `09`.

Enable a GPU for `02`, `03`, and `07`. The other stages can use CPU. The default dataset subset is shards `00–02`, requiring roughly 8 GB of Drive space. Edit the shard lists only if you intentionally want a larger matched subset.

Do not run the Kaggle notebooks against this Drive tree.
