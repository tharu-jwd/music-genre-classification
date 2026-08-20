# Full Kaggle process (shards 00–09) → next notebook → GitHub

**Branch:** `thevindu-branch`  
**Download kernel slug:** `thevifernando/dnn-download-data-1`  
**Input path in later notebooks:** `/kaggle/input/dnn-download-data-1/`

Do **not** git-push `.npy` files. GitHub gets **notebooks**; Kaggle stores **data**.

---

## 0. Disk reality

- Notebook `00` downloads **10 mel shards** (`raw_30s_melspecs-00.tar` … `-09.tar`).
- Each tar is extracted, then deleted.
- `/kaggle/working` is about **20 GB**. If a shard fails with no space, Save Version with shards you have, or continue remaining shards in a second kernel.
- Later notebooks **read mels from Add Input** (read-only). They do **not** copy 10 shards again.

---

## 1. Run notebook 00 (`dnn-download-data-1`)

1. [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**.
2. Settings: **Internet On**, **GPU Off**.
3. File → **Upload notebook** → `notebooks/00_kaggle_data_download.ipynb`  
   (or paste cells). Rename the kernel if you want; keep slug `dnn-download-data-1` if you already use that.
4. **Run All**. Wait until shards 00–09 finish (`n_npy` printed).
5. **Save Version** → **Save & Run All** (or Quick Save if already done).
6. **Advanced** → **Always save output** → Save.
7. Wait for version status **Success**.

That output **is** the dataset for later notebooks. You do not need:

```bash
kaggle kernels output thevifernando/dnn-download-data-1 -p /path/to/dest
```

on Kaggle. That CLI is only for your laptop.

---

## 2. Pass 00 → 01 (preprocessing)

1. **New Notebook**.
2. Upload `notebooks/01_preprocessing.ipynb`.
3. **Add Input** → **Notebook Output** → **`dnn-download-data-1`**.
4. Confirm `/kaggle/input/dnn-download-data-1/` exists (bootstrap cell prints this).
5. Internet **On** (so missing TSVs can wget).
6. **Run All**. Check `song_manifest.csv` split counts.
7. **Save Version + save output**.

---

## 3. Chain 01 → 02 → … → 09

For **each** next notebook:

| Notebook | GPU? | Add Input (at least) | What it writes |
|---|---|---|---|
| 02 CNN baseline | Yes | `dnn-download-data-1` **and** notebook 01 output (manifest) | `checkpoints/baseline/`, test JSON |
| 03 instrument | Yes | 00 + 01 | `features/instrument/`, `checkpoints/stage1/` |
| 04 / 05 / 06 features | No | 00 + 01 | `features/{rhythm,timbre,harmony}/` |
| 07 fusion | Yes | 00 + 01 + 03 + 04 + 05 + 06 | `checkpoints/stage2/`, test JSON |
| 08 ablations | No/Yes | 02 + 07 results | CSV tables |
| 09 explainability | No | 01 + 07 | figures + listening CSV |

**Add Input** can attach **several** previous notebook outputs. The bootstrap looks under all of `/kaggle/input`.

After each notebook: **Save Version with output** so the next one can attach it.

Suggested Kaggle titles:
- `dnn-download-data-1` (00)
- `dnn-01-preprocessing`
- `dnn-02-baseline`
- … through `dnn-09-explainability`

---

## 4. What each notebook does

1. **00** — annotations + mel shards **00–09**.
2. **01** — join mels to split-0 → `song_manifest.csv`.
3. **02** — CNN genre baseline, split-0 **test** metrics.
4. **03** — Stage 1 MIL instrument embeddings (64-d).
5. **04–06** — rhythm / timbre / harmony (librosa or mel proxy).
6. **07** — fusion + genre head, split-0 test.
7. **08** — ablations vs 0.7260 / 0.1592.
8. **09** — attention / qualitative XAI templates.

---

## 5. Upload **code** to GitHub (`thevindu-branch`)

On your PC, in this repo:

```bash
git checkout thevindu-branch
git add notebooks docs scripts README.md docs/contributor-logs/thevindu.md
git commit -m "Expand Kaggle pipeline to 10 mel shards and document kernel I/O."
git push origin thevindu-branch
```

Do **not** add `/kaggle/working` dumps or `.npy` shards.

---

## 6. Order checklist

- [ ] 00 Run All → Save output  
- [ ] 01 Add Input 00 → Run All → Save output  
- [ ] 02 Add Input 00+01, GPU → Run All → Save output  
- [ ] 03 same → embeddings  
- [ ] 04, 05, 06 (parallel OK)  
- [ ] 07 GPU, all features attached  
- [ ] 08, 09  
- [ ] Notebooks committed on `thevindu-branch`  
