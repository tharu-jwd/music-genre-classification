# 🎵 Concept-Guided Explainable Music Genre Classification

> Learning interpretable music representations through concept-guided embeddings for multi-label genre classification.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red.svg">
  <img src="https://img.shields.io/badge/Librosa-Latest-orange.svg">
  <img src="https://img.shields.io/badge/Research-Deep%20Learning-green.svg">
  <img src="https://img.shields.io/badge/Status-Active%20Development-yellow.svg">
</p>

---

# 📖 Overview

Music genre classification is traditionally treated as a black-box deep learning problem, where neural networks directly predict genres from audio without providing interpretable reasoning.

This research proposes a **Concept-Guided Explainable Deep Learning Framework** that learns human-understandable musical concepts before performing genre classification.

Instead of predicting genres directly, the model first learns embeddings representing musical concepts such as

- 🎸 Instrumentation
- 🥁 Rhythm
- 🎹 Timbre
- 🎼 Harmony

These learned concept embeddings are then fused into a unified music representation for final multi-label genre prediction.

The goal is to improve both **classification performance** and **model interpretability**.

---

# 🧠 Proposed Architecture

```
                         MP3 Audio
                             │
                             ▼
                    librosa.load()
                             │
                 15-second segmentation
                             │
                             ▼
               Log-Mel Spectrogram Windows
                             │
                             ▼
                  Shared CNN Encoder
                             │
        ┌──────────┬──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
 Instrument     Rhythm     Timbre     Harmony
 Embedding     Embedding   Embedding  Embedding
        │          │          │          │
        └──────────┴──────────┴──────────┘
                     Concatenation
                           │
                           ▼
                 Attention-based Pooling
                           │
                           ▼
                 Music Representation
                           │
                           ▼
             Multi-label Genre Classification
```

---

# 🔬 Research Objectives

The objectives of this research are

- Learn meaningful concept embeddings from music audio.
- Improve genre classification through intermediate musical concepts.
- Provide explainable representations rather than black-box predictions.
- Investigate the contribution of different musical concepts through ablation studies.
- Evaluate concept-guided learning against conventional end-to-end CNN models.

---

# 📂 Dataset

**Dataset**

MTG-Jamendo Dataset

Contains

- Multi-label genre annotations
- Instrument annotations
- Artist metadata
- Album metadata
- Audio recordings

Repository preprocessing converts each song into

```
Song
│
├── stacked Mel spectrogram (.npy)
│
└── metadata
```

Each stacked Mel contains multiple **15-second** windows.

Example

```
(12, 128, 469)

12 windows
128 Mel bins
469 time frames
```

---

# ⚙️ Preprocessing Pipeline

Each MP3 is decoded **exactly once**.

```
MP3
 │
 ▼
librosa.load()
 │
 ▼
Normalize Audio
 │
 ▼
Split into 15-second windows
 │
 ▼
Log-Mel Spectrogram
 │
 ▼
Stack windows
 │
 ▼
Song.npy
 │
 ▼
Delete MP3
```

This design avoids repeatedly decoding the same audio for different experiments.

---

# 🏗 Repository Structure

```
.
├── dataset/
│   ├── logmel_songs/
│   ├── song_manifest.csv
│   ├── label_schema.json
│   └── logs/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_instrument_embedding.ipynb
│   ├── 03_rhythm_embedding.ipynb
│   ├── 04_timbre_embedding.ipynb
│   ├── 05_harmony_embedding.ipynb
│   └── 06_genre_classifier.ipynb
│
├── models/
│
├── experiments/
│
├── results/
│
└── README.md
```

---

# 🎯 Concept Learning

The proposed model learns four independent concept spaces.

## Instrument Embedding

Learns instrument-family representations.

Examples

- Guitar
- Strings
- Keyboard
- Brass
- Woodwinds
- Percussion
- Voice
- Electronic

---

## Rhythm Embedding

Learns rhythmic characteristics including

- Tempo
- Beat strength
- Onset density
- Beat interval statistics

---

## Timbre Embedding

Learns spectral properties including

- Spectral centroid
- Bandwidth
- Contrast
- Flatness
- RMS energy
- Spectral flux

---

## Harmony Embedding

Learns harmonic information using

- Chroma
- Tonnetz

---

# 🧪 Training Strategy

Training proceeds in multiple stages.

```
Stage 1
↓

Instrument Representation Learning

↓

Stage 2

Rhythm Representation Learning

↓

Stage 3

Timbre Representation Learning

↓

Stage 4

Harmony Representation Learning

↓

Stage 5

Joint Concept-Guided Fine-tuning

↓

Final Genre Classification
```

---

# 📈 Evaluation

Evaluation metrics include

- Macro F1-score
- Micro F1-score
- Mean Average Precision (mAP)
- Precision
- Recall
- BCE Loss

Embedding quality will additionally be analyzed using

- UMAP
- t-SNE
- PCA

---

# 📊 Experiment Tracking

Experiments are tracked using

- MLflow
- TensorBoard
- CSV logs

Each experiment records

- Hyperparameters
- Training history
- Validation metrics
- Model checkpoints
- Embedding visualizations

---

# 💻 Technology Stack

- Python
- PyTorch
- Librosa
- NumPy
- Pandas
- MLflow
- TensorBoard
- Google Colab

---

# 🚀 Current Progress

- Dataset preprocessing / mel shard workflow (Drive + Colab)
- CNN baseline (genre + instrument) on MTG-Jamendo **split-0**
- Stage 1 instrument embedding (MIL + attention) — pending three bug-fix re-verification
- Phase 2 team workflow, Drive layout, and code scaffolds on branch `thevindu-branch`

### Phase 2 focus (→ 23 Aug)

- Re-verify Stage 1 fixes (`best_macro_map`, split leakage, mel path fallback)
- Rhythm / timbre / harmony **librosa** features (Members 2–3)
- Fusion (linear + attention) + multi-label genre classifier (Member 1)
- Ablations vs baseline 0.7260 ROC-AUC / 0.1592 PR-AUC (Member 4)
- Explainability eval + Phase 2 short paper (Member 5)

See [`docs/phase2-workflow.md`](docs/phase2-workflow.md) for the full plan and checklist.

---

# 📚 Citation

If you use this repository in academic work, please cite the corresponding publication once available.

---

# 📄 License

This repository is intended for academic research and educational purposes.

Please respect the licensing terms of the MTG-Jamendo dataset.

---

# 👨‍💻 Authors

**Tharupahan Jayawardhana**
**Dehan Wijesinghe**
**Thevindu Fernando**
**Anupama Wickramaratne**
**Senindu Dinapura**

Research in Explainable Artificial Intelligence (XAI), Music Information Retrieval (MIR), Deep Learning, and Representation Learning.

---

## ⭐ Acknowledgements

- MTG-Jamendo Dataset
- Music Technology Group (Universitat Pompeu Fabra)
- PyTorch
- Librosa
- Google Colab
