# 🎵 Concept-Guided Explainable Music Genre Classification

> Learning interpretable musical concept embeddings through progressive concept-guided representation learning for explainable multi-label music genre classification.

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue.svg">

<img src="https://img.shields.io/badge/PyTorch-2.x-red.svg">

<img src="https://img.shields.io/badge/Librosa-Latest-orange.svg">

<img src="https://img.shields.io/badge/Research-Explainable%20AI-green.svg">

<img src="https://img.shields.io/badge/Status-Active%20Development-yellow.svg">

</p>

---

# 📖 Overview

Traditional music genre classification models operate as black-box systems that directly map audio signals to genre labels, providing little insight into the musical reasoning behind their predictions.

This research proposes a **Concept-Guided Explainable Deep Learning Framework**, where the model first learns human-understandable musical concepts before performing final genre classification.

Instead of directly predicting genres, the framework progressively learns concept representations corresponding to

- 🎸 Instrumentation
- 🥁 Rhythm
- 🎹 Timbre
- 🎼 Harmony

Each concept is learned independently using the same shared audio representation, allowing every learned embedding to become semantically meaningful and interpretable.

The final genre classifier utilizes these learned concept embeddings to perform multi-label genre prediction while providing a transparent explanation of the musical evidence supporting each prediction.

---

# 🎯 Research Contributions

This work proposes

- Progressive concept-guided representation learning
- Attention-based Multiple Instance Learning (MIL) for song-level representation
- Explainable intermediate musical concepts
- Transferable concept embeddings
- Multi-label genre prediction using learned concept fusion
- Concept-level explainability for music understanding

---

# 🧠 Proposed Framework

## Overall Research Pipeline

```
                           MP3 Audio
                               │
                               ▼
                        librosa.load()
                               │
                               ▼
                    Normalize Waveform
                               │
                               ▼
                  15-second Window Segmentation
                               │
                               ▼
                  Log-Mel Spectrogram Windows
                               │
                               ▼
                  Shared Window-level CNN Encoder
                               │
                               ▼
                  Attention-based MIL Pooling
                               │
                               ▼
                 Song-level Latent Representation
                               │
                               ▼
                   Stage 1: Instrument Head
                               │
                               ▼
                 Instrument Concept Embedding
                               │
                               ▼
                     Save / Freeze Encoder
──────────────────────────────────────────────────────────────

                   Stage 2: Rhythm Head
                               │
                               ▼
                   Rhythm Concept Embedding
                               │
                               ▼
                     Save / Freeze Encoder
──────────────────────────────────────────────────────────────

                   Stage 3: Timbre Head
                               │
                               ▼
                   Timbre Concept Embedding
                               │
                               ▼
                     Save / Freeze Encoder
──────────────────────────────────────────────────────────────

                  Stage 4: Harmony Head
                               │
                               ▼
                  Harmony Concept Embedding
──────────────────────────────────────────────────────────────

          Concatenate All Learned Concept Embeddings
                               │
                               ▼
               Multi-label Genre Classification
```

---

# 💡 Why Progressive Concept Learning?

Unlike conventional end-to-end genre classifiers, the proposed framework separates musical understanding into multiple interpretable learning stages.

Each stage learns a specific musical concept independently while sharing the same CNN feature extractor.

Advantages include

- Better interpretability
- Transferable concept representations
- Reduced feature entanglement
- Easier ablation studies
- Modular architecture
- Improved explainability

---

# 📂 Dataset

## MTG-Jamendo Dataset

The experiments utilize the MTG-Jamendo dataset containing

- Audio recordings
- Instrument annotations
- Genre annotations
- Artist metadata
- Album metadata

Each song is preprocessed only once.

Output structure

```
Song
│
├── stacked Mel spectrogram (.npy)
│
├── metadata
│
└── concept labels
```

Example Mel tensor

```
(12, 128, 469)

12 windows
128 Mel bins
469 frames
```

---

# ⚙️ Audio Preprocessing Pipeline

Each MP3 file is decoded exactly once.

```
MP3 Audio
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
Stack Windows
     │
     ▼
Song.npy
     │
     ▼
Delete Temporary MP3
```

Advantages

- No repeated MP3 decoding
- Fast experimentation
- Efficient storage
- Resumable preprocessing

---

# 🏗 Stage 1 — Instrument Concept Learning

The shared encoder learns discriminative instrument-family representations using grouped MTG-Jamendo labels.

Current instrument groups include

- Guitar
- Bass
- Strings
- Keyboard
- Brass
- Woodwinds
- Percussion
- Voice
- Electronic
- Orchestra

Training objective

```
Mel Windows
      │
      ▼
CNN Encoder
      │
      ▼
Attention Pooling
      │
      ▼
Song Embedding
      │
      ▼
Instrument Prediction
      │
      ▼
BCEWithLogitsLoss
```

The learned embedding is exported for downstream tasks.

---

# 🥁 Stage 2 — Rhythm Concept Learning

The second stage reuses the learned encoder to model rhythmic characteristics including

- Tempo
- Beat structure
- Rhythmic density
- Groove patterns
- Percussive behavior

The resulting rhythm embedding captures temporal musical information.

---

# 🎹 Stage 3 — Timbre Concept Learning

The third stage focuses on spectral texture.

Representative concepts include

- Brightness
- Warmth
- Spectral centroid
- Spectral bandwidth
- Spectral contrast
- Spectral flatness
- Spectral flux

---

# 🎼 Stage 4 — Harmony Concept Learning

The final concept stage captures harmonic content using representations derived from

- Chroma
- Tonal relationships
- Harmonic progression
- Tonnetz representations

---

# 🎯 Final Genre Prediction

After all concept representations are learned

```
Instrument Embedding

+

Rhythm Embedding

+

Timbre Embedding

+

Harmony Embedding

↓

Feature Fusion

↓

Genre Classifier

↓

Multi-label Genre Prediction
```

The final prediction is therefore based upon interpretable musical concepts rather than latent black-box features.

---

# 📈 Training Strategy

```
Stage 1

↓

Train Instrument Concept Encoder

↓

Freeze Encoder

↓

Stage 2

↓

Learn Rhythm Concepts

↓

Freeze Encoder

↓

Stage 3

↓

Learn Timbre Concepts

↓

Freeze Encoder

↓

Stage 4

↓

Learn Harmony Concepts

↓

Feature Fusion

↓

Genre Classification
```

---

# 📊 Evaluation

Performance is evaluated using

- BCE Loss
- Macro F1
- Micro F1
- Mean Average Precision (mAP)
- Precision
- Recall

Representation quality is further analyzed using

- PCA
- t-SNE
- UMAP

Concept explainability is evaluated through

- Attention visualization
- Window importance analysis
- Concept confidence scores

---

# 📂 Repository Structure

```
dataset/
│
├── logmel_songs/
├── song_manifest.csv
├── label_schema.json
└── logs/

notebooks/
│
├── 01_preprocessing.ipynb
├── 02_instrument_learning.ipynb
├── 03_rhythm_learning.ipynb
├── 04_timbre_learning.ipynb
├── 05_harmony_learning.ipynb
└── 06_genre_prediction.ipynb

models/

results/

experiments/

README.md
```

---

# 📈 Experiment Tracking

Experiments are automatically logged using

- MLflow
- TensorBoard
- CSV histories

Recorded information includes

- Hyperparameters
- Training history
- Validation metrics
- Model checkpoints
- Learned embeddings
- Configuration files

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

## Completed

- MTG-Jamendo preprocessing
- Song-level Mel generation
- Window stacking
- Attention-based MIL pooling
- Instrument concept learning
- Instrument embedding extraction
- Experiment tracking

## Ongoing

- Rhythm concept learning
- Timbre concept learning
- Harmony concept learning
- Genre concept fusion
- Explainability analysis
- Ablation studies

---

# 📚 Citation

If this work contributes to your research, please cite the associated publication once released.

---

# 📄 License

This repository is intended for academic research and educational purposes.

Please respect the licensing terms of the MTG-Jamendo dataset.

---

# 👨‍💻 Authors

- Tharupahan Jayawardhana
- Dehan Wijesinghe
- Thevindu Fernando
- Anupama Wickramaratne
- Senindu Dinapura

Research Areas

- Explainable Artificial Intelligence (XAI)
- Music Information Retrieval (MIR)
- Deep Representation Learning
- Multi-label Learning
- Audio Signal Processing

---

# 🙏 Acknowledgements

- MTG-Jamendo Dataset
- Music Technology Group (Universitat Pompeu Fabra)
- PyTorch
- Librosa
- Google Colab
