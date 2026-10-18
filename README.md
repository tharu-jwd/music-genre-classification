# MuCoNet: Concept-Guided Music Genre Classification

MuCoNet predicts six music genres from a shared convolutional audio encoder and
four supervised concept branches: instruments (41 outputs), timbre (12), rhythm
(10), and harmony (12). The primary V2 configuration concatenates the concept
predictions and applies a linear genre classifier. A direct CNN baseline and
branch-removal comparisons use the same training and evaluation workflow.

## Current implementation

- [Architecture and rationale](docs/common-concept-architecture.md)
- [Editable architecture settings](configs/architecture.json) and
  [configuration guide](docs/architecture-config.md)
- [All 21 individual Modal commands](docs/paper-individual-runs.md)
- [Preparation, training, downloads, and local analysis](docs/paper-quickstart.md)
- [Detailed experiment workflow](docs/paper-experiment-workflow.md)
- [Shared encoder tensor and checkpoint contracts](docs/shared-cnn-encoder-architecture.md)

For the paper experiments, use `data/vector-dataset.csv` and the fixed split table.
The trainer fits descriptor scaling on training tracks only. The audio cache is
`mtg-jamendo/logmel_songs` in Modal. Follow the quickstart for workspace setup.

## Main entry points

| Purpose | Entry point |
|---|---|
| One complete paper run: training, validation, and test | `modal_paper.py` |
| Current joint concept training | `scripts/train_joint.py` |
| Direct CNN training | `scripts/train_cnn.py` |
| Local analysis of downloaded run artifacts | `scripts/analyze_paper_runs.py` |
| Current shared encoder implementation | `shared_encoder/` |
| Older encoder import path, forwarding to the current implementation | `scripts/shared_audio_encoder.py` |

Checkpoints and results stay in the configured run directories. Cleanup does not
change training settings or remove datasets, checkpoints, predictions, or results.

## Data preparation and optional workflows

[Colab](notebooks/colab/README.md) and [Kaggle](notebooks/kaggle/README.md) retain
download, preprocessing, and target-preparation notebooks. Branch packages retain
standalone training and extraction tools for starting from raw data or running
separate comparisons.

`scripts/run_all_fusion.py`, `scripts/run_concept_fusion_mock.py`, and the fixture
pipeline in `concept_fusion/` support synthetic contract checks. Their scores are
not the paper's real-audio CNN or joint-training results. The optional gated/token
fusion implementation is retained because it is still selectable in the trainer.
Harmony chroma/chord screening tools are also retained as a separate legacy route.

Historical plans and diagrams remain under `docs/` and the branch directories;
use the V2 architecture document above for the current paper configuration.
Paper drafting utilities and generated manuscript material live outside this
repository in the sibling `../paper-writing/` directory.

## Checks

Run the existing tests from this directory:

```powershell
python -m pytest
```

Large audio caches, extracted tables, checkpoints, and run artifacts are ignored
by Git. Generated Python and pytest caches can be removed and are recreated when
needed.
