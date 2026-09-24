import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ESSENTIAL_NOTEBOOKS = {
    "colab": (
        "00_download_to_drive.ipynb",
        "01_preprocessing.ipynb",
        "04_rhythm_targets.ipynb",
    ),
    "kaggle": (
        "00_kaggle_data_download.ipynb",
        "01_preprocessing.ipynb",
        "04_rhythm_targets.ipynb",
    ),
}
RETIRED_NOTEBOOKS = (
    "02_direct_cnn_baseline.ipynb",
    "03_instrument_pretraining.ipynb",
    "05_timbre_targets.ipynb",
    "06_harmony_targets.ipynb",
    "07_descriptor_fusion_baseline.ipynb",
    "08_baseline_evaluation.ipynb",
    "09_baseline_explainability.ipynb",
)


def code_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


class NotebookResourceControlsTest(unittest.TestCase):
    def test_essential_shared_notebooks_remain(self):
        for runtime, names in ESSENTIAL_NOTEBOOKS.items():
            for name in names:
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    self.assertTrue(path.is_file(), f"missing essential notebook {path}")

    def test_retired_numbered_notebooks_are_gone(self):
        for runtime in ("colab", "kaggle"):
            for name in RETIRED_NOTEBOOKS:
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    self.assertFalse(path.exists(), f"retired notebook still present: {path}")

    def test_manifest_publishes_required_availability_fields(self):
        required = (
            "waveform_available",
            "genre_available",
            "instrument_available",
            "rhythm_available",
            "timbre_available",
            "harmony_available",
        )
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "01_preprocessing.ipynb")
            for field in required:
                with self.subTest(runtime=runtime, field=field):
                    self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
