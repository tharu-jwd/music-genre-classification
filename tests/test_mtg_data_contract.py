import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.mtg_data_contract import NOTEBOOK_DATA_CONTRACT


def load_contract(annotation_dir: Path):
    namespace = {"ANN_DIR": annotation_dir, "Path": Path, "np": np, "re": re}
    exec(NOTEBOOK_DATA_CONTRACT, namespace)
    return namespace


def write_split(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]
    text = "\t".join(header) + "\n"
    text += "\n".join("\t".join(row) for row in rows) + "\n"
    path.write_text(text)


class MtgDataContractTest(unittest.TestCase):
    def test_variable_width_rows_preserve_every_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.tsv"
            write_split(path, [[
                "track_0000241", "artist_000005", "album_000033", "41/241.mp3", "340.1",
                "genre---rock", "genre---alternative", "genre---indie",
            ]])
            contract = load_contract(Path(tmp))
            rows = list(contract["iter_tsv_rows"](path))
            self.assertEqual(rows[0]["song_id"], "0000241")
            self.assertEqual(rows[0]["TAGS"], (
                "genre---rock", "genre---alternative", "genre---indie",
            ))

    def test_split_vocabulary_is_fixed_and_missing_annotation_is_masked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits" / "split-0"
            annotated_ids = []
            for index, split in enumerate(("train", "validation", "test"), start=1):
                sid = f"{index:07d}"
                annotated_ids.append(sid)
                write_split(split_dir / f"autotagging_genre-{split}.tsv", [[
                    f"track_{sid}", "artist_000001", "album_000001", f"00/{index}.mp3", "30.0",
                    "genre---rock", "genre---jazz",
                ]])
            contract = load_contract(root)
            targets, labels, available = contract["load_split_multihot"](
                [*annotated_ids, "0000999"], "genre", "genre"
            )
            self.assertEqual(labels, ["genre---jazz", "genre---rock"])
            np.testing.assert_array_equal(targets[:3], np.ones((3, 2), dtype=np.float32))
            np.testing.assert_array_equal(targets[3], np.zeros(2, dtype=np.float32))
            np.testing.assert_array_equal(available, [True, True, True, False])

    def test_vocabulary_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits" / "split-0"
            tags = {
                "train": ["genre---rock", "genre---jazz"],
                "validation": ["genre---rock"],
                "test": ["genre---rock", "genre---jazz"],
            }
            for index, split in enumerate(("train", "validation", "test"), start=1):
                write_split(split_dir / f"autotagging_genre-{split}.tsv", [[
                    f"track_{index:07d}", "artist_000001", "album_000001", f"00/{index}.mp3", "30.0",
                    *tags[split],
                ]])
            contract = load_contract(root)
            with self.assertRaisesRegex(ValueError, "vocabulary differs"):
                contract["load_split_multihot"](["0000001"], "genre", "genre")

    def test_logmel_is_split_into_ordered_windows_with_mask(self):
        contract = load_contract(Path("unused"))
        raw = np.arange(20, dtype=np.float32).reshape(2, 10)
        windows, mask = contract["segment_logmel"](
            raw, n_mels=2, n_frames=3, max_windows=3
        )
        self.assertEqual(windows.shape, (3, 2, 3))
        np.testing.assert_array_equal(mask, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(windows[0], raw[:, 0:3])
        np.testing.assert_array_equal(windows[1], raw[:, 3:6])
        np.testing.assert_array_equal(windows[2, :, :1], raw[:, 9:10])

    def test_short_logmel_padding_is_not_marked_real(self):
        contract = load_contract(Path("unused"))
        raw = np.ones((2, 4), dtype=np.float32)
        windows, mask = contract["segment_logmel"](
            raw, n_mels=2, n_frames=3, max_windows=4
        )
        np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(float(windows[1, :, 1:].sum()), 0.0)
        self.assertEqual(float(windows[2:].sum()), 0.0)

    def test_temporal_metadata_preserves_partial_lengths_and_selected_start_times(self):
        contract = load_contract(Path("unused"))
        raw = np.ones((96, 1366 * 5 + 7), dtype=np.float32)
        windows, mask, valid_frames, starts = contract["segment_logmel_with_metadata"](
            raw, n_mels=96, n_frames=1366, max_windows=3
        )
        self.assertEqual(windows.shape, (3, 96, 1366))
        self.assertEqual(mask.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(valid_frames.tolist(), [1366, 1366, 7])
        expected_frames = [0, 2 * 1366, 5 * 1366]
        self.assertTrue(np.allclose(starts, np.asarray(expected_frames) * 256 / 12000))

    def test_window_plan_matches_the_exact_selected_logmel_chunks(self):
        contract = load_contract(Path("unused"))
        raw = np.arange(20, dtype=np.float32).reshape(2, 10)
        plan = contract["logmel_window_plan"](
            raw,
            n_mels=2,
            n_frames=3,
            max_windows=3,
            sample_rate=12,
            hop_length=3,
        )
        self.assertEqual([item["frame_start"] for item in plan], [0, 3, 9])
        self.assertEqual([item["frame_end_exclusive"] for item in plan], [3, 6, 10])
        self.assertEqual([item["start_seconds"] for item in plan], [0.0, 0.75, 2.25])
        self.assertEqual([item["end_seconds"] for item in plan], [0.75, 1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
