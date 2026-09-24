import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOKS = (
    "02_direct_cnn_baseline.ipynb",
    "03_instrument_pretraining.ipynb",
    "07_descriptor_fusion_baseline.ipynb",
)
EXPECTED_GPU_JOBS = {
    "02_direct_cnn_baseline.ipynb": "direct_cnn",
    "03_instrument_pretraining.ipynb": "instrument_pretraining",
    "07_descriptor_fusion_baseline.ipynb": "descriptor_fusion",
}


def code_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


class NotebookResourceControlsTest(unittest.TestCase):
    def test_every_gpu_training_notebook_has_resource_guards(self):
        for runtime in ("colab", "kaggle"):
            for name in TRAINING_NOTEBOOKS:
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    source = code_source(path)
                    self.assertIn('os.environ.get("MAX_GPU_RUN_MINUTES", "120")', source)
                    self.assertIn("PATIENCE", source)
                    self.assertIn("wall-time cap reached", source)
                    self.assertIn("runtime.json", source)

    def test_every_gpu_training_notebook_requires_approval_and_keeps_provenance(self):
        for runtime in ("colab", "kaggle"):
            for name, expected_job in EXPECTED_GPU_JOBS.items():
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    source = code_source(path)
                    compact = "".join(source.split())
                    self.assertIn(
                        f'require_gpu_run_approval(DEVICE,"{expected_job}")',
                        compact,
                    )
                    self.assertIn("approved_run_limits(", source)
                    self.assertIn("apply_approved_cohort(manifest, GPU_RUN, MANIFEST)", source)
                    self.assertGreaterEqual(compact.count('"gpu_run":GPU_RUN'), 2)

    def test_gpu_cap_is_checked_between_batches_and_accounts_final_work(self):
        for runtime in ("colab", "kaggle"):
            for name in TRAINING_NOTEBOOKS:
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    source = code_source(path)
                    self.assertIn("GPU_RUN_STARTED = time.perf_counter()", source)
                    self.assertIn("TRAINING_DEADLINE", source)
                    self.assertIn("time.perf_counter() >= TRAINING_DEADLINE", source)
                    self.assertIn("training-time reserve reached between batches", source)
                    self.assertIn("write_gpu_termination_ledger(", source)
                    self.assertIn('print("preflight backward: OK")', source)
                    self.assertIn("opt.zero_grad(set_to_none=True)", source)
                    self.assertIn('print("final runtime", runtime)', source)
                    self.assertNotIn("train_started = time.perf_counter()", source)

    def test_instrument_export_obeys_the_total_gpu_deadline(self):
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "03_instrument_pretraining.ipynb")
            with self.subTest(runtime=runtime):
                self.assertIn("time.perf_counter() >= GPU_DEADLINE", source)
                self.assertIn("embedding_export_cap_reached", source)

    def test_test_evaluation_is_final_run_opt_in(self):
        for runtime in ("colab", "kaggle"):
            for name in TRAINING_NOTEBOOKS:
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    source = code_source(path)
                    self.assertIn('EVALUATE_TEST = os.environ.get("EVALUATE_TEST", "0") == "1"', source)
                    self.assertIn("if EVALUATE_TEST:", source)
                    self.assertIn("Test evaluation skipped", source.replace("Instrument test", "Test evaluation"))

    def test_audio_models_consume_ordered_masked_windows(self):
        for runtime in ("colab", "kaggle"):
            for name in ("02_direct_cnn_baseline.ipynb", "03_instrument_pretraining.ipynb"):
                path = ROOT / "notebooks" / runtime / name
                with self.subTest(runtime=runtime, notebook=name):
                    source = code_source(path)
                    self.assertIn("windows, mask = segment_logmel(", source)
                    self.assertIn("LOGMEL_SCHEMA_VERSION", source)
                    self.assertNotIn("x_mean = self._fix2d(x.mean(axis=0))", source)
                    self.assertNotIn("x = self._fix2d(x.mean(0))", source)

    def test_instrument_supervision_excludes_unannotated_songs(self):
        for runtime in ("colab", "kaggle"):
            path = ROOT / "notebooks" / runtime / "03_instrument_pretraining.ipynb"
            with self.subTest(runtime=runtime):
                source = code_source(path)
                self.assertIn('load_split_multihot(song_ids, "instrument", "instrument")', source)
                self.assertIn("manifest.instrument_available", source.replace('["instrument_available"]', ".instrument_available"))

    def test_manifest_publishes_required_availability_fields(self):
        required = (
            "waveform_available", "genre_available", "instrument_available", "rhythm_available",
            "timbre_available", "harmony_available",
        )
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "01_preprocessing.ipynb")
            for field in required:
                with self.subTest(runtime=runtime, field=field):
                    self.assertIn(field, source)

    def test_harmony_preflight_does_not_silently_choose_duplicate_audio(self):
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "06_harmony_targets.ipynb")
            with self.subTest(runtime=runtime):
                self.assertIn("duplicate_waveform_candidates", source)
                self.assertIn("resolved is not None", source)
                self.assertIn('manifest["waveform_available"]', source)
                self.assertIn("manifest.to_csv(MANIFEST, index=False)", source)
                self.assertNotIn("sid and sid not in audio_index", source)

    def test_harmony_cpu_ladder_is_opt_in_commit_pinned_and_cuda_disabled(self):
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "06_harmony_targets.ipynb")
            with self.subTest(runtime=runtime):
                self.assertIn('os.environ.get("RUN_HARMONY_CHROMA_GATE", "0") == "1"', source)
                self.assertIn('os.environ.get("PREPARE_HARMONY_SCREEN", "0") == "1"', source)
                self.assertIn('os.environ.get("RUN_HARMONY_SCREEN", "0") == "1"', source)
                self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = ""', source)
                self.assertIn('re.fullmatch(r"[0-9a-f]{40}", code_commit)', source)
                self.assertIn("checkout", source)
                self.assertIn("--detach", source)
                self.assertIn("run_environment.json", source)
                self.assertNotIn("--allow-test", source)

    def test_harmony_notebook_exposes_the_complete_cpu_gate_sequence(self):
        expected_scripts = (
            "freeze_experiment_cohort.py",
            "export_harmony_regions.py",
            "benchmark_harmony_extractors.py",
            "decide_harmony_extractor.py",
            "materialize_harmony_target_pilot.py",
            "cache_harmony_encoder_pilot.py",
            "build_harmony_screen_dataset.py",
            "screen_temporal_harmony_branch.py",
            "decide_harmony_branch_screen.py",
        )
        for runtime in ("colab", "kaggle"):
            source = code_source(ROOT / "notebooks" / runtime / "06_harmony_targets.ipynb")
            with self.subTest(runtime=runtime):
                for script in expected_scripts:
                    self.assertIn(script, source)
                self.assertLess(
                    source.index("validate_policy(json.loads(policy.read_text()))"),
                    source.index('scripts / "screen_temporal_harmony_branch.py"'),
                )


if __name__ == "__main__":
    unittest.main()
