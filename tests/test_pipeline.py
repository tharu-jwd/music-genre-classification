from pathlib import Path

from concept_fusion.contract import CONCEPT_ORDER
from concept_fusion.experiments import experiment_specs, jobs_for
from concept_fusion.fixtures import make_cohort
from concept_fusion.pipeline import PipelineConfig, run_all, train_one
from concept_fusion.projections import TokenAssembler
from concept_fusion.thresholds import fit_thresholds


def test_cohort_ids_are_disjoint():
    c = make_cohort(n_train=8, n_val=4, n_test=4, seed=0)
    assert not set(c.train.song_ids) & set(c.val.song_ids)
    assert not set(c.train.song_ids) & set(c.test.song_ids)
    assert not set(c.val.song_ids) & set(c.test.song_ids)
    assert all(len(i) == 7 for i in c.all_ids())


def test_enabled_mask_zeroes_disabled_only():
    c = make_cohort(n_train=4, n_val=2, n_test=2, seed=1)
    only_i = c.train.bundle.with_enabled_concepts(("instrument",))
    mask = only_i.fusion_mask()
    assert (mask[:, 0] == 1).all()
    assert (mask[:, 1:] == 0).all()
    assert TokenAssembler()(only_i).shape == (4, 4, 64)


def test_jobs_include_required_matrix():
    ids = {s.experiment_id for s in experiment_specs()}
    for required in ("B1", "C-I", "F-Concat", "F-Gated", "F-Hidden", "F-Shortcut"):
        assert required in ids
    assert len(jobs_for(quick=True)) == len(experiment_specs())
    assert len(jobs_for(quick=False)) > len(jobs_for(quick=True))


def test_thresholds_fit_on_validation_only():
    c = make_cohort(n_train=8, n_val=8, n_test=8, seed=2)
    spec = next(s for s in experiment_specs() if s.experiment_id == "F-Gated")
    rec = train_one(spec, 0, c, PipelineConfig(out_dir=Path("results/proposed/mock_test"), steps=3, batch_size=4))
    assert rec.split_used_for_selection == "validation"
    assert rec.extra["threshold_split"] == "validation"
    assert rec.extra["test_song_ids"] == c.test.song_ids
    # Fitting on test would use a different y; we only assert the recorded split.
    _ = fit_thresholds
    assert CONCEPT_ORDER[0] == "instrument"


def test_lambda_rhythm_is_configurable_and_recorded(tmp_path):
    cohort = make_cohort(n_train=4, n_val=4, n_test=4, seed=4)
    spec = next(s for s in experiment_specs() if s.experiment_id == "C-R")
    rec = train_one(
        spec,
        0,
        cohort,
        PipelineConfig(
            out_dir=tmp_path,
            steps=1,
            batch_size=4,
            lambda_rhythm=0.25,
            warmup=0,
            infer_steps=1,
        ),
    )
    assert rec.config["lambda_rhythm"] == 0.25
    assert rec.metrics["lambda_rhythm"] == 0.25


def test_run_all_quick(tmp_path):
    recs = run_all(quick=True, out_dir=tmp_path, steps=2)
    ids = {r.experiment_id for r in recs}
    assert "F-Gated" in ids and "B1" in ids and "C-I" in ids
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "cohort_ids.json").exists()
    assert (tmp_path / "faithfulness.json").exists()
    gated = next(r for r in recs if r.experiment_id == "F-Gated")
    assert gated.test_evaluated
    assert gated.metrics["dropout_was_trained"] is True
