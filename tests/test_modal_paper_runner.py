"""Check combined remote-job ordering without submitting Modal jobs."""
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")
import modal_app


@pytest.mark.parametrize("completed", [False, True])
def test_combined_job_trains_before_test_and_reuses_completed_training(monkeypatch, completed):
    calls = []
    arguments = {"run_name": "paper-v2-full-s17", "skip_test": True}

    def training(**kwargs):
        assert kwargs == arguments
        calls.append("train-validation")
        return {"best_epoch": 4}

    def evaluation(name, split):
        assert name == arguments["run_name"] and split == "test"
        calls.append("test")
        return {"metrics": {"macro_ap": 0.5}}

    monkeypatch.setattr(modal_app, "train_remote", SimpleNamespace(local=training))
    monkeypatch.setattr(modal_app, "evaluate_remote", SimpleNamespace(local=evaluation))
    result = modal_app.train_and_evaluate_remote.local(arguments, train_completed=completed)
    assert calls == (["test"] if completed else ["train-validation", "test"])
    assert result["test"]["metrics"]["macro_ap"] == 0.5
    assert (result["training"] is None) == completed


def test_combined_job_does_not_evaluate_after_training_failure(monkeypatch):
    def training(**kwargs):
        raise RuntimeError("Training failed")

    def evaluation(*args):
        pytest.fail("Test must not run after failed training")

    monkeypatch.setattr(modal_app, "train_remote", SimpleNamespace(local=training))
    monkeypatch.setattr(modal_app, "evaluate_remote", SimpleNamespace(local=evaluation))
    with pytest.raises(RuntimeError, match="Training failed"):
        modal_app.train_and_evaluate_remote.local({"run_name": "paper-v2-full-s17"})
