"""Check combined remote-job ordering without submitting Modal jobs."""
from types import SimpleNamespace

import pytest

pytest.importorskip("modal")
import modal_app
import modal_paper


@pytest.mark.parametrize("completed", [False, True])
def test_combined_job_trains_before_test_and_reuses_completed_training(monkeypatch, completed):
    calls = []
    arguments = {"run_name": "paper-v2-full-s42", "skip_test": True}

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
        modal_app.train_and_evaluate_remote.local({"run_name": "paper-v2-full-s42"})


@pytest.mark.parametrize("state", ["missing", "complete", "partial", "evaluated"])
def test_single_run_is_submitted_durably_without_waiting(monkeypatch, capsys, state):
    from scripts.paper_artifacts import METRIC_VERSION
    from scripts.paper_experiments import load_manifest, selected_runs

    name = "paper-v2-cnn-s456"
    run = selected_runs(load_manifest("experiments/paper21-modal.json"), phase="all", run_id=name)[0]
    status = dict(state="complete" if state == "evaluated" else state,
                  config=run["config"], metric_version=METRIC_VERSION,
                  evaluations={"test": state == "evaluated"})
    submissions = []

    def statuses(names):
        assert names == [name]
        return {name: status}

    class DurableJob:
        def with_options(self, **kwargs):
            assert kwargs == {"gpu": "A10"}
            return self

        def remote(self, *args, **kwargs):
            pytest.fail("Training must not use a synchronous remote call")

        def spawn(self, arguments, train_completed):
            assert arguments["run_name"] == name
            assert arguments["seed"] == 456
            assert arguments["model"] == "cnn"
            assert arguments["skip_test"] is True
            assert train_completed == (state == "complete")
            submissions.append(arguments)
            return SimpleNamespace(object_id="fc-test-durable",
                                   get=lambda *args, **kwargs: pytest.fail("Submission must not wait for completion"))

    monkeypatch.setattr(modal_paper, "paper_status_remote", SimpleNamespace(remote=statuses))
    monkeypatch.setattr(modal_paper, "train_and_evaluate_remote", DurableJob())
    invoke = modal_paper.paper.info.raw_f
    if state == "partial":
        with pytest.raises(RuntimeError, match="Partial run exists"):
            invoke(action="run", phase="all", run_name=name, execute=True)
        assert not submissions
    else:
        invoke(action="run", phase="all", run_name=name, execute=True)
        assert len(submissions) == (0 if state == "evaluated" else 1)
        output = capsys.readouterr().out
        if state != "evaluated":
            assert '"status": "submitted"' in output
            assert '"function_call_id": "fc-test-durable"' in output
            assert '"results_volume": "music-genre-runs"' in output
