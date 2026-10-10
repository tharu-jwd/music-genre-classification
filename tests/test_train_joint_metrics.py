import pytest
import torch

from scripts.train_joint import (
    _masked_regression_metrics,
    _multilabel_metrics,
)


def test_multilabel_metrics_report_per_tag_and_aggregate_scores():
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    probabilities = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])

    metrics = _multilabel_metrics(probabilities, targets, ("a", "b"))

    assert metrics["macro_average_precision"] == pytest.approx(1.0)
    assert metrics["micro_average_precision"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert metrics["binary_accuracy"] == pytest.approx(1.0)
    assert metrics["per_tag"]["a"]["support"] == 2


def test_regression_metrics_ignore_masked_values():
    targets = torch.tensor([[0.0, 1.0], [2.0, 99.0]])
    predictions = torch.tensor([[1.0, 1.0], [2.0, -99.0]])
    mask = torch.tensor([[True, True], [True, False]])

    metrics = _masked_regression_metrics(predictions, targets, mask, ("a", "b"))

    assert metrics["n_observed"] == 3
    assert metrics["mae_standardized"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse_standardized"] == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metrics["per_feature"]["b"]["mae"] == pytest.approx(0.0)


def test_tied_scores_match_sklearn_average_precision():
    from sklearn.metrics import average_precision_score
    truth = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    scores = torch.full_like(truth, 0.5)
    result = _multilabel_metrics(scores, truth, ("instrument",))
    assert result["macro_average_precision"] == pytest.approx(average_precision_score(truth[:, 0], scores[:, 0]))
    assert result["macro_average_precision"] == pytest.approx(0.5)


def test_unknown_instrument_cells_do_not_affect_any_metric():
    truth = torch.tensor([[1.0, float('nan')], [0.0, float('nan')], [float('nan'), float('nan')]])
    scores = torch.tensor([[0.9, 0.99], [0.1, 0.99], [0.99, 0.99]])
    mask = torch.isfinite(truth)
    result = _multilabel_metrics(scores, truth, ("known", "unknown"), mask=mask)
    assert result["macro_average_precision"] == result["micro_average_precision"] == 1.0
    assert result["macro_f1"] == result["micro_f1"] == result["binary_accuracy"] == 1.0
    assert result["n_observed"] == 2
    assert result["per_tag"]["unknown"]["f1"] is None
    assert result["per_tag"]["unknown"]["average_precision"] is None


def test_entirely_unknown_instrument_metrics_are_undefined():
    result = _multilabel_metrics(torch.ones(2, 1), torch.full((2, 1), float('nan')),
                                ("unknown",), mask=torch.zeros(2, 1))
    assert result["macro_average_precision"] is None
    assert result["macro_f1"] is None
    assert result["micro_f1"] is None
