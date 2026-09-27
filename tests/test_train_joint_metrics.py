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
