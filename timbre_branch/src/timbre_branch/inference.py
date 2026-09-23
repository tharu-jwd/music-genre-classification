"""Inference helpers that retain standardized fusion output and readable units."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .constants import FEATURE_COLUMNS
from .model import TimbreBranch
from .preprocessing import TimbreStandardizer


@torch.no_grad()
def predict_timbre_concepts(
    model: TimbreBranch,
    h_audio: Tensor,
    standardizer: TimbreStandardizer,
) -> dict[str, Any]:
    """Return the 35-D fusion tensor and its interpretable original-unit view."""
    model.eval()
    standardized = model(h_audio)
    original = standardizer.inverse_transform(
        standardized.detach().cpu().numpy().astype(np.float64)
    )
    return {
        "z_timbre": standardized,
        "d_hat_standardized": standardized,
        "d_hat_original_units": original,
        "feature_names": FEATURE_COLUMNS,
    }
