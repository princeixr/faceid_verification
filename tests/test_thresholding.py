from __future__ import annotations

import pandas as pd
import pytest

from src.config import Config
from src.thresholding import (
    apply_threshold,
    generate_threshold_grid,
    get_best_threshold,
    select_threshold,
)


def _cfg(range_vals=(0.3, 0.9), increment=0.2, default=0.7) -> Config:
    return Config.from_dict(
        {
            "paths": {"project_root": ".", "out_root": "outputs"},
            "similarity_threshold": {
                "range": list(range_vals),
                "increment": increment,
                "default": default,
            },
            "similarity": {"direction": "higher"},
        }
    )


def test_apply_threshold_higher_means_same() -> None:
    out = apply_threshold([0.1, 0.7, 0.8], 0.7, higher_means_same=True)
    assert out.tolist() == [0, 1, 1]


def test_apply_threshold_lower_means_same() -> None:
    out = apply_threshold([0.1, 0.7, 0.8], 0.7, higher_means_same=False)
    assert out.tolist() == [1, 1, 0]


def test_generate_threshold_grid_includes_end() -> None:
    grid = generate_threshold_grid(_cfg(range_vals=(0.3, 0.7), increment=0.2))
    assert grid == [0.3, 0.5, 0.7]


def test_generate_threshold_grid_invalid_increment_raises() -> None:
    with pytest.raises(ValueError):
        generate_threshold_grid(_cfg(increment=0.0))


def test_select_threshold_uses_balanced_accuracy() -> None:
    rows = [
        {"threshold": 0.5, "balanced_accuracy": 0.60, "accuracy": 0.70, "f1_score": 0.61},
        {"threshold": 0.6, "balanced_accuracy": 0.75, "accuracy": 0.68, "f1_score": 0.59},
    ]
    t = select_threshold(rows, "max_balanced_accuracy")
    assert t == 0.6


def test_get_best_threshold_returns_expected_rule_choice() -> None:
    # At threshold 0.5, predictions are perfect; at threshold 0.7 they are not.
    df = pd.DataFrame(
        {
            "similarity_score": [0.9, 0.8, 0.4, 0.3],
            "label": [1, 1, 0, 0],
        }
    )
    cfg = _cfg(range_vals=(0.5, 0.9), increment=0.2, default=0.7)
    best_t, best_metrics, sweep = get_best_threshold(df, cfg, rule="max_accuracy")
    assert best_t == 0.5
    assert best_metrics["accuracy"] == 1.0
    assert len(sweep) == 2
