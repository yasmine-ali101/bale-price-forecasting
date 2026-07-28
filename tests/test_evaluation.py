"""Tests for metrics and splitting."""

import numpy as np
import pandas as pd
import pytest

from bales import evaluation


def test_perfect_forecast_scores_zero_error():
    y = np.array([1.0, 2.0, 3.0])

    result = evaluation.score(y, y)

    assert result["MAE"] == pytest.approx(0.0)
    assert result["MAPE"] == pytest.approx(0.0)


def test_directional_accuracy_rewards_getting_the_move_right():
    last_known = np.array([1.0, 1.0, 1.0, 1.0])
    y_true = np.array([1.1, 0.9, 1.2, 0.8])       # up, down, up, down
    y_pred = np.array([1.05, 0.95, 0.9, 1.3])     # up, down, down, up -> 2/4

    result = evaluation.directional_accuracy(y_true, y_pred, last_known)

    assert result == pytest.approx(50.0)


def test_directional_accuracy_is_undefined_when_the_model_never_calls_a_move():
    """The random walk predicts exactly the last value, so it makes no call.

    Scoring that 0% would read as 'always wrong' rather than 'never answered'.
    """
    last_known = np.array([1.0, 1.0])
    y_true = np.array([1.1, 0.9])
    y_pred = last_known.copy()

    assert np.isnan(evaluation.directional_accuracy(y_true, y_pred, last_known))


def test_chronological_split_preserves_time_order():
    frame = pd.DataFrame(
        {"value": range(200)}, index=pd.date_range("2021-01-01", periods=200, freq="D")
    )

    split = evaluation.chronological_split(frame)

    assert split.train.index.max() < split.validation.index.min()
    assert split.validation.index.max() < split.test.index.min()


def test_rolling_origin_folds_never_train_on_future_data():
    frame = pd.DataFrame(
        {"value": range(500)}, index=pd.date_range("2021-01-01", periods=500, freq="D")
    )

    folds = evaluation.rolling_origin_folds(frame, n_folds=4, horizon_days=30)

    assert len(folds) == 4
    for train, test in folds:
        assert train.index.max() < test.index.min()
        assert len(test) == 30
