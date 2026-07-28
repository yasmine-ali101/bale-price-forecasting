"""Metrics and chronological splitting for the price series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_known: np.ndarray) -> float:
    """Share of periods where the forecast got the *direction* of the move right.

    For a procurement decision this can matter more than absolute error: buying
    ahead of a rise is valuable even if the magnitude is off, and a model with
    excellent MAE that systematically calls the direction wrong is worse than
    useless.
    """
    actual_move = np.sign(y_true - last_known)
    predicted_move = np.sign(y_pred - last_known)

    # A model that predicts exactly the last known value (the random walk) never
    # calls a direction at all. Scoring that as 0% would read as "always wrong"
    # when the truth is "never answered" — so it is undefined, not zero.
    if not (predicted_move != 0).any():
        return float("nan")

    scored = actual_move != 0
    if not scored.any():
        return float("nan")
    return float(np.mean(actual_move[scored] == predicted_move[scored]) * 100)


def score(y_true, y_pred, last_known=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    result = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if last_known is not None:
        result["DirAcc"] = directional_accuracy(
            y_true, y_pred, np.asarray(last_known, dtype=float)
        )
    return result


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> dict:
        return {
            name: {
                "rows": len(frame),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
            }
            for name, frame in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


def chronological_split(
    frame: pd.DataFrame, validation_fraction: float = 0.15, test_fraction: float = 0.20
) -> Split:
    n = len(frame)
    n_test = int(n * test_fraction)
    n_validation = int(n * validation_fraction)
    n_train = n - n_validation - n_test
    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("Series is too short for the requested split fractions.")
    return Split(
        frame.iloc[:n_train],
        frame.iloc[n_train : n_train + n_validation],
        frame.iloc[n_train + n_validation :],
    )


def rolling_origin_folds(
    frame: pd.DataFrame, n_folds: int = 5, horizon_days: int = 30
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window backtest folds."""
    total = len(frame)
    first_cut = total - n_folds * horizon_days
    if first_cut <= 0:
        raise ValueError("Not enough history for the requested number of folds.")
    return [
        (frame.iloc[: first_cut + i * horizon_days],
         frame.iloc[first_cut + i * horizon_days : first_cut + (i + 1) * horizon_days])
        for i in range(n_folds)
    ]
