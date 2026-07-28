"""Forecasting models for the daily bale price.

Includes a faithful reproduction of the original notebook's configuration
(`NotebookBaseline`) so the effect of each fix is measurable rather than asserted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Forecaster:
    name = "base"

    def fit(self, train: pd.DataFrame, target: str, features: list[str], **kwargs):
        raise NotImplementedError

    def predict(self, test: pd.DataFrame, target: str, features: list[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NaiveLastValue(Forecaster):
    """Random walk: tomorrow's price is today's price.

    For a commodity price this is the benchmark that matters. Financial series
    are close to random walks, and beating this one is genuinely hard — a model
    that cannot is telling you the series has no exploitable structure at this
    horizon.
    """

    horizon: int = 7
    name: str = "Naive (random walk)"

    def fit(self, train, target, features, **kwargs):
        self._fallback = float(train[target].iloc[-1])
        return self

    def predict(self, test, target, features):
        column = f"lag_{self.horizon}"
        if column in test:
            return test[column].fillna(self._fallback).to_numpy()
        return np.full(len(test), self._fallback)


@dataclass
class SeasonalNaive(Forecaster):
    """Predict the value from four weeks ago."""

    period: int = 28
    name: str = "Seasonal naive (28d)"

    def fit(self, train, target, features, **kwargs):
        self._fallback = float(train[target].mean())
        return self

    def predict(self, test, target, features):
        return test[f"lag_{self.period}"].fillna(self._fallback).to_numpy()


@dataclass
class RollingMean(Forecaster):
    """Mean of the last 14 observed days — a smoothed random walk."""

    window: int = 14
    name: str = "Rolling mean (14d)"

    def fit(self, train, target, features, **kwargs):
        self._fallback = float(train[target].mean())
        return self

    def predict(self, test, target, features):
        return test[f"roll_mean_{self.window}"].fillna(self._fallback).to_numpy()


@dataclass
class NotebookBaseline(Forecaster):
    """The original notebook's model, reproduced exactly.

    Calendar-only features (including `Year`) and `learning_rate=1e-6` with 100
    estimators. At that learning rate the ensemble moves ~0.01% of the residual
    per tree, so after 100 trees it has barely departed from its initial
    prediction. Reproduced here to quantify the gap rather than assert it.
    """

    name: str = "Notebook baseline (calendar only, lr=1e-6)"
    _model: object | None = None
    _features: list[str] = field(default_factory=list)

    CALENDAR_ONLY = ["day", "month", "year", "dayofyear", "quarter", "weekofyear"]

    def fit(self, train, target, features, **kwargs):
        from xgboost import XGBRegressor

        frame = train.copy()
        frame["year"] = frame.index.year
        self._features = [c for c in self.CALENDAR_ONLY if c in frame.columns]
        self._model = XGBRegressor(
            n_estimators=100, learning_rate=0.000001, random_state=42
        )
        self._model.fit(frame[self._features], frame[target])
        return self

    def predict(self, test, target, features):
        frame = test.copy()
        frame["year"] = frame.index.year
        return np.asarray(self._model.predict(frame[self._features]), dtype=float)


@dataclass
class RidgeForecaster(Forecaster):
    """Regularised linear model on the same features the tree gets.

    Worth including because linear models extrapolate trends and trees do not —
    on a trending series that can matter more than model capacity.
    """

    alpha: float = 1.0
    name: str = "Ridge (calendar + lags)"
    _model: object | None = None

    def fit(self, train, target, features, **kwargs):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._model.fit(train[features], train[target])
        return self

    def predict(self, test, target, features):
        return np.asarray(self._model.predict(test[features]), dtype=float)


@dataclass
class XGBoostForecaster(Forecaster):
    """Gradient-boosted trees on calendar + lag + market features."""

    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 1.5,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    name: str = "XGBoost (calendar + lags)"
    _model: object | None = None

    def fit(self, train, target, features, eval_set: pd.DataFrame | None = None, **kwargs):
        from xgboost import XGBRegressor

        params = dict(self.params)
        if eval_set is not None and len(eval_set):
            params["early_stopping_rounds"] = 50
        self._model = XGBRegressor(**params)
        fit_kwargs = {}
        if eval_set is not None and len(eval_set):
            fit_kwargs = {"eval_set": [(eval_set[features], eval_set[target])], "verbose": False}
        self._model.fit(train[features], train[target], **fit_kwargs)
        return self

    def predict(self, test, target, features):
        return np.asarray(self._model.predict(test[features]), dtype=float)

    def importances(self, features: list[str]) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=features).sort_values(
            ascending=False
        )


@dataclass
class XGBoostDelta(Forecaster):
    """XGBoost predicting the *change* from the last observed price, not the level.

    The standard fix when a tree model underperforms a naive baseline on a
    trending series. Trees predict a weighted average of training leaf values, so
    they cannot output a level they never saw — on an upward-trending price they
    are structurally biased low.

    Reframing the target as `y - lag_h` removes the trend from what the model has
    to learn: the residual is roughly mean-zero and stationary, which is the
    regime trees handle well. The prediction is reconstructed by adding the last
    observed price back.
    """

    horizon: int = 7
    params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 8,
            "reg_lambda": 3.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    name: str = "XGBoost on price change (Δ target)"
    _model: object | None = None

    def _anchor(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[f"lag_{self.horizon}"].to_numpy()

    def fit(self, train, target, features, eval_set: pd.DataFrame | None = None, **kwargs):
        from xgboost import XGBRegressor

        params = dict(self.params)
        if eval_set is not None and len(eval_set):
            params["early_stopping_rounds"] = 50
        self._model = XGBRegressor(**params)

        y_train = train[target].to_numpy() - self._anchor(train)
        fit_kwargs = {}
        if eval_set is not None and len(eval_set):
            fit_kwargs = {
                "eval_set": [
                    (eval_set[features], eval_set[target].to_numpy() - self._anchor(eval_set))
                ],
                "verbose": False,
            }
        self._model.fit(train[features], y_train, **fit_kwargs)
        return self

    def predict(self, test, target, features):
        delta = np.asarray(self._model.predict(test[features]), dtype=float)
        return self._anchor(test) + delta

    def importances(self, features: list[str]) -> pd.Series:
        return pd.Series(self._model.feature_importances_, index=features).sort_values(
            ascending=False
        )


def default_models() -> list[Forecaster]:
    return [
        NotebookBaseline(),
        NaiveLastValue(),
        SeasonalNaive(),
        RollingMean(),
        RidgeForecaster(),
        XGBoostForecaster(),
        XGBoostDelta(),
    ]
