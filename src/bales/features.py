"""Feature engineering for the daily bale-price series.

The original notebook trained on calendar features only, `Day`, `Month`, `Year`,
`Day_of_the_year`, `Quarter`, `week_of_year`. That cannot work for this target,
for a specific and instructive reason:

`Year` takes the value 2023 in the test set and never appears in training. A
decision tree splits on thresholds it has seen; asked to predict at `Year=2023`
it falls into the `Year <= 2022` leaf and returns a 2022-level price. Since the
series trends upward, every forecast is biased low, and no amount of tuning
fixes it, the information simply is not in the feature set.

Lags fix it by giving the model the recent *level* of the series instead of asking
it to infer level from a calendar label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Business horizon: a week ahead, for procurement planning.
HORIZON = 7

LAGS = (7, 8, 9, 14, 21, 28, 56)
ROLLING_WINDOWS = (7, 14, 30)

TARGET = "Unit_Price_in_USD"


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar parts plus cyclical encodings.

    `Year` is deliberately excluded, see the module docstring. `month` and
    `dayofyear` are kept because they carry genuine seasonality that repeats
    across years, unlike a monotonically increasing year label.
    """
    out = frame.copy()
    idx = out.index
    out["day"] = idx.day
    out["month"] = idx.month
    out["dayofweek"] = idx.dayofweek
    out["dayofyear"] = idx.dayofyear
    out["quarter"] = idx.quarter
    out["weekofyear"] = idx.isocalendar().week.astype(int)
    out["is_weekend"] = (idx.dayofweek >= 4).astype(int)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    return out


def add_lag_features(
    frame: pd.DataFrame,
    target: str = TARGET,
    lags: tuple[int, ...] = LAGS,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Lagged levels, rolling statistics, and momentum.

    Every lag must be >= `horizon`, or the feature would not exist at forecast
    time. `add_lag_features` refuses shorter ones rather than silently producing
    leaked metrics.
    """
    invalid = [lag for lag in lags if lag < horizon]
    if invalid:
        raise ValueError(
            f"Lags {invalid} are shorter than the {horizon}-day horizon and would leak "
            "future information."
        )

    out = frame.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)

    shifted = out[target].shift(horizon)
    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()
        out[f"roll_min_{window}"] = shifted.rolling(window).min()
        out[f"roll_max_{window}"] = shifted.rolling(window).max()

    # Momentum: recent direction, which a level-only feature set cannot express.
    # Derived from shifts of the target rather than from named lag columns, so
    # these stay valid whatever `lags` the caller asked for.
    anchor = out[target].shift(horizon)
    out["momentum_7"] = anchor - out[target].shift(horizon + 7)
    out["momentum_21"] = anchor - out[target].shift(horizon + 21)
    return out


def add_market_features(frame: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Lagged trading-activity and FX features.

    These are observed quantities, so they must be lagged by the horizon too, next week's trade count is no more knowable than next week's price.
    """
    out = frame.copy()
    if "transactions" in out:
        out["transactions_lag"] = out["transactions"].shift(horizon)
        out["transactions_roll_14"] = out["transactions"].shift(horizon).rolling(14).mean()
    if "fx_sell" in out:
        out["fx_lag"] = out["fx_sell"].shift(horizon)
        # Rate of devaluation, the pressure on nominal prices.
        out["fx_change_28"] = out["fx_sell"].shift(horizon) - out["fx_sell"].shift(horizon + 28)
    return out


def build_features(frame: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Run the full pipeline and drop rows without complete history."""
    out = add_calendar_features(frame)
    out = add_lag_features(out, target=target)
    out = add_market_features(out)
    derived = [c for c in out.columns if c.startswith(("lag_", "roll_", "momentum_", "fx_", "transactions_"))]
    return out.dropna(subset=[target, *derived])


def feature_columns(frame: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Model inputs, excluding the target and any same-day observed quantity.

    `mean_price_usd`, `mean_price_egp`, `fx_sell` and friends are all measured on
    the day being predicted, so none of them can be inputs.
    """
    leaky = {
        target,
        "mean_price_usd",
        "mean_price_egp",
        "fx_sell",
        "accepted_weight",
        "total_weight",
        "mean_deduction",
        "transactions",
    }
    return [c for c in frame.columns if c not in leaky]
