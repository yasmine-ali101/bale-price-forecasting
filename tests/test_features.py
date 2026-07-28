"""Tests for feature engineering — principally the leakage guards."""

import numpy as np
import pandas as pd
import pytest

from bales import features


def daily_frame(n: int = 400) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Unit_Price_in_USD": np.linspace(0.40, 0.55, n),
            "transactions": np.full(n, 9.0),
            "fx_sell": np.linspace(15.7, 30.9, n),
            "mean_price_egp": np.linspace(6.3, 17.0, n),
            "mean_price_usd": np.linspace(0.40, 0.55, n),
            "accepted_weight": np.full(n, 50000.0),
            "total_weight": np.full(n, 52000.0),
            "mean_deduction": np.full(n, 0.06),
        },
        index=index,
    )


def test_lags_shorter_than_the_horizon_are_rejected():
    with pytest.raises(ValueError, match="leak future information"):
        features.add_lag_features(daily_frame(), lags=(3,), horizon=7)


def test_lag_columns_hold_the_value_from_exactly_that_many_days_earlier():
    frame = features.add_lag_features(daily_frame(), lags=(7,), windows=())

    assert frame["lag_7"].iloc[100] == pytest.approx(frame["Unit_Price_in_USD"].iloc[93])


def test_rolling_features_exclude_the_horizon_window():
    frame = daily_frame()

    result = features.add_lag_features(frame, lags=(7,), windows=(7,), horizon=7)

    expected = frame["Unit_Price_in_USD"].shift(7).rolling(7).mean().iloc[200]
    assert result["roll_mean_7"].iloc[200] == pytest.approx(expected)


def test_year_is_excluded_from_calendar_features():
    """The original notebook's fatal feature.

    `Year` is monotonic, so its test-set value never appears in training and a
    tree cannot split on it usefully — every forecast collapses to the last
    year it saw.
    """
    frame = features.add_calendar_features(daily_frame())

    assert "year" not in frame.columns
    assert "Year" not in frame.columns
    # Genuinely repeating seasonality is kept.
    assert "month" in frame.columns
    assert "dayofyear" in frame.columns


def test_market_features_are_lagged_not_contemporaneous():
    """Next week's trade count is no more knowable than next week's price."""
    frame = features.add_market_features(daily_frame(), horizon=7)

    assert frame["fx_lag"].iloc[100] == pytest.approx(frame["fx_sell"].iloc[93])


def test_feature_columns_exclude_every_same_day_observation():
    frame = features.build_features(daily_frame(600))

    columns = features.feature_columns(frame)

    for leaked in (
        "Unit_Price_in_USD", "mean_price_usd", "mean_price_egp",
        "fx_sell", "accepted_weight", "transactions",
    ):
        assert leaked not in columns
    assert "lag_7" in columns
    assert "fx_lag" in columns


def test_momentum_captures_direction_of_recent_change():
    frame = features.build_features(daily_frame(600))

    # The synthetic series rises monotonically, so momentum must be positive.
    assert (frame["momentum_7"] > 0).all()


def test_build_features_leaves_no_missing_values_in_derived_columns():
    frame = features.build_features(daily_frame(600))

    derived = [
        c for c in frame.columns
        if c.startswith(("lag_", "roll_", "momentum_", "fx_", "transactions_"))
    ]
    assert not frame[derived].isna().any().any()
