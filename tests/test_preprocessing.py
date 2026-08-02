"""Tests for the ledger -> daily USD series path."""

import numpy as np
import pandas as pd
import pytest

from bales import preprocessing


def transactions(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_duplicates_and_priceless_rows_are_dropped():
    frame = transactions(
        [
            {"Date": pd.Timestamp("2022-01-01"), "Price": 10.0},
            {"Date": pd.Timestamp("2022-01-01"), "Price": 10.0},  # exact duplicate
            {"Date": pd.Timestamp("2022-01-02"), "Price": np.nan},
            {"Date": pd.Timestamp("2022-01-03"), "Price": 12.0},
        ]
    )

    cleaned = preprocessing.clean_transactions(frame)

    assert len(cleaned) == 2
    assert cleaned["Price"].tolist() == [10.0, 12.0]


def test_weekend_trades_get_the_last_published_rate():
    """FX is published on business days; a Saturday trade uses Friday's rate.

    A plain inner join would silently discard every weekend transaction.
    """
    trades = transactions(
        [
            {"Date": pd.Timestamp("2022-01-07"), "Price": 30.0},  # Friday
            {"Date": pd.Timestamp("2022-01-08"), "Price": 30.0},  # Saturday
        ]
    )
    rates = pd.DataFrame(
        {"Date": [pd.Timestamp("2022-01-07")], "Buy": [15.6], "Sell": [15.7]}
    )

    merged = preprocessing.merge_exchange_rates(trades, rates)

    assert len(merged) == 2
    assert merged["Sell"].tolist() == [15.7, 15.7]


def test_rates_are_backfilled_for_trades_predating_the_first_quote():
    trades = transactions([{"Date": pd.Timestamp("2022-01-01"), "Price": 30.0}])
    rates = pd.DataFrame(
        {"Date": [pd.Timestamp("2022-01-05")], "Buy": [15.6], "Sell": [15.7]}
    )

    merged = preprocessing.merge_exchange_rates(trades, rates)

    assert merged["Sell"].iloc[0] == 15.7


def test_usd_conversion_divides_by_the_sell_rate():
    merged = pd.DataFrame({"Price": [31.4], "Sell": [15.7]})

    result = preprocessing.to_usd(merged)

    assert result["Unit_Price_in_USD"].iloc[0] == pytest.approx(2.0)


def test_daily_price_is_weighted_by_accepted_weight():
    """A 30-tonne lot must move the daily price more than a 100 kg lot."""
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2022-01-01")] * 2,
            "Unit_Price_in_USD": [1.0, 2.0],
            "Accept": [100.0, 30000.0],
            "Weight": [100.0, 30000.0],
            "Deduction": [0.0, 0.0],
            "Sell": [15.7, 15.7],
            "Price": [15.7, 31.4],
        }
    )

    daily = preprocessing.to_daily_series(frame)

    expected = (1.0 * 100 + 2.0 * 30000) / 30100
    assert daily["Unit_Price_in_USD"].iloc[0] == pytest.approx(expected)
    # An unweighted mean would have given 1.5, far from the market truth.
    assert daily["Unit_Price_in_USD"].iloc[0] > 1.9


def test_non_trading_days_are_forward_filled_onto_a_continuous_index():
    frame = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-04")],
            "Unit_Price_in_USD": [1.0, 2.0],
            "Accept": [100.0, 100.0],
            "Weight": [100.0, 100.0],
            "Deduction": [0.0, 0.0],
            "Sell": [15.7, 15.7],
            "Price": [15.7, 31.4],
        }
    )

    daily = preprocessing.to_daily_series(frame)

    assert len(daily) == 4
    assert daily["Unit_Price_in_USD"].tolist() == [1.0, 1.0, 1.0, 2.0]
    # Days with no trading record zero transactions, not a missing value.
    assert daily["transactions"].tolist() == [1.0, 0.0, 0.0, 1.0]
