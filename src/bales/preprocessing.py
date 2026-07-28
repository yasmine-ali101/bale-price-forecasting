"""Transaction ledger -> clean daily USD price series.

The pipeline's one substantive economic decision lives here: prices are converted
to USD before anything else. Over 2021-2023 the EGP lost roughly half its value
in three discrete devaluations. A nominal-EGP series over that window is
dominated by the currency, and a model fitted to it will forecast the exchange
rate while appearing to forecast plastic.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_exchange_rates(
    transactions: pd.DataFrame, exchange_rates: pd.DataFrame
) -> pd.DataFrame:
    """Attach the official rate to each transaction.

    The FX sheet is published on business days only, so a plain join drops every
    weekend transaction. `merge_asof` carries the last published rate forward,
    which is what actually applies to a Saturday trade.
    """
    left = transactions.sort_values("Date")
    right = exchange_rates.sort_values("Date")
    merged = pd.merge_asof(left, right, on="Date", direction="backward")

    unmatched = int(merged["Sell"].isna().sum())
    if unmatched:
        # `merge_asof(direction="backward")` leaves NaN for any trade predating
        # the first published quote. There is nothing earlier to carry forward,
        # so fall back to the earliest rate in the FX sheet itself — filling from
        # the merged frame would not work, since those rows are NaN too.
        logger.info("Back-filling %d transaction(s) preceding the first FX quote", unmatched)
        first = right.iloc[0]
        merged["Sell"] = merged["Sell"].fillna(first["Sell"])
        merged["Buy"] = merged["Buy"].fillna(first["Buy"])
    return merged


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and rows with no usable price."""
    before = len(transactions)
    cleaned = transactions.drop_duplicates()
    deduped = before - len(cleaned)

    cleaned = cleaned.dropna(subset=["Price"])
    dropped = len(transactions.drop_duplicates()) - len(cleaned)

    logger.info("Removed %d duplicate row(s) and %d row(s) with no price", deduped, dropped)
    return cleaned.reset_index(drop=True)


def to_usd(merged: pd.DataFrame) -> pd.DataFrame:
    """Add the inflation-adjusted unit price."""
    out = merged.copy()
    out["Unit_Price_in_USD"] = out["Price"] / out["Sell"]
    return out


def to_daily_series(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into one row per calendar day.

    The daily price is **weight-weighted**, not a plain mean: a 400 kg lot and a
    30-tonne lot are not equally informative about the market price, and an
    unweighted mean lets tiny trades swing the series.
    """
    frame = transactions.copy()
    frame["_weighted"] = frame["Unit_Price_in_USD"] * frame["Accept"]

    daily = frame.groupby("Date").agg(
        weighted_sum=("_weighted", "sum"),
        accepted_weight=("Accept", "sum"),
        transactions=("Unit_Price_in_USD", "size"),
        mean_price_usd=("Unit_Price_in_USD", "mean"),
        total_weight=("Weight", "sum"),
        mean_deduction=("Deduction", "mean"),
        fx_sell=("Sell", "mean"),
        mean_price_egp=("Price", "mean"),
    )
    daily["Unit_Price_in_USD"] = daily["weighted_sum"] / daily["accepted_weight"]
    daily = daily.drop(columns=["weighted_sum"])

    # A continuous daily index; days with no trading are carried forward, since
    # the market price does not cease to exist on a quiet Friday.
    daily = daily.asfreq("D")
    gaps = int(daily["Unit_Price_in_USD"].isna().sum())
    if gaps:
        logger.info("Forward-filling %d non-trading day(s)", gaps)
    daily["Unit_Price_in_USD"] = daily["Unit_Price_in_USD"].ffill()
    daily["fx_sell"] = daily["fx_sell"].ffill()
    daily["transactions"] = daily["transactions"].fillna(0)

    return daily.dropna(subset=["Unit_Price_in_USD"])


def build_daily_series(
    transactions: pd.DataFrame, exchange_rates: pd.DataFrame
) -> pd.DataFrame:
    """Full path: raw ledger + FX sheet -> clean daily USD price series."""
    cleaned = clean_transactions(transactions)
    merged = merge_exchange_rates(cleaned, exchange_rates)
    priced = to_usd(merged)
    return to_daily_series(priced)
