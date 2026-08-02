"""Reproducible sample-data generator for recycled-plastic bale transactions.

The original study used a private ledger from a plastics recycler (2021-2023,
three annual sheets) plus official EGP/USD exchange rates. Neither can be
redistributed, so this module synthesises both with the same schema and the same
economics.

The economics matter more than the schema here. Egypt devalued the pound heavily
across the study window, roughly 15.7 to 30+ EGP/USD in two large steps, while
local bale prices rose in nominal terms. A model trained on nominal EGP prices
is mostly learning the devaluation, not the commodity. Converting to USD is what
separates the real price signal from the currency collapse, and it is the central
idea the pipeline is built around.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["PET Clear", "PET Blue", "PET Green", "PET Mixed"]
SUPPLY_AREAS = ["Cairo", "Giza", "Alexandria", "Qalyubia", "Sharqia", "Dakahlia"]
BULKING_STATIONS = ["BS-North", "BS-South", "BS-East", "BS-West", "BS-Central"]

# Approximate official EGP/USD rate at each devaluation step.
FX_STEPS = [
    ("2021-01-01", 15.70),
    ("2022-03-21", 18.30),   # first major devaluation
    ("2022-10-27", 24.10),   # second
    ("2023-01-11", 29.70),   # third
    ("2023-12-31", 30.90),
]


@dataclass(frozen=True)
class GeneratorConfig:
    start: str = "2021-01-01"
    end: str = "2023-12-31"
    seed: int = 20210101
    mean_daily_transactions: float = 9.0


def _build_fx_curve(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """Piecewise EGP/USD rate: flat plateaus, sharp steps, small daily jitter."""
    anchors = pd.Series(
        {pd.Timestamp(date): rate for date, rate in FX_STEPS}, dtype=float
    ).sort_index()
    curve = anchors.reindex(anchors.index.union(index)).ffill().bfill()
    curve = curve.reindex(index)
    # Central banks hold a peg then jump; a little noise keeps it from being a step function.
    jitter = 1 + rng.normal(0, 0.0015, len(index)).cumsum() * 0.02
    return curve * jitter


def generate(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(transactions, exchange_rates)`."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    days = pd.date_range(cfg.start, cfg.end, freq="D")
    fx = _build_fx_curve(days, rng)

    # True USD price: slow upward drift, annual seasonality (collection is
    # weather- and holiday-driven), and a persistent AR(1) wander.
    t = np.arange(len(days))
    trend = 0.42 + 0.10 * (t / len(days))
    seasonal = 1 + 0.09 * np.sin(2 * np.pi * (days.dayofyear - 60) / 365.25)

    shocks = rng.normal(0, 0.011, len(days))
    wander = np.zeros(len(days))
    for i in range(1, len(days)):
        wander[i] = 0.94 * wander[i - 1] + shocks[i]

    usd_price = np.clip(trend * seasonal * (1 + wander), 0.12, None)
    egp_price = usd_price * fx.to_numpy()

    records = []
    for i, day in enumerate(days):
        # Fridays are light; the yard is quieter.
        lam = cfg.mean_daily_transactions * (0.35 if day.dayofweek == 4 else 1.0)
        for _ in range(rng.poisson(lam)):
            category = rng.choice(CATEGORIES)
            # Clear PET commands a premium; mixed is discounted.
            premium = {"PET Clear": 1.12, "PET Blue": 1.00, "PET Green": 0.93, "PET Mixed": 0.82}[
                category
            ]
            price = egp_price[i] * premium * (1 + rng.normal(0, 0.045))
            weight = float(np.clip(rng.gamma(3.2, 2200), 400, 32000))
            deduction = float(np.clip(rng.beta(2, 28), 0, 0.35))
            accept = weight * (1 - deduction)
            reject = weight - accept
            cost = accept * price
            records.append(
                {
                    "Date": day,
                    "Supplier Name": f"SUP-{rng.integers(1, 60):03d}",
                    "CAT": category,
                    "Price": round(price, 3),
                    "Weight": round(weight, 1),
                    "Deduction": round(deduction, 4),
                    "Accept": round(accept, 1),
                    "Reject": round(reject, 1),
                    "Cost": round(cost, 2),
                    "Invoice": round(cost * (1 + rng.uniform(0.01, 0.05)), 2),
                    "Carta Series": f"C{rng.integers(10000, 99999)}",
                    "Supply Area": rng.choice(SUPPLY_AREAS),
                    "Bulking Station Name": rng.choice(BULKING_STATIONS),
                }
            )

    transactions = pd.DataFrame.from_records(records)

    # A few real-world blemishes the cleaning code has to survive.
    n = len(transactions)
    transactions.loc[rng.choice(n, size=int(n * 0.012), replace=False), "Price"] = np.nan
    duplicates = transactions.sample(int(n * 0.008), random_state=7)
    transactions = pd.concat([transactions, duplicates], ignore_index=True)
    transactions = transactions.sort_values("Date").reset_index(drop=True)

    exchange_rates = pd.DataFrame(
        {
            "Date": days,
            "Buy": np.round(fx.to_numpy() * 0.995, 4),
            "Sell": np.round(fx.to_numpy(), 4),
        }
    )
    # The official-rate sheet is only published on business days.
    exchange_rates = exchange_rates[exchange_rates["Date"].dt.dayofweek < 5].reset_index(drop=True)

    return transactions, exchange_rates


def write_sample(directory: str | Path = "data") -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    transactions, exchange_rates = generate()
    transactions_path = target / "pet_local_prices_sample.csv"
    fx_path = target / "official_exchange_rates_sample.csv"
    transactions.to_csv(transactions_path, index=False)
    exchange_rates.to_csv(fx_path, index=False)
    return transactions_path, fx_path


def load(directory: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both sheets, generating them if absent."""
    target = Path(directory)
    transactions_path = target / "pet_local_prices_sample.csv"
    fx_path = target / "official_exchange_rates_sample.csv"
    if not (transactions_path.exists() and fx_path.exists()):
        write_sample(target)
    return (
        pd.read_csv(transactions_path, parse_dates=["Date"]),
        pd.read_csv(fx_path, parse_dates=["Date"]),
    )


if __name__ == "__main__":
    paths = write_sample()
    for path in paths:
        print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
