"""End-to-end experiment: ledger + FX -> daily USD series -> models -> metrics.

    python scripts/run_experiment.py

Writes results/metrics.json, results/metrics.md and the plots the README embeds.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bales import data, evaluation, features, models, preprocessing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiment")

RESULTS = ROOT / "results"
TARGET = features.TARGET
HORIZON = features.HORIZON


def prepare() -> tuple[pd.DataFrame, dict]:
    transactions, exchange_rates = data.load(ROOT / "data")
    stats = {
        "transaction_rows": int(len(transactions)),
        "duplicate_rows": int(transactions.duplicated().sum()),
        "rows_missing_price": int(transactions["Price"].isna().sum()),
        "fx_rows": int(len(exchange_rates)),
        "fx_min": float(exchange_rates["Sell"].min()),
        "fx_max": float(exchange_rates["Sell"].max()),
    }
    stats["egp_devaluation_pct"] = round(
        (stats["fx_max"] / stats["fx_min"] - 1) * 100, 1
    )

    daily = preprocessing.build_daily_series(transactions, exchange_rates)
    stats["daily_rows"] = int(len(daily))
    stats["period"] = f"{daily.index.min().date()} → {daily.index.max().date()}"
    stats["usd_price_min"] = round(float(daily[TARGET].min()), 4)
    stats["usd_price_max"] = round(float(daily[TARGET].max()), 4)
    stats["egp_price_min"] = round(float(daily["mean_price_egp"].min()), 3)
    stats["egp_price_max"] = round(float(daily["mean_price_egp"].max()), 3)

    engineered = features.build_features(daily)
    stats["modelling_rows"] = int(len(engineered))
    return engineered, stats, daily


def _format_direction(value: float) -> str:
    """Render directional accuracy, showing `n/a` for models that never call a move."""
    return "n/a" if value != value else f"{value:.1f}%"  # NaN != NaN


def evaluate_all(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame, object]:
    split = evaluation.chronological_split(frame)
    feature_cols = features.feature_columns(frame)

    results: dict[str, dict] = {}
    predictions = pd.DataFrame(index=split.test.index)
    predictions["actual"] = split.test[TARGET].to_numpy()
    last_known = split.test[f"lag_{HORIZON}"].to_numpy()
    fitted_xgb = None

    for model in models.default_models():
        started = time.perf_counter()
        try:
            if isinstance(model, (models.XGBoostForecaster, models.XGBoostDelta)):
                model.fit(split.train, TARGET, feature_cols, eval_set=split.validation)
                fitted_xgb = model
            else:
                model.fit(split.train, TARGET, feature_cols)
            y_pred = model.predict(split.test, TARGET, feature_cols)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model %s failed", model.name)
            results[model.name] = {"error": str(exc)}
            continue

        elapsed = time.perf_counter() - started
        metrics = evaluation.score(split.test[TARGET].to_numpy(), y_pred, last_known)
        metrics["fit_predict_seconds"] = round(elapsed, 2)
        results[model.name] = metrics
        predictions[model.name] = y_pred
        logger.info(
            "%-46s MAE=%.4f RMSE=%.4f MAPE=%5.2f%% DirAcc=%s",
            model.name, metrics["MAE"], metrics["RMSE"], metrics["MAPE"],
            _format_direction(metrics["DirAcc"]),
        )

    report = {
        "protocol": {
            "horizon_days": HORIZON,
            "description": (
                f"{HORIZON}-day-ahead forecasts. All features are lagged by at least "
                f"{HORIZON} days, so every input is observable at forecast time."
            ),
        },
        "split": split.describe(),
        "test_metrics": results,
    }
    return report, predictions, fitted_xgb


def backtest(frame: pd.DataFrame, n_folds: int = 5) -> dict:
    feature_cols = features.feature_columns(frame)
    folds = evaluation.rolling_origin_folds(frame, n_folds=n_folds, horizon_days=30)
    out: dict[str, list[dict]] = {}

    for factory in (models.NaiveLastValue, models.RollingMean, models.RidgeForecaster, models.XGBoostDelta):
        per_fold = []
        for i, (train, test) in enumerate(folds, 1):
            model = factory()
            model.fit(train, TARGET, feature_cols)
            metrics = evaluation.score(
                test[TARGET].to_numpy(),
                model.predict(test, TARGET, feature_cols),
                test[f"lag_{HORIZON}"].to_numpy(),
            )
            metrics["fold"] = i
            per_fold.append(metrics)
        out[factory().name] = per_fold

    summary = {
        name: {
            metric: round(sum(f[metric] for f in folds_) / len(folds_), 4)
            for metric in ("MAE", "RMSE", "MAPE", "DirAcc")
        }
        for name, folds_ in out.items()
    }
    return {"folds": out, "mean": summary}


def make_plots(frame, daily, predictions, xgb) -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Why USD normalisation matters: EGP price vs USD price vs FX rate.
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(daily.index, daily["mean_price_egp"], color="#dc2626")
    axes[0].set_title("Nominal bale price (EGP/kg) — dominated by currency devaluation")
    axes[0].set_ylabel("EGP/kg")
    axes[1].plot(daily.index, daily["fx_sell"], color="#7c3aed")
    axes[1].set_title("Official EGP/USD exchange rate")
    axes[1].set_ylabel("EGP per USD")
    axes[2].plot(daily.index, daily[TARGET], color="#059669")
    axes[2].set_title("Real bale price (USD/kg) — the actual modelling target")
    axes[2].set_ylabel("USD/kg")
    fig.tight_layout()
    fig.savefig(RESULTS / "why_usd_normalisation.png", dpi=140)
    plt.close(fig)

    # 2. Actual vs predicted on the test set.
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(predictions.index, predictions["actual"], label="Actual", color="#111827", linewidth=2)
    for name, colour in (
        ("XGBoost on price change (Δ target)", "#2563eb"),
        ("Rolling mean (14d)", "#059669"),
        ("Notebook baseline (calendar only, lr=1e-6)", "#dc2626"),
    ):
        if name in predictions:
            ax.plot(predictions.index, predictions[name], label=name, linewidth=1.3,
                    alpha=0.85, color=colour)
    ax.set_title(f"Bale price: actual vs {HORIZON}-day-ahead forecast (test set)")
    ax.set_ylabel("USD/kg")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_vs_actual.png", dpi=140)
    plt.close(fig)

    # 3. Feature importance.
    if xgb is not None:
        importances = xgb.importances(features.feature_columns(frame)).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.sort_values().plot.barh(ax=ax, color="#2563eb")
        ax.set_title("XGBoost feature importance (top 15)")
        fig.tight_layout()
        fig.savefig(RESULTS / "feature_importance.png", dpi=140)
        plt.close(fig)


def write_markdown(report: dict) -> None:
    d = report["dataset"]
    lines = [
        "# Experiment results", "", "_Generated by `scripts/run_experiment.py`._", "",
        "## Dataset", "", "| Property | Value |", "|---|---|",
        f"| Transaction rows | {d['transaction_rows']:,} |",
        f"| Duplicate rows removed | {d['duplicate_rows']:,} |",
        f"| Rows with missing price | {d['rows_missing_price']:,} |",
        f"| Daily series length | {d['daily_rows']:,} |",
        f"| Rows available for modelling | {d['modelling_rows']:,} |",
        f"| Period | {d['period']} |",
        f"| EGP/USD range | {d['fx_min']:.2f} → {d['fx_max']:.2f} ({d['egp_devaluation_pct']}% devaluation) |",
        f"| Nominal price range (EGP/kg) | {d['egp_price_min']:.2f} → {d['egp_price_max']:.2f} |",
        f"| Real price range (USD/kg) | {d['usd_price_min']:.3f} → {d['usd_price_max']:.3f} |",
        "", "## Hold-out test results", "",
        f"_{report['evaluation']['protocol']['description']}_", "",
        "| Model | MAE | RMSE | MAPE | Directional acc. |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in report["evaluation"]["test_metrics"].items():
        if "error" in m:
            lines.append(f"| {name} | — | — | — | failed |")
            continue
        direction = 'n/a' if m['DirAcc'] != m['DirAcc'] else f"{m['DirAcc']:.1f}%"
        lines.append(
            f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['MAPE']:.2f}% | {direction} |"
        )

    lines += ["", "## Expanding-window backtest (5 folds, mean)", "",
              "| Model | MAE | RMSE | MAPE | Directional acc. |", "|---|---:|---:|---:|---:|"]
    for name, m in report["backtest"]["mean"].items():
        direction = 'n/a' if m['DirAcc'] != m['DirAcc'] else f"{m['DirAcc']:.1f}%"
        lines.append(
            f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['MAPE']:.2f}% | {direction} |"
        )

    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    frame, dataset_stats, daily = prepare()
    evaluation_report, predictions, xgb = evaluate_all(frame)
    backtest_report = backtest(frame)
    make_plots(frame, daily, predictions, xgb)

    report = {
        "dataset": dataset_stats,
        "evaluation": evaluation_report,
        "backtest": backtest_report,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    predictions.to_csv(RESULTS / "test_predictions.csv")
    logger.info("Wrote results to %s", RESULTS)


if __name__ == "__main__":
    main()
