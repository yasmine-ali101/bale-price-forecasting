"""Hyperparameter search for the gradient-boosted models.

    python scripts/tune.py

Searches on the **validation** split only. The test split is never touched here,
because selecting a configuration by its test score is how a model comes to look
excellent in a README and fail in production: the reported number stops being an
estimate of unseen performance and becomes a record of how many configurations
were tried.

Writes results/tuning.json with the winning parameters, which run_experiment.py
then evaluates once on test.
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bales import data, evaluation, features, models, preprocessing  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tune")

RESULTS = ROOT / "results"
TARGET = features.TARGET
HORIZON = features.HORIZON

GRID = {
    "n_estimators": [300, 600, 1000],
    "learning_rate": [0.01, 0.03, 0.08],
    "max_depth": [2, 3, 5],
    "min_child_weight": [3, 8, 15],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1.0, 3.0, 10.0],
}


def sampled_grid(limit: int = 60, seed: int = 42):
    """A random sample of the full grid.

    The full product is 1,458 configurations. On a 1,000-row series that is
    mostly a way to overfit the validation split, so a random sample of 60 gives
    nearly the same best-case result for a fraction of the search.
    """
    import random

    rng = random.Random(seed)
    keys = list(GRID)
    combinations = list(itertools.product(*(GRID[k] for k in keys)))
    rng.shuffle(combinations)
    for values in combinations[:limit]:
        yield dict(zip(keys, values))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    transactions, exchange_rates = data.load(ROOT / "data")
    daily = preprocessing.build_daily_series(transactions, exchange_rates)
    frame = features.build_features(daily)
    split = evaluation.chronological_split(frame)
    feature_cols = features.feature_columns(frame)

    anchor = split.validation[f"lag_{HORIZON}"].to_numpy()
    y_validation = split.validation[TARGET].to_numpy()

    # Baselines on the same validation split, so tuning is aimed at a real bar.
    baselines = {}
    for factory in (models.NaiveLastValue, models.SeasonalNaive, models.RollingMean,
                    models.RidgeForecaster):
        model = factory()
        model.fit(split.train, TARGET, feature_cols)
        scores = evaluation.score(
            y_validation, model.predict(split.validation, TARGET, feature_cols), anchor
        )
        baselines[model.name] = scores
        logger.info("baseline %-28s val MAE=%.4f MAPE=%.2f%%",
                    model.name, scores["MAE"], scores["MAPE"])

    best_baseline = min(baselines.items(), key=lambda kv: kv[1]["MAE"])
    logger.info("best baseline on validation: %s at MAE %.4f",
                best_baseline[0], best_baseline[1]["MAE"])

    trials = []
    for i, params in enumerate(sampled_grid(), 1):
        for target_mode, factory in (("level", models.XGBoostForecaster),
                                     ("delta", models.XGBoostDelta)):
            model = factory()
            model.params = {**model.params, **params}
            model.fit(split.train, TARGET, feature_cols)
            scores = evaluation.score(
                y_validation, model.predict(split.validation, TARGET, feature_cols), anchor
            )
            trials.append({"target": target_mode, "params": params, **scores})
        if i % 15 == 0:
            logger.info("  %d/60 configurations evaluated", i)

    trials.sort(key=lambda t: t["MAE"])
    best = trials[0]
    logger.info("best tuned: target=%s val MAE=%.4f MAPE=%.2f%% DirAcc=%.1f%%",
                best["target"], best["MAE"], best["MAPE"], best["DirAcc"])
    logger.info("best params: %s", best["params"])

    beat = best["MAE"] < best_baseline[1]["MAE"]
    logger.info("tuned model beats best baseline on validation: %s", beat)

    report = {
        "note": (
            "Selected on the validation split only. The test split is scored once, "
            "in run_experiment.py, using these parameters."
        ),
        "configurations_tried": len(trials),
        "baselines_on_validation": {k: v for k, v in baselines.items()},
        "best_baseline": {"name": best_baseline[0], **best_baseline[1]},
        "best": best,
        "top_10": trials[:10],
        "tuned_beats_baseline_on_validation": beat,
    }
    (RESULTS / "tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS / "tuning.json")


if __name__ == "__main__":
    main()
