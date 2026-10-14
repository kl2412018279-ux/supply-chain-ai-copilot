"""Train the LightGBM container-volume (TEU) forecasting model.

* Time-based holdout (no leakage): the last ``holdout_days`` are used for validation.
* Benchmarked against a seasonal-naive baseline (same weekday last week).
* Optional small hyper-parameter search, every trial logged to MLflow.
* The saved bundle contains the model *and* the recent history needed for
  recursive multi-day forecasts, so inference needs no database access.

Usage::

    python -m src.forecasting.train --data data/raw/container_volumes.csv --tune --mlflow
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import get_settings
from src.forecasting.data import FEATURES, MIN_HISTORY, build_features, load_volumes

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "n_estimators": 1000,
    "random_state": 42,
    "verbose": -1,
}
SEARCH_SPACE: dict[str, list[Any]] = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.08]}


@dataclass
class TrainResult:
    bundle: dict[str, Any]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    model_path: Path


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def _fit(params: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params)
    # LightGBM >= 4.7 renamed eval_set to eval_X/eval_y; support both.
    if "eval_X" in inspect.signature(model.fit).parameters:
        eval_kwargs = {"eval_X": (valid[FEATURES],), "eval_y": (valid["teu"],)}
    else:
        eval_kwargs = {"eval_set": [(valid[FEATURES], valid["teu"])]}
    model.fit(
        train[FEATURES],
        train["teu"],
        **eval_kwargs,
        eval_metric="l1",
        categorical_feature=["commodity_id"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _mlflow():
    try:
        import mlflow  # noqa: PLC0415

        return mlflow
    except ImportError:
        logger.warning("mlflow is not installed; skipping experiment tracking.")
        return None


def train(
    volumes: pd.DataFrame,
    model_path: str | Path | None = None,
    holdout_days: int = 56,
    tune: bool = False,
    use_mlflow: bool = False,
    params: dict[str, Any] | None = None,
) -> TrainResult:
    """Train, evaluate and persist the forecasting model bundle."""
    model_path = Path(model_path or get_settings().model_path)
    categories = sorted(volumes["commodity"].unique())
    feats = build_features(volumes, categories)

    cutoff = feats["date"].max() - pd.Timedelta(days=holdout_days)
    train_df, valid_df = feats[feats["date"] <= cutoff], feats[feats["date"] > cutoff]
    if train_df.empty or valid_df.empty:
        raise ValueError("Not enough history for the requested holdout window.")

    base_params = {**DEFAULT_PARAMS, **(params or {})}
    candidates = (
        [
            {**base_params, **dict(zip(SEARCH_SPACE, combo, strict=True))}
            for combo in itertools.product(*SEARCH_SPACE.values())
        ]
        if tune
        else [base_params]
    )

    mlflow = _mlflow() if use_mlflow else None
    if mlflow:
        uri = get_settings().mlflow_tracking_uri
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("container-volume-forecasting")
        mlflow.start_run(run_name=f"lgbm-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")

    try:
        best: tuple[float, dict[str, Any], lgb.LGBMRegressor] | None = None
        for trial_params in candidates:
            model = _fit(trial_params, train_df, valid_df)
            trial_metrics = regression_metrics(valid_df["teu"], model.predict(valid_df[FEATURES]))
            logger.info("trial %s -> %s", _short(trial_params), trial_metrics)
            if mlflow and tune:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(_short(trial_params))
                    mlflow.log_metrics({f"val_{k}": v for k, v in trial_metrics.items()})
            if best is None or trial_metrics["mae"] < best[0]:
                best = (trial_metrics["mae"], trial_params, model)

        assert best is not None
        _, best_params, best_model = best
        valid_pred = np.clip(best_model.predict(valid_df[FEATURES]), 0, None)
        metrics = regression_metrics(valid_df["teu"], valid_pred)
        baseline = regression_metrics(valid_df["teu"], valid_df["lag_7"])

        # Refit on all data with the tuned number of trees.
        final_params = {**best_params, "n_estimators": max(best_model.best_iteration_ or 100, 50)}
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(feats[FEATURES], feats["teu"], categorical_feature=["commodity_id"])

        history = {
            commodity: g.sort_values("date")["teu"]
            .to_numpy(dtype=float)[-MIN_HISTORY * 2 :]
            .tolist()
            for commodity, g in volumes.groupby("commodity")
        }
        last_dates = {
            commodity: g["date"].max().strftime("%Y-%m-%d")
            for commodity, g in volumes.groupby("commodity")
        }
        bundle = {
            "model": final_model,
            "features": FEATURES,
            "categories": categories,
            "history": history,
            "last_dates": last_dates,
            "metrics": metrics,
            "baseline_metrics": baseline,
            "params": _short(final_params),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(feats)),
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = model_path.with_suffix(f".{os.getpid()}.tmp")
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, model_path)  # atomic: readers never see a half-written file

        if mlflow:
            mlflow.log_params(_short(final_params))
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})
            mlflow.log_artifact(str(model_path))
    finally:
        if mlflow:
            mlflow.end_run()

    return TrainResult(bundle, metrics, baseline, model_path)


def _short(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in {"verbose", "objective"}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the LightGBM container-volume forecaster.")
    parser.add_argument("--data", default=str(settings.volumes_csv_path))
    parser.add_argument("--out", default=str(settings.model_path))
    parser.add_argument("--holdout-days", type=int, default=56)
    parser.add_argument("--tune", action="store_true", help="run a small grid search")
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    args = parser.parse_args()

    result = train(
        load_volumes(args.data),
        model_path=args.out,
        holdout_days=args.holdout_days,
        tune=args.tune,
        use_mlflow=args.mlflow,
    )
    m, b = result.metrics, result.baseline_metrics
    print(f"Model saved to {result.model_path}")
    print(f"  LightGBM       MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")
    print(f"  Seasonal naive MAE={b['mae']:.2f}  RMSE={b['rmse']:.2f}  R2={b['r2']:.3f}")


if __name__ == "__main__":
    main()
