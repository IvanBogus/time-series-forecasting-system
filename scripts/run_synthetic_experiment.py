"""Run synthetic verification experiments for the forecasting pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import run_anomaly_detection
from src.evaluation import evaluate_forecast
from src.models.approximation import (
    select_exponential_moving_average_one_step_span,
    select_moving_average_one_step_window,
)
from src.models.deep_learning import select_mlp_window_size
from src.models.polynomial import select_local_polynomial_one_step_configuration
from src.synthetic import (
    SyntheticSeriesConfig,
    SyntheticTrendType,
    generate_synthetic_series,
    synthetic_train_test_split,
)
from src.visualization import save_synthetic_verification_plot


def _json_ready(value: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-serializable Python values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def run_single_synthetic_experiment(trend_type: SyntheticTrendType) -> dict[str, Any]:
    """Run anomaly cleaning and forecasting for one synthetic trend type."""
    figure_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / f"synthetic_{trend_type}_verification.png"
    )
    config = SyntheticSeriesConfig(
        trend_type=trend_type,
        random_seed={"linear": 101, "quadratic": 202, "exponential": 303}[trend_type],
    )
    data = generate_synthetic_series(config)

    anomaly_results = run_anomaly_detection(data, "value", rolling_window=7)
    primary_cleaned = anomaly_results["rolling_median"].cleaned_data
    verification_data = data.copy()
    verification_data["cleaned_value"] = primary_cleaned["value"].to_numpy()

    train_data, test_data = synthetic_train_test_split(verification_data, train_size=0.8)
    train_series = train_data["cleaned_value"]
    test_series = test_data["true_trend"]

    polynomial_result = select_local_polynomial_one_step_configuration(
        train_series,
        test_data["cleaned_value"],
        degrees=range(1, 5),
        windows=(30, 60, 90, "all"),
    )
    moving_average_result = select_moving_average_one_step_window(
        train_series,
        test_data["cleaned_value"],
        windows=(3, 7, 14, 21),
    )
    ema_result = select_exponential_moving_average_one_step_span(
        train_series,
        test_data["cleaned_value"],
        spans=(3, 7, 14, 21),
    )
    deep_learning_result = select_mlp_window_size(
        train_series,
        test_data["cleaned_value"],
        window_sizes=(3, 7, 14, 21),
    )

    forecasts = {
        (
            f"Local polynomial N={polynomial_result.best_window}, "
            f"d={polynomial_result.best_degree}"
        ): polynomial_result.predictions,
        f"MA w={moving_average_result.best_parameter}": moving_average_result.predictions,
        f"EMA s={ema_result.best_parameter}": ema_result.predictions,
        f"MLP k={deep_learning_result.best_window_size}": deep_learning_result.predictions,
    }
    save_synthetic_verification_plot(
        data=verification_data,
        date_column="date",
        true_column="true_trend",
        noisy_column="noisy_value",
        observed_column="value",
        cleaned_column="cleaned_value",
        forecast_dates=test_data["date"],
        forecasts=forecasts,
        output_path=figure_path,
        title=f"Synthetic {trend_type.title()} Verification",
    )

    injected_mask = verification_data["is_injected_anomaly"].astype(bool)
    detected_mask = anomaly_results["rolling_median"].anomaly_mask.astype(bool)
    true_positives = int((injected_mask & detected_mask).sum())
    false_positives = int((~injected_mask & detected_mask).sum())
    false_negatives = int((injected_mask & ~detected_mask).sum())

    return {
        "config": {
            "trend_type": config.trend_type,
            "n_points": config.n_points,
            "noise_std": config.noise_std,
            "anomaly_fraction": config.anomaly_fraction,
            "anomaly_scale": config.anomaly_scale,
            "random_seed": config.random_seed,
        },
        "anomaly_detection": {
            "injected_anomaly_count": int(injected_mask.sum()),
            "rolling_median_detected_count": anomaly_results["rolling_median"].anomaly_count,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "all_methods": {
                method_name: result.anomaly_count
                for method_name, result in anomaly_results.items()
            },
        },
        "forecasting": {
            "target": "true_trend",
            "train_size": len(train_data),
            "test_size": len(test_data),
            "models": {
                "local_polynomial_one_step": {
                    "parameters": {
                        "best_window": polynomial_result.best_window,
                        "best_degree": polynomial_result.best_degree,
                    },
                    "metrics": evaluate_forecast(test_series, polynomial_result.predictions),
                },
                "moving_average_one_step": {
                    "parameters": {"best_window": moving_average_result.best_parameter},
                    "metrics": evaluate_forecast(test_series, moving_average_result.predictions),
                },
                "exponential_moving_average_one_step": {
                    "parameters": {"best_span": ema_result.best_parameter},
                    "metrics": evaluate_forecast(test_series, ema_result.predictions),
                },
                "deep_learning_mlp_one_step": {
                    "parameters": {
                        "backend": deep_learning_result.backend,
                        "best_window_size": deep_learning_result.best_window_size,
                    },
                    "metrics": evaluate_forecast(test_series, deep_learning_result.predictions),
                },
            },
        },
        "figure": str(figure_path),
    }


def run_synthetic_experiments() -> dict[str, Any]:
    """Run synthetic verification for all supported trend types."""
    results = {
        trend_type: run_single_synthetic_experiment(trend_type)
        for trend_type in ("linear", "quadratic", "exponential")
    }
    metrics_path = PROJECT_ROOT / "reports" / "metrics" / "synthetic_verification.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )
    return {"metrics": str(metrics_path), "results": results}


def main() -> None:
    """Execute all synthetic verification experiments."""
    summary = run_synthetic_experiments()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_ready))


if __name__ == "__main__":
    main()

