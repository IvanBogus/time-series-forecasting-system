"""Visualization utilities for time series analysis."""

from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PathLike = Union[str, Path]


def save_time_series_plot(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    output_path: PathLike,
    title: str = "Oschadbank USD Time Series",
) -> Path:
    """Save a line plot of a time series to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(data[date_column], data[value_column], linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_anomalies_plot(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    anomaly_mask: pd.Series,
    output_path: PathLike,
    title: str = "Original Series with Anomalies Highlighted",
) -> Path:
    """Save the original series with detected anomalies highlighted."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        data[date_column],
        data[value_column],
        linewidth=1.6,
        label="Original",
    )
    anomalies = data.loc[anomaly_mask]
    axis.scatter(
        anomalies[date_column],
        anomalies[value_column],
        color="crimson",
        s=32,
        label="Anomalies",
        zorder=3,
    )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_cleaned_comparison_plot(
    original_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    date_column: str,
    value_column: str,
    output_path: PathLike,
    title: str = "Cleaned vs Original Series",
) -> Path:
    """Save a comparison plot for original and cleaned time series."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        original_data[date_column],
        original_data[value_column],
        linewidth=1.3,
        alpha=0.65,
        label="Original",
    )
    axis.plot(
        cleaned_data[date_column],
        cleaned_data[value_column],
        linewidth=1.8,
        label="Cleaned",
    )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_forecast_comparison_plot(
    dates: pd.Series,
    y_true: pd.Series,
    forecasts: dict[str, pd.Series],
    output_path: PathLike,
    title: str = "Forecast Comparison",
) -> Path:
    """Save a comparison plot for actual test values and model forecasts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(dates, y_true, linewidth=2.0, label="Actual", color="black")
    for model_name, forecast in forecasts.items():
        axis.plot(dates, forecast, linewidth=1.5, label=model_name)

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_metric_selection_plot(
    metrics_by_parameter: dict[str, dict[str, float]],
    output_path: PathLike,
    parameter_label: str,
    title: str,
) -> Path:
    """Save an RMSE/MAE selection plot for model parameters."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parameters = list(metrics_by_parameter.keys())
    rmse_values = [metrics_by_parameter[parameter]["RMSE"] for parameter in parameters]
    mae_values = [metrics_by_parameter[parameter]["MAE"] for parameter in parameters]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(parameters, rmse_values, marker="o", label="RMSE")
    axis.plot(parameters, mae_values, marker="o", label="MAE")
    axis.set_title(title)
    axis.set_xlabel(parameter_label)
    axis.set_ylabel("Error")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_local_polynomial_top_configs_plot(
    metrics_by_configuration: dict[str, dict[str, float]],
    output_path: PathLike,
    top_n: int = 10,
    title: str = "Top Local Polynomial Configurations",
) -> Path:
    """Save RMSE/MAE for the top local polynomial configurations by RMSE."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_items = sorted(
        metrics_by_configuration.items(),
        key=lambda item: item[1]["RMSE"],
    )[:top_n]
    labels = []
    rmse_values = []
    mae_values = []
    for configuration, metrics in top_items:
        parts = dict(part.split("=", 1) for part in configuration.split("|"))
        labels.append(f"N={parts['window']},d={parts['degree']}")
        rmse_values.append(metrics["RMSE"])
        mae_values.append(metrics["MAE"])

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(labels, rmse_values, marker="o", label="RMSE")
    axis.plot(labels, mae_values, marker="o", label="MAE")
    axis.set_title(title)
    axis.set_xlabel("Configuration")
    axis.set_ylabel("Error")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_approximation_selection_plot(
    moving_average_metrics: dict[str, dict[str, float]],
    ema_metrics: dict[str, dict[str, float]],
    output_path: PathLike,
    moving_average_one_step_metrics: dict[str, dict[str, float]] | None = None,
    ema_one_step_metrics: dict[str, dict[str, float]] | None = None,
    title: str = "Approximation Parameter Selection",
) -> Path:
    """Save RMSE curves for moving average and EMA parameter selection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    ma_parameters = list(moving_average_metrics.keys())
    ema_parameters = list(ema_metrics.keys())
    axis.plot(
        ma_parameters,
        [moving_average_metrics[parameter]["RMSE"] for parameter in ma_parameters],
        marker="o",
        label="MA recursive RMSE",
    )
    axis.plot(
        ema_parameters,
        [ema_metrics[parameter]["RMSE"] for parameter in ema_parameters],
        marker="o",
        label="EMA recursive RMSE",
    )
    if moving_average_one_step_metrics is not None:
        ma_one_step_parameters = list(moving_average_one_step_metrics.keys())
        axis.plot(
            ma_one_step_parameters,
            [
                moving_average_one_step_metrics[parameter]["RMSE"]
                for parameter in ma_one_step_parameters
            ],
            marker="o",
            label="MA one-step RMSE",
        )
    if ema_one_step_metrics is not None:
        ema_one_step_parameters = list(ema_one_step_metrics.keys())
        axis.plot(
            ema_one_step_parameters,
            [
                ema_one_step_metrics[parameter]["RMSE"]
                for parameter in ema_one_step_parameters
            ],
            marker="o",
            label="EMA one-step RMSE",
        )
    axis.set_title(title)
    axis.set_xlabel("Window / Span")
    axis.set_ylabel("RMSE")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_deep_learning_forecast_plot(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: PathLike,
    title: str = "Deep Learning One-Step Forecast",
) -> Path:
    """Save actual vs MLP forecast values for the test split."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(dates, y_true, linewidth=2.0, label="Actual", color="black")
    axis.plot(dates, y_pred, linewidth=1.8, label="MLP forecast")
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_synthetic_verification_plot(
    data: pd.DataFrame,
    date_column: str,
    true_column: str,
    noisy_column: str,
    observed_column: str,
    cleaned_column: str,
    forecast_dates: pd.Series,
    forecasts: dict[str, pd.Series],
    output_path: PathLike,
    title: str,
) -> Path:
    """Save a synthetic verification plot with trend, data, cleaning, and forecasts."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(13, 7))
    axis.plot(
        data[date_column],
        data[true_column],
        linewidth=2.0,
        color="black",
        label="True trend",
    )
    axis.plot(
        data[date_column],
        data[noisy_column],
        linewidth=1.0,
        alpha=0.55,
        label="Noisy data",
    )
    axis.scatter(
        data.loc[data["is_injected_anomaly"], date_column],
        data.loc[data["is_injected_anomaly"], observed_column],
        color="crimson",
        s=28,
        label="Injected anomalies",
        zorder=3,
    )
    axis.plot(
        data[date_column],
        data[cleaned_column],
        linewidth=1.8,
        label="Cleaned data",
    )
    for model_name, forecast in forecasts.items():
        axis.plot(forecast_dates, forecast, linewidth=1.4, label=model_name)

    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path


def save_anomaly_methods_comparison_plot(
    data: pd.DataFrame,
    date_column: str,
    value_column: str,
    anomaly_masks: dict[str, pd.Series],
    output_path: PathLike,
    title: str = "Anomaly Methods Comparison",
) -> Path:
    """Save a compact comparison of anomaly masks from multiple detectors."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    method_names = list(anomaly_masks.keys())
    figure, axes = plt.subplots(
        nrows=len(method_names),
        ncols=1,
        figsize=(13, max(3, 2.4 * len(method_names))),
        sharex=True,
    )
    if len(method_names) == 1:
        axes = [axes]

    for axis, method_name in zip(axes, method_names):
        mask = anomaly_masks[method_name].astype(bool)
        axis.plot(data[date_column], data[value_column], linewidth=1.1, color="steelblue")
        anomalies = data.loc[mask]
        axis.scatter(
            anomalies[date_column],
            anomalies[value_column],
            color="crimson",
            s=22,
            zorder=3,
        )
        axis.set_title(f"{method_name}: {int(mask.sum())} anomalies")
        axis.set_ylabel(value_column)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    figure.suptitle(title)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)

    return path
