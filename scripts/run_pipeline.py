"""Run the first-stage Oschadbank USD time series pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import run_anomaly_detection
from src.data_loader import load_excel
from src.evaluation import evaluate_forecast
from src.models.approximation import (
    exponential_moving_average_forecast,
    exponential_moving_average_one_step_forecast,
    moving_average_forecast,
    moving_average_one_step_forecast,
    select_exponential_moving_average_one_step_span,
    select_exponential_moving_average_span,
    select_moving_average_one_step_window,
    select_moving_average_window,
)
from src.models.alpha_beta_filter import (
    forecast_alpha_beta,
    forecast_alpha_beta_one_step,
    optimize_alpha_beta,
)
from src.models.baseline import naive_forecast, naive_recursive_forecast
from src.models.deep_learning import mlp_one_step_forecast, select_mlp_window_size
from src.models.polynomial import (
    local_polynomial_forecast,
    local_polynomial_one_step_forecast,
    local_polynomial_recursive_forecast,
    polynomial_forecast,
    select_local_polynomial_configuration,
    select_local_polynomial_one_step_configuration,
    select_local_polynomial_recursive_configuration,
    select_polynomial_degree,
)
from src.preprocessing import preprocess_time_series
from src.statistics import calculate_basic_statistics
from src.visualization import (
    save_approximation_selection_plot,
    save_anomalies_plot,
    save_anomaly_methods_comparison_plot,
    save_cleaned_comparison_plot,
    save_deep_learning_forecast_plot,
    save_forecast_comparison_plot,
    save_local_polynomial_top_configs_plot,
    save_metric_selection_plot,
    save_time_series_plot,
)


def choose_value_column(data_columns: list[str], date_column: str) -> str:
    """Choose the default numerical value column for the first pipeline run."""
    preferred_columns = ("продаж", "sell", "курснбу", "nbu", "купівля", "buy")
    searchable_columns = [column for column in data_columns if column != date_column]

    for preferred in preferred_columns:
        for column in searchable_columns:
            if preferred in column.lower():
                return column

    if searchable_columns:
        return searchable_columns[0]

    raise ValueError("No value column found after preprocessing.")


def train_test_split_time_series(data_length: int, train_size: float = 0.8) -> tuple[slice, slice]:
    """Create chronological train/test slices for a time series."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    split_index = int(data_length * train_size)
    if split_index == 0 or split_index >= data_length:
        raise ValueError("Both train and test splits must contain at least one row.")
    return slice(0, split_index), slice(split_index, data_length)


def train_validation_test_split_time_series(
    data_length: int,
    train_size: float = 0.7,
    validation_size: float = 0.1,
) -> tuple[slice, slice, slice]:
    """Create chronological train/validation/test slices for a time series."""
    if data_length < 3:
        raise ValueError("Time series must contain at least three rows.")
    if train_size <= 0 or validation_size <= 0:
        raise ValueError("train_size and validation_size must be positive.")
    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must be less than 1.")

    train_end = int(data_length * train_size)
    validation_end = int(data_length * (train_size + validation_size))
    if train_end == 0 or validation_end <= train_end or validation_end >= data_length:
        raise ValueError("Train, validation, and test splits must be non-empty.")

    return (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, data_length),
    )


def best_polynomial_metrics_by_window(
    metrics_by_configuration: dict[str, dict[str, float]],
    prefix: str,
) -> dict[str, dict[str, float]]:
    """Keep the best polynomial configuration for each local window."""
    best_by_window: dict[str, tuple[str, dict[str, float]]] = {}
    for configuration, metrics in metrics_by_configuration.items():
        parts = dict(part.split("=", 1) for part in configuration.split("|"))
        window = parts["window"]
        degree = parts["degree"]
        current = best_by_window.get(window)
        if current is None or metrics["RMSE"] < current[1]["RMSE"]:
            best_by_window[window] = (degree, metrics)

    return {
        f"{prefix} N={window}, d={degree}": metrics
        for window, (degree, metrics) in best_by_window.items()
    }


def run_pipeline() -> dict[str, Any]:
    """Load, preprocess, summarize, detect anomalies, and plot the dataset."""
    dataset_path = PROJECT_ROOT / "data" / "raw" / "Oschadbank_USD.xls"
    processed_path = PROJECT_ROOT / "data" / "processed" / "oschadbank_usd_clean.csv"
    anomaly_processed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "oschadbank_usd_cleaned_anomalies.csv"
    )
    metrics_path = PROJECT_ROOT / "reports" / "metrics" / "basic_statistics.json"
    anomaly_report_path = PROJECT_ROOT / "reports" / "metrics" / "anomaly_report.json"
    adaptive_anomaly_report_path = (
        PROJECT_ROOT / "reports" / "metrics" / "adaptive_anomaly_report.json"
    )
    figure_path = PROJECT_ROOT / "reports" / "figures" / "oschadbank_usd_series.png"
    anomalies_figure_path = (
        PROJECT_ROOT / "reports" / "figures" / "anomalies_detected.png"
    )
    adaptive_anomalies_figure_path = (
        PROJECT_ROOT / "reports" / "figures" / "adaptive_anomalies_detected.png"
    )
    anomaly_methods_comparison_path = (
        PROJECT_ROOT / "reports" / "figures" / "anomaly_methods_comparison.png"
    )
    comparison_figure_path = (
        PROJECT_ROOT / "reports" / "figures" / "cleaned_vs_original.png"
    )
    forecast_metrics_path = PROJECT_ROOT / "reports" / "metrics" / "forecast_metrics.json"
    model_recommendations_path = (
        PROJECT_ROOT / "reports" / "metrics" / "model_recommendations.json"
    )
    forecast_comparison_path = (
        PROJECT_ROOT / "reports" / "figures" / "forecast_comparison.png"
    )
    forecast_comparison_best_path = (
        PROJECT_ROOT / "reports" / "figures" / "forecast_comparison_best.png"
    )
    global_polynomial_selection_path = (
        PROJECT_ROOT / "reports" / "figures" / "global_polynomial_degree_selection.png"
    )
    local_polynomial_selection_path = (
        PROJECT_ROOT / "reports" / "figures" / "local_polynomial_selection.png"
    )
    approximation_selection_path = (
        PROJECT_ROOT / "reports" / "figures" / "approximation_selection.png"
    )
    deep_learning_forecast_path = (
        PROJECT_ROOT / "reports" / "figures" / "deep_learning_forecast.png"
    )
    alpha_beta_forecast_path = (
        PROJECT_ROOT / "reports" / "figures" / "alpha_beta_forecast.png"
    )
    alpha_beta_metrics_path = (
        PROJECT_ROOT / "reports" / "metrics" / "alpha_beta_metrics.json"
    )

    raw_data = load_excel(dataset_path)
    cleaned_data, date_column = preprocess_time_series(raw_data)
    value_column = choose_value_column(list(cleaned_data.columns), date_column)
    stats = calculate_basic_statistics(cleaned_data[value_column])
    anomaly_results = run_anomaly_detection(cleaned_data, value_column)
    primary_anomaly_result = anomaly_results["rolling_median"]
    adaptive_anomaly_result = anomaly_results["adaptive_local_mad"]
    combined_anomaly_mask = None
    for result in anomaly_results.values():
        if combined_anomaly_mask is None:
            combined_anomaly_mask = result.anomaly_mask.copy()
        else:
            combined_anomaly_mask = combined_anomaly_mask | result.anomaly_mask
    if combined_anomaly_mask is None:
        raise ValueError("No anomaly detection results were produced.")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(processed_path, index=False)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    anomaly_output = cleaned_data[[date_column, value_column]].copy()
    anomaly_output = anomaly_output.rename(columns={value_column: "original_value"})
    for method_name, result in anomaly_results.items():
        anomaly_output[f"{method_name}_is_anomaly"] = result.anomaly_mask.to_numpy()
        anomaly_output[f"{method_name}_cleaned_value"] = result.cleaned_data[
            value_column
        ].to_numpy()
    anomaly_output["combined_is_anomaly"] = combined_anomaly_mask.to_numpy()
    anomaly_output["final_cleaned_value"] = primary_anomaly_result.cleaned_data[
        value_column
    ].to_numpy()
    anomaly_processed_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_output.to_csv(anomaly_processed_path, index=False)

    anomaly_report = {
        method_name: {
            "anomaly_count": result.anomaly_count,
            "anomaly_rate": result.anomaly_count / len(cleaned_data),
            "replacement": (
                "rolling_median" if method_name == "rolling_median" else "interpolation"
            ),
        }
        for method_name, result in anomaly_results.items()
    }
    anomaly_report["primary_cleaning_method"] = primary_anomaly_result.method
    anomaly_report["combined_anomaly_count"] = int(combined_anomaly_mask.sum())
    anomaly_report["value_column"] = value_column
    anomaly_report["rows"] = len(cleaned_data)
    anomaly_report_path.parent.mkdir(parents=True, exist_ok=True)
    anomaly_report_path.write_text(
        json.dumps(anomaly_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    adaptive_diagnostics = adaptive_anomaly_result.diagnostics
    adaptive_zero_anomalies = int(
        (
            adaptive_anomaly_result.anomaly_mask
            & cleaned_data[value_column].eq(0)
        ).sum()
    )
    adaptive_total_anomalies = adaptive_anomaly_result.anomaly_count
    adaptive_non_zero_anomalies = adaptive_total_anomalies - adaptive_zero_anomalies

    def summarize_diagnostic(column_name: str) -> dict[str, float] | None:
        """Return compact descriptive statistics for an adaptive diagnostic column."""
        if adaptive_diagnostics is None:
            return None
        series = adaptive_diagnostics[column_name]
        return {
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
            "q95": float(series.quantile(0.95)),
            "q99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }

    adaptive_anomaly_report = {
        "method": "adaptive_local_mad",
        "parameters": {
            "window": 21,
            "mad_threshold": 3.5,
            "derivative_threshold": 6.0,
            "min_mad": 0.03,
            "mad_floor_ratio": 0.2,
            "min_step_mad": 0.05,
            "step_floor_ratio": 0.2,
            "min_abs_step": 0.25,
            "replacement": "interpolation",
        },
        "total_anomalies": adaptive_total_anomalies,
        "zero_anomalies": adaptive_zero_anomalies,
        "non_zero_anomalies": adaptive_non_zero_anomalies,
        "anomaly_count": adaptive_total_anomalies,
        "anomaly_rate": adaptive_total_anomalies / len(cleaned_data),
        "compared_methods": anomaly_report,
        "diagnostics_summary": {
            "local_mad": summarize_diagnostic("local_mad"),
            "step_mad": summarize_diagnostic("step_mad"),
            "amplitude_score": summarize_diagnostic("amplitude_score"),
            "derivative_score": summarize_diagnostic("derivative_score"),
            "final_score": summarize_diagnostic("final_score"),
        },
        "r_and_d_conclusion_uk": (
            "Запропонований Adaptive Local MAD Anomaly Detector є власним "
            "алгоритмом виявлення аномалій для часових рядів. На відміну "
            "від z-score та IQR, він не використовує глобальні статистики "
            "всієї вибірки, а оцінює локальний тренд і локальний рівень "
            "шуму у ковзному вікні. Це дозволяє враховувати "
            "нестаціонарність, локальну волатильність та різкі зміни ряду."
        ),
    }
    adaptive_anomaly_report_path.write_text(
        json.dumps(adaptive_anomaly_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saved_figure = save_time_series_plot(
        cleaned_data,
        date_column=date_column,
        value_column=value_column,
        output_path=figure_path,
    )
    saved_anomalies_figure = save_anomalies_plot(
        cleaned_data,
        date_column=date_column,
        value_column=value_column,
        anomaly_mask=combined_anomaly_mask,
        output_path=anomalies_figure_path,
        title="Original Series with Combined Anomalies Highlighted",
    )
    saved_adaptive_anomalies_figure = save_anomalies_plot(
        cleaned_data,
        date_column=date_column,
        value_column=value_column,
        anomaly_mask=adaptive_anomaly_result.anomaly_mask,
        output_path=adaptive_anomalies_figure_path,
        title="Adaptive Local MAD Anomalies",
    )
    saved_anomaly_methods_comparison_figure = save_anomaly_methods_comparison_plot(
        cleaned_data,
        date_column=date_column,
        value_column=value_column,
        anomaly_masks={
            method_name: result.anomaly_mask
            for method_name, result in anomaly_results.items()
        },
        output_path=anomaly_methods_comparison_path,
    )
    saved_comparison_figure = save_cleaned_comparison_plot(
        original_data=cleaned_data,
        cleaned_data=primary_anomaly_result.cleaned_data,
        date_column=date_column,
        value_column=value_column,
        output_path=comparison_figure_path,
    )

    forecast_data = primary_anomaly_result.cleaned_data[[date_column, value_column]].copy()
    train_slice, validation_slice, test_slice = train_validation_test_split_time_series(
        len(forecast_data),
        train_size=0.7,
        validation_size=0.1,
    )
    train_data = forecast_data.iloc[train_slice]
    validation_data = forecast_data.iloc[validation_slice]
    test_data = forecast_data.iloc[test_slice]
    train_series = train_data[value_column]
    validation_series = validation_data[value_column]
    test_series = test_data[value_column]
    train_validation_data = forecast_data.iloc[: test_slice.start]
    train_validation_series = train_validation_data[value_column]

    baseline_predictions = naive_forecast(train_validation_series, test_series)
    baseline_recursive_predictions = naive_recursive_forecast(
        train_validation_series,
        len(test_series),
    )
    polynomial_result = select_polynomial_degree(
        train_series,
        validation_series,
        degrees=range(1, 9),
    )
    polynomial_predictions = polynomial_forecast(
        train_validation_series,
        len(test_series),
        polynomial_result.best_degree,
    )
    local_polynomial_result = select_local_polynomial_configuration(
        train_series,
        validation_series,
        degrees=range(1, 9),
        windows=(30, 60, 90, 120, 180, "all"),
    )
    local_polynomial_predictions = local_polynomial_forecast(
        train_validation_series,
        len(test_series),
        local_polynomial_result.best_degree,
        local_polynomial_result.best_window,
    )
    local_polynomial_recursive_result = select_local_polynomial_recursive_configuration(
        train_series,
        validation_series,
        degrees=range(1, 9),
        windows=(30, 60, 90, 120, 180, "all"),
    )
    local_polynomial_recursive_predictions = local_polynomial_recursive_forecast(
        train_validation_series,
        len(test_series),
        local_polynomial_recursive_result.best_degree,
        local_polynomial_recursive_result.best_window,
    )
    local_polynomial_one_step_result = select_local_polynomial_one_step_configuration(
        train_series,
        validation_series,
        degrees=range(1, 9),
        windows=(30, 60, 90, 120, 180, "all"),
    )
    # Online one-step forecasts update with the actual test value only after
    # producing the current step forecast.
    local_polynomial_one_step_predictions = local_polynomial_one_step_forecast(
        train_validation_series,
        test_series,
        local_polynomial_one_step_result.best_degree,
        local_polynomial_one_step_result.best_window,
    )
    moving_average_result = select_moving_average_window(
        train_series,
        validation_series,
        windows=(3, 5, 7, 14, 21, 30),
    )
    moving_average_predictions = moving_average_forecast(
        train_validation_series,
        len(test_series),
        moving_average_result.best_parameter,
    )
    moving_average_one_step_result = select_moving_average_one_step_window(
        train_series,
        validation_series,
        windows=(3, 5, 7, 14, 21, 30),
    )
    moving_average_one_step_predictions = moving_average_one_step_forecast(
        train_validation_series,
        test_series,
        moving_average_one_step_result.best_parameter,
    )
    ema_result = select_exponential_moving_average_span(
        train_series,
        validation_series,
        spans=(3, 5, 7, 14, 21, 30),
    )
    ema_predictions = exponential_moving_average_forecast(
        train_validation_series,
        len(test_series),
        ema_result.best_parameter,
    )
    ema_one_step_result = select_exponential_moving_average_one_step_span(
        train_series,
        validation_series,
        spans=(3, 5, 7, 14, 21, 30),
    )
    ema_one_step_predictions = exponential_moving_average_one_step_forecast(
        train_validation_series,
        test_series,
        ema_one_step_result.best_parameter,
    )
    deep_learning_result = select_mlp_window_size(
        train_series,
        validation_series,
        window_sizes=(3, 7, 14, 21),
    )
    deep_learning_predictions, deep_learning_backend = mlp_one_step_forecast(
        train_validation_series,
        test_series,
        deep_learning_result.best_window_size,
    )
    alpha_beta_result = optimize_alpha_beta(
        train_series,
        validation_series,
    )
    alpha_beta_recursive_predictions = forecast_alpha_beta(
        train_validation_series,
        steps=len(test_series),
        alpha=float(alpha_beta_result["best_alpha"]),
        beta=float(alpha_beta_result["best_beta"]),
    )
    alpha_beta_one_step_predictions = forecast_alpha_beta_one_step(
        train_validation_series,
        test_series,
        alpha=float(alpha_beta_result["best_alpha"]),
        beta=float(alpha_beta_result["best_beta"]),
    )
    alpha_beta_recursive_metrics = evaluate_forecast(
        test_series,
        alpha_beta_recursive_predictions,
    )
    alpha_beta_one_step_metrics = evaluate_forecast(
        test_series,
        alpha_beta_one_step_predictions,
    )
    alpha_beta_metrics = {
        "best_alpha": alpha_beta_result["best_alpha"],
        "best_beta": alpha_beta_result["best_beta"],
        "forecast_horizon": len(test_series),
        "split": {
            "train_size": len(train_data),
            "validation_size": len(validation_data),
            "test_size": len(test_data),
            "selection_data": "train -> validation",
            "final_fit_data": "train + validation",
            "final_evaluation_data": "test only",
        },
        "alpha_beta_recursive": {
            "MAE": alpha_beta_recursive_metrics["MAE"],
            "RMSE": alpha_beta_recursive_metrics["RMSE"],
            "R2": alpha_beta_recursive_metrics["R2"],
        },
        "alpha_beta_one_step": {
            "MAE": alpha_beta_one_step_metrics["MAE"],
            "RMSE": alpha_beta_one_step_metrics["RMSE"],
            "R2": alpha_beta_one_step_metrics["R2"],
        },
        "conclusion": (
            "Alpha-beta filter С” РїСЂРѕСЃС‚РёРј СЂРµРєСѓСЂРµРЅС‚РЅРёРј РјРµС‚РѕРґРѕРј Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ. "
            "Р’С–РЅ РґРѕР±СЂРµ РїС–РґС…РѕРґРёС‚СЊ РґР»СЏ РєРѕСЂРѕС‚РєРѕСЃС‚СЂРѕРєРѕРІРѕРіРѕ РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ С‚Р° "
            "Р·РјРµРЅС€РµРЅРЅСЏ С€СѓРјСѓ, Р°Р»Рµ РјРµРЅС€ РµС„РµРєС‚РёРІРЅРёР№ РґР»СЏ РґРѕРІРіРѕСЃС‚СЂРѕРєРѕРІРѕРіРѕ РїСЂРѕРіРЅРѕР·Сѓ "
            "РЅРµР»С–РЅС–Р№РЅРёС… С‚СЂРµРЅРґС–РІ. Recursive alpha-beta forecast РЅР°РєРѕРїРёС‡СѓС” "
            "РїРѕС…РёР±РєСѓ С‡РµСЂРµР· velocity drift, С‚РѕРґС– СЏРє one-step СЂРµР¶РёРј РєСЂР°С‰Рµ "
            "РїС–РґС…РѕРґРёС‚СЊ РґР»СЏ online-Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ РєРѕСЂРѕС‚РєРѕСЃС‚СЂРѕРєРѕРІРёС… Р·РјС–РЅ."
        ),
    }
    alpha_beta_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_beta_metrics_path.write_text(
        json.dumps(alpha_beta_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    forecast_metrics = {
        "split": {
            "train_ratio": 0.7,
            "validation_ratio": 0.1,
            "test_ratio": 0.2,
            "train_size": len(train_data),
            "validation_size": len(validation_data),
            "test_size": len(test_data),
            "train_index_range": [train_slice.start, train_slice.stop - 1],
            "validation_index_range": [
                validation_slice.start,
                validation_slice.stop - 1,
            ],
            "test_index_range": [test_slice.start, test_slice.stop - 1],
            "model_selection_data": "train + validation only",
            "final_evaluation_data": "test only",
            "final_fit_data": "train + validation",
        },
        "models": {
            "naive_one_step": {
                "parameters": {
                    "strategy": "y[t+1] = y[t]",
                    "fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, baseline_predictions),
            },
            "naive_recursive": {
                "parameters": {
                    "strategy": "last train+validation value carried forward",
                    "fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, baseline_recursive_predictions),
            },
            "global_polynomial": {
                "parameters": {
                    "best_degree": polynomial_result.best_degree,
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, polynomial_predictions),
                "validation_metrics_by_degree": polynomial_result.metrics_by_degree,
            },
            "local_polynomial": {
                "parameters": {
                    "best_window": local_polynomial_result.best_window,
                    "best_degree": local_polynomial_result.best_degree,
                    "tested_windows": [30, 60, 90, 120, 180, "all"],
                    "x_normalization": "standardized local index before polyfit",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, local_polynomial_predictions),
                "validation_metrics_by_configuration": (
                    local_polynomial_result.metrics_by_configuration
                ),
            },
            "local_polynomial_recursive": {
                "parameters": {
                    "best_window": local_polynomial_recursive_result.best_window,
                    "best_degree": local_polynomial_recursive_result.best_degree,
                    "history_update": "prediction",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(
                    test_series,
                    local_polynomial_recursive_predictions,
                ),
                "validation_metrics_by_configuration": (
                    local_polynomial_recursive_result.metrics_by_configuration
                ),
            },
            "local_polynomial_one_step": {
                "parameters": {
                    "best_window": local_polynomial_one_step_result.best_window,
                    "best_degree": local_polynomial_one_step_result.best_degree,
                    "history_update": "actual after each one-step prediction",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(
                    test_series,
                    local_polynomial_one_step_predictions,
                ),
                "validation_metrics_by_configuration": (
                    local_polynomial_one_step_result.metrics_by_configuration
                ),
            },
            "moving_average_recursive": {
                "parameters": {
                    "best_window": moving_average_result.best_parameter,
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, moving_average_predictions),
                "validation_metrics_by_window": moving_average_result.metrics_by_parameter,
            },
            "moving_average_one_step": {
                "parameters": {
                    "best_window": moving_average_one_step_result.best_parameter,
                    "history_update": "actual after each one-step prediction",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(
                    test_series,
                    moving_average_one_step_predictions,
                ),
                "validation_metrics_by_window": (
                    moving_average_one_step_result.metrics_by_parameter
                ),
            },
            "exponential_moving_average_recursive": {
                "parameters": {
                    "best_span": ema_result.best_parameter,
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, ema_predictions),
                "validation_metrics_by_span": ema_result.metrics_by_parameter,
            },
            "exponential_moving_average_one_step": {
                "parameters": {
                    "best_span": ema_one_step_result.best_parameter,
                    "history_update": "actual after each one-step prediction",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, ema_one_step_predictions),
                "validation_metrics_by_span": ema_one_step_result.metrics_by_parameter,
            },
            "deep_learning_mlp_one_step": {
                "parameters": {
                    "backend": deep_learning_backend,
                    "best_window_size": deep_learning_result.best_window_size,
                    "tested_window_sizes": [3, 7, 14, 21],
                    "scaler": "MinMaxScaler",
                    "history_update": "actual after each one-step prediction",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                },
                "metrics": evaluate_forecast(test_series, deep_learning_predictions),
                "validation_metrics_by_window_size": deep_learning_result.metrics_by_window,
            },
            "alpha_beta_recursive": {
                "parameters": {
                    "best_alpha": alpha_beta_result["best_alpha"],
                    "best_beta": alpha_beta_result["best_beta"],
                    "dt": 1.0,
                    "selection_metric": "validation MAE",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                    "history_update": "prediction",
                },
                "metrics": alpha_beta_recursive_metrics,
                "validation_metrics_by_configuration": alpha_beta_result[
                    "metrics_by_configuration"
                ],
            },
            "alpha_beta_one_step": {
                "parameters": {
                    "best_alpha": alpha_beta_result["best_alpha"],
                    "best_beta": alpha_beta_result["best_beta"],
                    "dt": 1.0,
                    "selection_metric": "validation MAE",
                    "selection_data": "train -> validation",
                    "final_fit_data": "train + validation",
                    "history_update": "actual after each one-step prediction",
                },
                "metrics": alpha_beta_one_step_metrics,
            },
        },
    }
    forecast_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_metrics_path.write_text(
        json.dumps(forecast_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    online_candidates = {
        "local_polynomial_one_step": forecast_metrics["models"][
            "local_polynomial_one_step"
        ]["metrics"],
        "moving_average_one_step": forecast_metrics["models"][
            "moving_average_one_step"
        ]["metrics"],
        "exponential_moving_average_one_step": forecast_metrics["models"][
            "exponential_moving_average_one_step"
        ]["metrics"],
        "deep_learning_mlp_one_step": forecast_metrics["models"][
            "deep_learning_mlp_one_step"
        ]["metrics"],
        "alpha_beta_one_step": forecast_metrics["models"]["alpha_beta_one_step"][
            "metrics"
        ],
    }
    best_online_model = min(
        online_candidates,
        key=lambda model_name: online_candidates[model_name]["RMSE"],
    )
    model_recommendations = {
        "selection_basis": "Final test RMSE after chronological 70/10/20 split.",
        "best_short_term_online_forecasting": {
            "model": best_online_model,
            "metrics": online_candidates[best_online_model],
            "recommendation": (
                "Р”Р»СЏ РєРѕСЂРѕС‚РєРѕСЃС‚СЂРѕРєРѕРІРѕРіРѕ online-РїСЂРѕРіРЅРѕР·СѓРІР°РЅРЅСЏ РґРѕС†С–Р»СЊРЅРѕ РѕР±СЂР°С‚Рё "
                f"{best_online_model}, РѕСЃРєС–Р»СЊРєРё С†СЏ РјРѕРґРµР»СЊ РјР°С” РЅР°Р№РјРµРЅС€РёР№ RMSE "
                "РЅР° С„С–РЅР°Р»СЊРЅРѕРјСѓ test-РЅР°Р±РѕСЂС– СЃРµСЂРµРґ one-step РјРµС‚РѕРґС–РІ."
            ),
        },
        "global_polynomial": (
            "Global polynomial РіС–СЂС€Рµ РїС–РґС…РѕРґРёС‚СЊ РґР»СЏ С†СЊРѕРіРѕ СЂСЏРґСѓ, Р±Рѕ РѕРґРЅР° РіР»РѕР±Р°Р»СЊРЅР° "
            "РєСЂРёРІР° РїРѕРіР°РЅРѕ Р°РґР°РїС‚СѓС”С‚СЊСЃСЏ РґРѕ Р»РѕРєР°Р»СЊРЅРёС… Р·РјС–РЅ РєСѓСЂСЃСѓ С‚Р° РґР°С” РЅРµСЃС‚Р°Р±С–Р»СЊРЅРёР№ "
            "РґРѕРІРіРѕСЃС‚СЂРѕРєРѕРІРёР№ РїСЂРѕРіРЅРѕР·."
        ),
        "local_polynomial": (
            "Local polynomial РґРѕС†С–Р»СЊРЅРёР№, РєРѕР»Рё РІ СЂСЏРґС– С” Р»РѕРєР°Р»СЊРЅС– С‚СЂРµРЅРґРё: РјРѕРґРµР»СЊ "
            "РІСЂР°С…РѕРІСѓС” РѕСЃС‚Р°РЅРЅС” РІС–РєРЅРѕ СЃРїРѕСЃС‚РµСЂРµР¶РµРЅСЊ С– РєСЂР°С‰Рµ СЂРµР°РіСѓС” РЅР° РєРѕСЂРѕС‚РєС– Р·РјС–РЅРё, "
            "Р°Р»Рµ РїРѕС‚СЂРµР±СѓС” РїС–РґР±РѕСЂСѓ window С‚Р° degree."
        ),
        "mlp": (
            "MLP РїРѕРєР°Р·Р°РІ РїСЂР°С†РµР·РґР°С‚РЅРёР№ one-step РїСЂРѕРіРЅРѕР· РЅР° РѕС‡РёС‰РµРЅРёС… РґР°РЅРёС…, Р°Р»Рµ "
            "РЅРµ СЃС‚Р°РІ Р±РµР·СѓРјРѕРІРЅРѕ РЅР°Р№РєСЂР°С‰РёРј; Р№РѕРіРѕ РІР°СЂС‚Рѕ СЂРѕР·РіР»СЏРґР°С‚Рё СЏРє РґРѕРґР°С‚РєРѕРІРёР№ "
            "РЅРµР»С–РЅС–Р№РЅРёР№ РјРµС‚РѕРґ Сѓ РєРѕРјРїР»РµРєСЃС– РјРѕРґРµР»РµР№."
        ),
        "alpha_beta": (
            "Alpha-beta recursive РЅР°РєРѕРїРёС‡СѓС” РїРѕС…РёР±РєСѓ С‡РµСЂРµР· velocity drift РЅР° РґРѕРІРіРѕРјСѓ "
            "РіРѕСЂРёР·РѕРЅС‚С–, С‚РѕРґС– СЏРє alpha-beta one-step РґРѕР±СЂРµ РїС–РґС…РѕРґРёС‚СЊ РґР»СЏ "
            "online-Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ С‚Р° РєРѕСЂРѕС‚РєРѕСЃС‚СЂРѕРєРѕРІРѕРіРѕ РїСЂРѕРіРЅРѕР·Сѓ."
        ),
        "complex_method_recommendation": (
            "Р”Р»СЏ РїСЂР°РєС‚РёС‡РЅРѕРіРѕ РІРёРєРѕСЂРёСЃС‚Р°РЅРЅСЏ РІР°СЂС‚Рѕ РєРѕРјР±С–РЅСѓРІР°С‚Рё РѕС‡РёС‰РµРЅРЅСЏ Р°РЅРѕРјР°Р»С–Р№, "
            "EMA Р°Р±Рѕ alpha-beta one-step РґР»СЏ online-Р·РіР»Р°РґР¶СѓРІР°РЅРЅСЏ, local polynomial "
            "РґР»СЏ Р»РѕРєР°Р»СЊРЅРёС… С‚СЂРµРЅРґС–РІ С– MLP СЏРє РґРѕРґР°С‚РєРѕРІСѓ РїРµСЂРµРІС–СЂРєСѓ РЅРµР»С–РЅС–Р№РЅРёС… Р·Р°Р»РµР¶РЅРѕСЃС‚РµР№."
        ),
    }
    model_recommendations_path.parent.mkdir(parents=True, exist_ok=True)
    model_recommendations_path.write_text(
        json.dumps(model_recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    saved_forecast_figure = save_forecast_comparison_plot(
        dates=test_data[date_column],
        y_true=test_series,
        forecasts={
            "Naive one-step": baseline_predictions,
            "Naive recursive": baseline_recursive_predictions,
            f"Global poly d={polynomial_result.best_degree}": polynomial_predictions,
            (
                f"Local poly N={local_polynomial_result.best_window}, "
                f"d={local_polynomial_result.best_degree}"
            ): local_polynomial_predictions,
            (
                f"Local poly one-step N={local_polynomial_one_step_result.best_window}, "
                f"d={local_polynomial_one_step_result.best_degree}"
            ): local_polynomial_one_step_predictions,
            f"MA recursive w={moving_average_result.best_parameter}": moving_average_predictions,
            f"EMA one-step s={ema_one_step_result.best_parameter}": ema_one_step_predictions,
            (
                f"MLP one-step k={deep_learning_result.best_window_size}"
            ): deep_learning_predictions,
            (
                "Alpha-beta one-step "
                f"a={alpha_beta_result['best_alpha']}, b={alpha_beta_result['best_beta']}"
            ): alpha_beta_one_step_predictions,
        },
        output_path=forecast_comparison_path,
    )
    saved_forecast_best_figure = save_forecast_comparison_plot(
        dates=test_data[date_column],
        y_true=test_series,
        forecasts={
            "Naive one-step": baseline_predictions,
            f"EMA one-step s={ema_one_step_result.best_parameter}": ema_one_step_predictions,
            (
                "Alpha-beta one-step "
                f"a={alpha_beta_result['best_alpha']}, b={alpha_beta_result['best_beta']}"
            ): alpha_beta_one_step_predictions,
            f"MA one-step w={moving_average_one_step_result.best_parameter}": (
                moving_average_one_step_predictions
            ),
            (
                f"MLP one-step k={deep_learning_result.best_window_size}"
            ): deep_learning_predictions,
            (
                f"Local poly one-step N={local_polynomial_one_step_result.best_window}, "
                f"d={local_polynomial_one_step_result.best_degree}"
            ): local_polynomial_one_step_predictions,
        },
        output_path=forecast_comparison_best_path,
        title="Best One-Step Forecast Comparison",
    )
    saved_alpha_beta_figure = save_forecast_comparison_plot(
        dates=test_data[date_column],
        y_true=test_series,
        forecasts={
            (
                "Alpha-beta recursive "
                f"a={alpha_beta_result['best_alpha']}, b={alpha_beta_result['best_beta']}"
            ): alpha_beta_recursive_predictions,
            (
                "Alpha-beta one-step "
                f"a={alpha_beta_result['best_alpha']}, b={alpha_beta_result['best_beta']}"
            ): alpha_beta_one_step_predictions,
        },
        output_path=alpha_beta_forecast_path,
        title="Alpha-Beta Filter Forecast",
    )
    saved_deep_learning_figure = save_deep_learning_forecast_plot(
        dates=test_data[date_column],
        y_true=test_series,
        y_pred=deep_learning_predictions,
        output_path=deep_learning_forecast_path,
    )
    saved_global_polynomial_selection_figure = save_metric_selection_plot(
        metrics_by_parameter=polynomial_result.metrics_by_degree,
        output_path=global_polynomial_selection_path,
        parameter_label="Polynomial degree",
        title="Global Polynomial Degree Selection",
    )
    saved_local_polynomial_selection_figure = save_local_polynomial_top_configs_plot(
        metrics_by_configuration=local_polynomial_one_step_result.metrics_by_configuration,
        output_path=local_polynomial_selection_path,
        top_n=10,
        title="Top-10 Local Polynomial One-Step Configurations",
    )
    saved_approximation_selection_figure = save_approximation_selection_plot(
        moving_average_metrics=moving_average_result.metrics_by_parameter,
        ema_metrics=ema_result.metrics_by_parameter,
        moving_average_one_step_metrics=moving_average_one_step_result.metrics_by_parameter,
        ema_one_step_metrics=ema_one_step_result.metrics_by_parameter,
        output_path=approximation_selection_path,
    )

    return {
        "dataset": str(dataset_path),
        "rows": len(cleaned_data),
        "date_column": date_column,
        "value_column": value_column,
        "statistics": stats,
        "anomaly_report": anomaly_report,
        "processed_data": str(processed_path),
        "anomaly_cleaned_data": str(anomaly_processed_path),
        "figure": str(saved_figure),
        "anomalies_figure": str(saved_anomalies_figure),
        "cleaned_comparison_figure": str(saved_comparison_figure),
        "forecast_metrics": str(forecast_metrics_path),
        "alpha_beta_metrics": str(alpha_beta_metrics_path),
        "model_recommendations": str(model_recommendations_path),
        "forecast_comparison_figure": str(saved_forecast_figure),
        "forecast_comparison_best_figure": str(saved_forecast_best_figure),
        "alpha_beta_forecast_figure": str(saved_alpha_beta_figure),
        "deep_learning_forecast_figure": str(saved_deep_learning_figure),
        "global_polynomial_selection_figure": str(saved_global_polynomial_selection_figure),
        "local_polynomial_selection_figure": str(saved_local_polynomial_selection_figure),
        "approximation_selection_figure": str(saved_approximation_selection_figure),
        "metrics": str(metrics_path),
        "anomaly_metrics": str(anomaly_report_path),
        "adaptive_anomaly_report": str(adaptive_anomaly_report_path),
        "adaptive_anomalies_figure": str(saved_adaptive_anomalies_figure),
        "anomaly_methods_comparison_figure": str(saved_anomaly_methods_comparison_figure),
    }


def main() -> None:
    """Execute the pipeline and print a compact run summary."""
    summary = run_pipeline()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()












