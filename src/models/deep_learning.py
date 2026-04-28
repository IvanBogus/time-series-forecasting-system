"""MLP-based one-step forecasting for time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from src.evaluation import evaluate_forecast

DeepLearningBackend = Literal["tensorflow_keras", "sklearn_mlp"]


@dataclass(frozen=True)
class DeepLearningSelectionResult:
    """Result of deep learning window-size selection."""

    backend: DeepLearningBackend
    best_window_size: int
    predictions: np.ndarray
    metrics_by_window: dict[str, dict[str, float]]


def create_sliding_window_dataset(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create supervised learning samples from a one-dimensional time series."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if len(values) <= window_size:
        raise ValueError("Series length must be greater than window_size.")

    features: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(window_size, len(values)):
        features.append(values[index - window_size : index])
        targets.append(float(values[index]))

    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def _build_keras_model(window_size: int) -> object:
    """Build a small Keras MLP model."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _fit_keras_model(features: np.ndarray, targets: np.ndarray, window_size: int) -> object:
    """Fit a Keras MLP with early stopping."""
    from tensorflow import keras

    model = _build_keras_model(window_size)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )
    model.fit(
        features,
        targets,
        epochs=300,
        batch_size=16,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )
    return model


def _fit_sklearn_model(features: np.ndarray, targets: np.ndarray) -> MLPRegressor:
    """Fit an sklearn MLPRegressor fallback model."""
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=25,
        max_iter=2000,
        random_state=42,
        shuffle=False,
    )
    model.fit(features, targets)
    return model


def _predict_one_step_scaled(
    model: object,
    backend: DeepLearningBackend,
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Run rolling one-step predictions in scaled space."""
    history = train_scaled.astype(float).tolist()
    predictions: list[float] = []

    for actual_value in test_scaled:
        feature = np.asarray(history[-window_size:], dtype=float).reshape(1, -1)
        if backend == "tensorflow_keras":
            prediction = float(model.predict(feature, verbose=0)[0, 0])
        else:
            prediction = float(model.predict(feature)[0])
        predictions.append(prediction)
        history.append(float(actual_value))

    return np.asarray(predictions, dtype=float)


def mlp_one_step_forecast(
    train: pd.Series,
    test: pd.Series,
    window_size: int,
    prefer_tensorflow: bool = True,
) -> tuple[np.ndarray, DeepLearningBackend]:
    """Train an MLP on sliding windows and predict the test split one step ahead."""
    train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)
    if len(train_values) <= window_size:
        raise ValueError("Train series length must be greater than window_size.")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values).ravel()
    test_scaled = scaler.transform(test_values).ravel()
    features, targets = create_sliding_window_dataset(train_scaled, window_size)

    backend: DeepLearningBackend
    try:
        if not prefer_tensorflow:
            raise ImportError("TensorFlow disabled by configuration.")
        model = _fit_keras_model(features, targets, window_size)
        backend = "tensorflow_keras"
    except ImportError:
        model = _fit_sklearn_model(features, targets)
        backend = "sklearn_mlp"

    predictions_scaled = _predict_one_step_scaled(
        model,
        backend,
        train_scaled,
        test_scaled,
        window_size,
    )
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
    return predictions, backend


def select_mlp_window_size(
    train: pd.Series,
    test: pd.Series,
    window_sizes: Iterable[int] = (3, 7, 14, 21),
) -> DeepLearningSelectionResult:
    """Select the best MLP sliding window size by validation RMSE."""
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    metrics_by_window: dict[str, dict[str, float]] = {}
    predictions_by_window: dict[int, np.ndarray] = {}
    backend_by_window: dict[int, DeepLearningBackend] = {}

    for window_size in window_sizes:
        predictions, backend = mlp_one_step_forecast(train, test, window_size)
        predictions_by_window[window_size] = predictions
        backend_by_window[window_size] = backend
        metrics_by_window[str(window_size)] = evaluate_forecast(test_values, predictions)

    best_window_size = int(
        min(metrics_by_window, key=lambda key: metrics_by_window[key]["RMSE"])
    )
    return DeepLearningSelectionResult(
        backend=backend_by_window[best_window_size],
        best_window_size=best_window_size,
        predictions=predictions_by_window[best_window_size],
        metrics_by_window=metrics_by_window,
    )
