"""Métricas de evaluación y estadísticas para predicción de tiempos de viaje."""

import math
from typing import Dict, Sequence, Union

import numpy as np


def mse(preds: Sequence[float], targets: Sequence[float]) -> float:
    """Mean Squared Error."""
    p, t = np.asarray(preds), np.asarray(targets)
    if len(p) == 0:
        return 0.0
    return float(np.mean((p - t) ** 2))


def rmse(preds: Sequence[float], targets: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    return math.sqrt(mse(preds, targets))


def mae(preds: Sequence[float], targets: Sequence[float]) -> float:
    """Mean Absolute Error."""
    p, t = np.asarray(preds), np.asarray(targets)
    if len(p) == 0:
        return 0.0
    return float(np.mean(np.abs(p - t)))


def mape(preds: Sequence[float], targets: Sequence[float]) -> float:
    """Mean Absolute Percentage Error (%).

    Excluye targets == 0 para evitar divisiones por cero.
    """
    p, t = np.asarray(preds, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    mask = t != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100.0)


def r2_score(preds: Sequence[float], targets: Sequence[float]) -> float:
    """Coeficiente de determinación R²."""
    p, t = np.asarray(preds), np.asarray(targets)
    if len(p) == 0:
        return 0.0
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(
    preds: Sequence[float], targets: Sequence[float]
) -> Dict[str, float]:
    """Calcula todas las métricas de regresión."""
    return {
        "mse": mse(preds, targets),
        "rmse": rmse(preds, targets),
        "mae": mae(preds, targets),
        "mape": mape(preds, targets),
        "r2": r2_score(preds, targets),
    }


def travel_time_stats(
    preds: Sequence[float], targets: Sequence[float]
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Estadísticas descriptivas de predicciones vs. targets.

    Returns:
        Dict con métricas de regresión + estadísticas descriptivas
        de predicciones y targets (media, mediana, std, percentiles).
    """
    p, t = np.asarray(preds), np.asarray(targets)

    def _describe(arr: np.ndarray, label: str) -> Dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            f"{label}_mean": float(np.mean(arr)),
            f"{label}_median": float(np.median(arr)),
            f"{label}_std": float(np.std(arr)),
            f"{label}_min": float(np.min(arr)),
            f"{label}_max": float(np.max(arr)),
            f"{label}_p25": float(np.percentile(arr, 25)),
            f"{label}_p75": float(np.percentile(arr, 75)),
            f"{label}_p90": float(np.percentile(arr, 90)),
        }

    result = compute_all_metrics(preds, targets)
    result.update(_describe(p, "pred"))
    result.update(_describe(t, "target"))

    # Error distribution
    errors = p - t
    result.update(_describe(errors, "error"))

    return result
