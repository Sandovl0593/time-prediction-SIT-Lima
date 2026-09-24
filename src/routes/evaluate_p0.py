"""Calcula el reporte reproducible de la prueba P0 desde predicciones CSV.

P0 evalúa la configuración base con las rutas que tienen simultáneamente
objetivo y predicción cubiertos. El reporte describe target--pred por modelo y
por bin de kilometraje, sin usar indicadores de curvatura o densidad.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GAT_PATH = Path("src/outputs/all_routes_gat.csv")
DEFAULT_GATV2_PATH = Path("src/outputs/all_routes_gatv2.csv")
DEFAULT_OUTPUT_PATH = Path("src/outputs/lima/p0_metrics.csv")
REQUIRED_COLUMNS = {"target", "tol_prox", "has_target", "covered"}


def _as_boolean(series: pd.Series) -> pd.Series:
    """Interpret boolean CSV columns without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "si", "sí"})


def _metrics(frame: pd.DataFrame, prediction_column: str = "pred") -> dict[str, float | int]:
    target = frame["target"].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    residual = prediction - target
    squared_error = float(np.square(residual).sum())
    target_variation = float(np.square(target - target.mean()).sum())
    return {
        "n_routes": len(frame),
        "mean_target_s": float(target.mean()),
        "mean_prediction_s": float(prediction.mean()),
        "mean_bias_s": float(residual.mean()),
        "rmse_s": float(np.sqrt(np.mean(np.square(residual)))),
        "mae_s": float(np.mean(np.abs(residual))),
        "mape_pct": float(np.mean(np.abs(residual) / np.maximum(np.abs(target), 1e-12)) * 100),
        "r2": float(1 - squared_error / target_variation) if target_variation else float("nan"),
    }


def load_valid_predictions(
    predictions_path: Path,
    prediction_column: str = "pred",
) -> pd.DataFrame:
    """Load the covered P0 rows that have finite target and prediction values."""
    predictions_path = Path(predictions_path)
    data = pd.read_csv(predictions_path, low_memory=False)
    missing = (REQUIRED_COLUMNS | {prediction_column}).difference(data.columns)
    if missing:
        raise ValueError(f"{predictions_path}: faltan columnas {sorted(missing)}")

    target = pd.to_numeric(data["target"], errors="coerce")
    prediction = pd.to_numeric(data[prediction_column], errors="coerce")
    valid = data.loc[
        _as_boolean(data["has_target"])
        & _as_boolean(data["covered"])
        & np.isfinite(target)
        & np.isfinite(prediction)
    ].copy()
    valid["target"] = target.loc[valid.index]
    valid[prediction_column] = prediction.loc[valid.index]
    if valid.empty:
        raise ValueError(
            f"{predictions_path}: no hay filas P0 cubiertas con target y "
            f"{prediction_column} finitos"
        )
    return valid


def evaluate_predictions(
    predictions_path: Path,
    model: str,
    prediction_column: str = "pred",
) -> pd.DataFrame:
    """Return global and per-bin P0 metrics for one model prediction file."""
    valid = load_valid_predictions(predictions_path, prediction_column)

    rows = [{"model": model, "bin_km": "global", **_metrics(valid, prediction_column)}]
    for bin_km, group in valid.groupby("tol_prox", sort=True):
        rows.append(
            {
                "model": model,
                "bin_km": float(bin_km),
                **_metrics(group, prediction_column),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gat", type=Path, default=DEFAULT_GAT_PATH, help="CSV de rutas GAT para P0")
    parser.add_argument("--gatv2", type=Path, default=DEFAULT_GATV2_PATH, help="CSV de rutas GATv2 para P0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV de resumen que se generará")
    parser.add_argument(
        "--prediction-column",
        default="pred",
        help="Columna que se evaluará como predicción (por ejemplo, pred o sum_edge_predictions_s)",
    )
    args = parser.parse_args()

    report = pd.concat(
        [
            evaluate_predictions(args.gat, "GAT", args.prediction_column),
            evaluate_predictions(args.gatv2, "GATv2", args.prediction_column),
        ],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Reporte P0 escrito en {args.output} ({len(report)} filas)")


if __name__ == "__main__":
    main()
