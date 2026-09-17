"""Calcula el reporte reproducible de la prueba P0 desde predicciones CSV.

P0 evalúa la configuración base con las rutas que tienen simultáneamente
objetivo y predicción cubiertos. El reporte separa la comparación directa
target--pred por modelo y por bin de kilometraje, sin usar los indicadores de
curvatura como sustituto de la evaluación predictiva.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"target", "pred", "tol_prox", "has_target", "covered"}


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame["target"].to_numpy(dtype=float)
    prediction = frame["pred"].to_numpy(dtype=float)
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


def evaluate_predictions(predictions_path: Path, model: str) -> pd.DataFrame:
    """Return global and per-bin P0 metrics for one model prediction file."""
    data = pd.read_csv(predictions_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{predictions_path}: faltan columnas {sorted(missing)}")

    valid = data.loc[
        data["has_target"].fillna(False).astype(bool)
        & data["covered"].fillna(False).astype(bool)
        & data["target"].notna()
        & data["pred"].notna()
    ].copy()
    if valid.empty:
        raise ValueError(f"{predictions_path}: no hay filas P0 con target y predicción cubiertos")

    rows = [{"model": model, "bin_km": "global", **_metrics(valid)}]
    for bin_km, group in valid.groupby("tol_prox", sort=True):
        rows.append({"model": model, "bin_km": float(bin_km), **_metrics(group)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gat", type=Path, required=True, help="CSV de predicciones GAT para P0")
    parser.add_argument("--gatv2", type=Path, required=True, help="CSV de predicciones GATv2 para P0")
    parser.add_argument("--output", type=Path, required=True, help="CSV de resumen que se generará")
    args = parser.parse_args()

    report = pd.concat(
        [
            evaluate_predictions(args.gat, "GAT"),
            evaluate_predictions(args.gatv2, "GATv2"),
        ],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Reporte P0 escrito en {args.output} ({len(report)} filas)")


if __name__ == "__main__":
    main()
