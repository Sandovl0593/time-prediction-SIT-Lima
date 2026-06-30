"""Análisis derivado del CSV maestro de rutas candidatas.

Carga `src/outputs/routes/route_candidates.csv` y genera vistas derivadas:

- routes_summary.csv      : estadísticas por configuración/escenario
- routes_distribution.csv : distribución de longitudes y scores
- straight_routes.csv     : subconjunto filtrado por índice de rectitud
- straight_routes_by_config.json : recuento de rutas rectas por escenario
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.config import Config, STRAIGHT_THRESHOLD

# Rutas por defecto (relativas a la raíz del proyecto)
_DEFAULT_MASTER_CSV = Path("src") / "outputs" / "routes" / "route_candidates.csv"
_DEFAULT_OUTPUT_DIR = Path("src") / "outputs" / "routes"


def load_master_csv(master_csv: Optional[Path] = None) -> pd.DataFrame:
    """Carga el CSV maestro de rutas candidatas.

    Args:
        master_csv: Ruta al archivo. Si es None usa la ruta por defecto.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(master_csv) if master_csv is not None else _DEFAULT_MASTER_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"CSV maestro no encontrado: {path}. "
            "Ejecuta primero el pipeline con --process-nyc para generarlo."
        )
    df = pd.read_csv(path)
    return df


def build_routes_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Genera estadísticas agregadas por escenario.

    Columnas del resultado: scenario_id, total_candidates, accepted_count,
    acceptance_rate, mean_score, median_score, mean_length_real_km,
    mean_straightness_index.
    """
    agg = (
        df.groupby("scenario_id")
        .agg(
            total_candidates=("score", "count"),
            accepted_count=("accepted_by_tolerance", lambda s: s.fillna(False).sum()),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            mean_length_real_km=("length_real_km", "mean"),
            mean_straightness_index=("straightness_index", "mean"),
        )
        .reset_index()
    )
    agg["acceptance_rate"] = (
        agg["accepted_count"] / agg["total_candidates"].replace(0, np.nan)
    )
    return agg


def build_routes_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Genera distribución de rutas agrupada por bin de criterio km (tol_prox).

    Los bins fijos son km = 1, 2, 5, 10, 15, 20, correspondientes a los
    valores de `tol_prox` asignados a cada ruta candidata.

    Columnas del resultado: km_bin, count, accepted_count, acceptance_rate,
    mean_score, std_score, mean_km_offset, std_km_offset,
    mean_straightness_index.
    """
    if "tol_prox" not in df.columns:
        return pd.DataFrame()

    dist = (
        df.groupby("tol_prox")
        .agg(
            count=("score", "count"),
            accepted_count=("accepted_by_tolerance", lambda s: s.fillna(False).sum()),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            mean_km_offset=("km_offset", "mean"),
            std_km_offset=("km_offset", "std"),
            mean_straightness_index=("straightness_index", "mean"),
        )
        .reset_index()
        .rename(columns={"tol_prox": "km_bin"})
    )
    dist["acceptance_rate"] = dist["accepted_count"] / dist["count"].replace(0, np.nan)
    return dist


def run_route_analysis(
    master_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None
    # straightness_threshold: Optional[float] = None,
) -> dict:
    """Ejecuta el análisis completo y guarda todos los artefactos derivados.

    Artefactos generados en `output_dir` (por defecto src/outputs/routes/):
        - routes_summary.csv
        - routes_distribution.csv
        - straight_routes.csv
        - straight_routes_by_config.json

    Después genera el CSV maestro de segmentos en src/topsegments/:
        - master_segments.csv
        - config_export_summary.json

    Returns:
        Diccionario con rutas de los artefactos generados.
    """
    from src.routes.export_top_segments import export_master_segments

    out_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_master_csv(master_csv)

    artifacts: dict = {}

    # routes_summary.csv
    summary = build_routes_summary(df)
    summary_path = out_dir / "routes_summary.csv"
    summary.to_csv(summary_path, index=False)
    artifacts["routes_summary"] = str(summary_path)

    # routes_distribution.csv
    distribution = build_routes_distribution(df)
    dist_path = out_dir / "routes_distribution.csv"
    distribution.to_csv(dist_path, index=False)
    artifacts["routes_distribution"] = str(dist_path)

    return artifacts


if __name__ == "__main__":
    result = run_route_analysis()
    print("Artefactos generados:")
    for key, path in result.items():
        print(f"  {key}: {path}")
