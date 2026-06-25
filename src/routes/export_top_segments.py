"""Export de segmentos filtrados por configuración (A / B / C).

Input : src/outputs/routes/straight_routes.csv
        (rutas ya filtradas por índice de rectitud ≥ straightness_threshold)

Output: src/topsegments/config_A.csv
        src/topsegments/config_B.csv
        src/topsegments/config_C.csv

Criterio de selección (búsqueda sin límite de cantidad):
---------------------------------------------------------
Para cada configuración X con parámetros (km_tolerance, curve_penalty):

  1. km_filter:
         abs(km_offset) / (tol_prox + ε) ≤ km_tolerance

  2. score_filter:
         config_score ≥ score_threshold
         donde:
           config_score    = straightness_index − curve_penalty · |km_offset| / (tol_prox + ε)
           score_threshold = straightness_threshold − curve_penalty · km_tolerance

     La score_threshold representa el mínimo esperable para un segmento
     perfectamente recto (straightness_index = straightness_threshold) que
     se encuentra exactamente en el borde de la tolerancia km.

Rangos de parámetros (definidos en config.CURVE_PENALTY_RANGE / KM_TOLERANCE_RANGE):
  curve_penalty  : 0.5 (ideal, penaliza fuertemente curvas) → 0.1 (peor caso realista)
  km_tolerance   : 0.2 (ideal, ±20% del bin)               → 0.5 (peor caso realista, ±50%)

Las configuraciones A/B/C están dentro de ese rango; cuanto más estricta
la config, menor será el conjunto de candidatos pero de mayor calidad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import ROUTE_CONFIGS, Config

_STRAIGHT_CSV_DEFAULT = Path("src") / "outputs" / "routes" / "straight_routes.csv"
_TOPSEGMENTS_DIR = Path("src") / "topsegments"
_EPS = 1e-9


def _compute_config_score(df: pd.DataFrame, curve_penalty: float) -> pd.Series:
    """config_score = straightness_index − curve_penalty · |km_offset| / (tol_prox + ε)"""
    return (
        df["straightness_index"]
        - curve_penalty * df["km_offset"].abs() / (df["tol_prox"] + _EPS)
    )


def _score_threshold(straightness_threshold: float, curve_penalty: float, km_tolerance: float) -> float:
    """Umbral mínimo de config_score para la configuración.

    Equivale al score de un segmento que tiene straightness_index exactamente en
    el umbral de rectitud y km_offset exactamente en el límite de la tolerancia km.
    """
    return straightness_threshold - curve_penalty * km_tolerance


def filter_segments_for_config(
    df_straight: pd.DataFrame,
    km_tolerance: float,
    curve_penalty: float,
    straightness_threshold: float = 0.9,
) -> pd.DataFrame:
    """Filtra straight_routes para una configuración concreta.

    Aplica km_filter y score_filter sin límite de cantidad de tramos.
    Añade columnas trazables: config_score, score_threshold,
    config_km_tolerance, config_curve_penalty.
    """
    df = df_straight.copy()

    rel_km_error = df["km_offset"].abs() / (df["tol_prox"] + _EPS)
    km_mask = rel_km_error <= km_tolerance

    config_scores = _compute_config_score(df, curve_penalty)
    thresh = _score_threshold(straightness_threshold, curve_penalty, km_tolerance)
    score_mask = config_scores >= thresh

    keep = km_mask & score_mask
    result = df[keep].copy()
    result["config_score"] = config_scores[keep].values
    result["score_threshold"] = thresh
    result["config_km_tolerance"] = km_tolerance
    result["config_curve_penalty"] = curve_penalty

    return result.sort_values("config_score", ascending=False).reset_index(drop=True)


def export_segments_by_config(
    straight_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    route_configs: Optional[Dict] = None,
    straightness_threshold: Optional[float] = None,
) -> Dict[str, str]:
    """Exporta segmentos filtrados para cada configuración (A, B, C).

    Lee straight_routes.csv, aplica los filtros de cada config y escribe:
        src/topsegments/config_A.csv
        src/topsegments/config_B.csv
        src/topsegments/config_C.csv
        src/topsegments/config_export_summary.json

    Returns:
        Diccionario {config_X: path_csv, config_export_summary: path_json}.
    """
    src_path = Path(straight_csv) if straight_csv is not None else _STRAIGHT_CSV_DEFAULT
    out_dir = Path(output_dir) if output_dir is not None else _TOPSEGMENTS_DIR
    configs = route_configs if route_configs is not None else ROUTE_CONFIGS
    s_threshold = (
        straightness_threshold
        if straightness_threshold is not None
        else Config().straightness_threshold
    )

    if not src_path.exists():
        raise FileNotFoundError(
            f"straight_routes.csv no encontrado en: {src_path}. "
            "Ejecuta primero --route-analysis para generarlo."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    df_straight = pd.read_csv(src_path)

    artifacts: Dict[str, str] = {}
    summary: Dict[str, dict] = {}

    for letter, params in configs.items():
        km_tol = float(params["km_tolerance"])
        pen = float(params["curve_penalty"])
        thresh = _score_threshold(s_threshold, pen, km_tol)

        df_filtered = filter_segments_for_config(
            df_straight,
            km_tolerance=km_tol,
            curve_penalty=pen,
            straightness_threshold=s_threshold,
        )

        out_path = out_dir / f"config_{letter}.csv"
        df_filtered.to_csv(out_path, index=False)
        artifacts[f"config_{letter}"] = str(out_path)

        summary[letter] = {
            "km_tolerance": km_tol,
            "curve_penalty": pen,
            "score_threshold": thresh,
            "total_straight_in": len(df_straight),
            "segments_exported": len(df_filtered),
            "mean_config_score": (
                float(df_filtered["config_score"].mean()) if not df_filtered.empty else None
            ),
            "mean_straightness_index": (
                float(df_filtered["straightness_index"].mean()) if not df_filtered.empty else None
            ),
            "out_path": str(out_path),
        }

    summary_path = out_dir / "config_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    artifacts["config_export_summary"] = str(summary_path)

    return artifacts
