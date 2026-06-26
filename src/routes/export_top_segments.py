"""Export del CSV maestro de segmentos filtrados con parámetros ideales.

Input : src/outputs/routes/straight_routes.csv
        (rutas ya filtradas por índice de rectitud ≥ straightness_threshold)

Output: src/topsegments/master_segments.csv
        src/topsegments/config_export_summary.json

Criterio de selección (búsqueda sin límite de cantidad):
---------------------------------------------------------
Parámetros únicos e ideales (IDEAL_CURVE_PENALTY, IDEAL_KM_TOLERANCE):

  1. km_filter:
         abs(km_offset) / (tol_prox + ε) ≤ IDEAL_KM_TOLERANCE

  2. score_filter:
         config_score ≥ score_threshold
         donde:
           config_score    = straightness_index − IDEAL_CURVE_PENALTY · |km_offset| / (tol_prox + ε)
           score_threshold = straightness_threshold − IDEAL_CURVE_PENALTY · IDEAL_KM_TOLERANCE

No se utilizan configuraciones experimentales múltiples (A/B/C).
La agrupación posterior se realiza por tol_prox y km_offset derivados del CSV maestro.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import IDEAL_CURVE_PENALTY, IDEAL_KM_TOLERANCE, Config

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


def export_master_segments(
    straight_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    curve_penalty: float = IDEAL_CURVE_PENALTY,
    km_tolerance: float = IDEAL_KM_TOLERANCE,
    straightness_threshold: Optional[float] = None,
) -> Dict[str, str]:
    """Exporta el CSV maestro de segmentos filtrados con parámetros ideales.

    Lee straight_routes.csv, aplica los filtros con IDEAL_CURVE_PENALTY e
    IDEAL_KM_TOLERANCE y escribe un único CSV maestro:
        src/topsegments/master_segments.csv
        src/topsegments/config_export_summary.json

    Toda agrupación posterior (por tol_prox, km_offset) debe derivarse
    de este CSV maestro en lugar de reconstruir configuraciones artificiales.

    Returns:
        Diccionario {master_segments: path_csv, config_export_summary: path_json}.
    """
    src_path = Path(straight_csv) if straight_csv is not None else _STRAIGHT_CSV_DEFAULT
    out_dir = Path(output_dir) if output_dir is not None else _TOPSEGMENTS_DIR
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

    thresh = _score_threshold(s_threshold, curve_penalty, km_tolerance)
    df_master = filter_segments_for_config(
        df_straight,
        km_tolerance=km_tolerance,
        curve_penalty=curve_penalty,
        straightness_threshold=s_threshold,
    )

    out_path = out_dir / "master_segments.csv"
    df_master.to_csv(out_path, index=False)

    summary = {
        "master": {
            "km_tolerance": km_tolerance,
            "curve_penalty": curve_penalty,
            "score_threshold": thresh,
            "total_straight_in": len(df_straight),
            "segments_exported": len(df_master),
            "mean_config_score": (
                float(df_master["config_score"].mean()) if not df_master.empty else None
            ),
            "mean_straightness_index": (
                float(df_master["straightness_index"].mean()) if not df_master.empty else None
            ),
            "out_path": str(out_path),
        }
    }

    summary_path = out_dir / "config_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    return {
        "master_segments": str(out_path),
        "config_export_summary": str(summary_path),
    }
