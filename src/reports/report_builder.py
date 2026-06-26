"""Módulo de reporte: agrega artefactos de entrenamiento en tex_data.json.

Lee los artefactos generados por trainer.py y produce un JSON estructurado
listo para alimentar las tablas y figuras del documento LaTeX.

Estructura de directorios esperada (outputs de trainer.py):
    src/outputs/training/<model>/<timestamp>/
        final_metrics.json          — métricas generales sobre test split
        test_predictions.csv        — edge_idx, pred, target
        eval_master/
            metrics.json            — métricas sobre test ∩ master_segments
            predictions.csv         — edge_idx, u, v, pred, target, tol_prox, km_offset, ...
        eval_straight/              (misma estructura)

Artefacto de salida:
    src/outputs/reports/tex_data.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import KM_BINS, IDEAL_CURVE_PENALTY, IDEAL_KM_TOLERANCE

_TRAINING_BASE = Path("src") / "outputs" / "training"
_REPORTS_DIR = Path("src") / "outputs" / "reports"
_MODELS_ORDERED = ("graphsage", "gat", "gatv2")

# Rangos de r = 1 / straightness_index usados en tab:rmse_por_penalidad
_R_BINS = [
    ("r≤1.05",       0.0,   1.05),
    ("1.05<r≤1.10",  1.05,  1.10),
    ("1.10<r≤1.20",  1.10,  1.20),
    ("r>1.20",       1.20,  math.inf),
]


# ---------------------------------------------------------------------------
# Helpers de descubrimiento de corridas
# ---------------------------------------------------------------------------

def discover_latest_run(base_dir: Path, model: str) -> Optional[Path]:
    """Retorna el run_dir más reciente para un encoder bajo base_dir/<model>/.

    Busca subdirectorios con nombre tipo YYYYMMDD_HHMMSS que contengan
    final_metrics.json. Toma el más reciente por orden lexicográfico.
    """
    model_dir = Path(base_dir) / model
    if not model_dir.is_dir():
        return None
    candidates = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and (d / "final_metrics.json").exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_run_metrics(run_dir: Path) -> dict:
    """Lee final_metrics.json de un run_dir."""
    path = Path(run_dir) / "final_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"final_metrics.json no encontrado en {run_dir}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_subset_metrics(run_dir: Path, subset_name: str) -> Optional[dict]:
    """Lee eval_{subset_name}/metrics.json. Retorna None si no existe o fue omitido."""
    path = Path(run_dir) / f"eval_{subset_name}" / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("skipped"):
        return None
    return data


def load_subset_predictions(run_dir: Path, subset_name: str) -> Optional[pd.DataFrame]:
    """Lee eval_{subset_name}/predictions.csv. Retorna None si no existe."""
    path = Path(run_dir) / f"eval_{subset_name}" / "predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(str(path), low_memory=False)


# ---------------------------------------------------------------------------
# Constructores de tablas individuales
# ---------------------------------------------------------------------------

def build_encoder_comparison_table(
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """tab:metricas_general — 3 encoders evaluados sobre el test set completo.

    Returns:
        List de dicts: [{encoder, mse, rmse, mae, mape, r2, n_test}, ...]
    """
    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    rows = []
    for model in _MODELS_ORDERED:
        run_dir = discover_latest_run(base_dir, model)
        if run_dir is None:
            rows.append({"encoder": model, "error": "run_not_found"})
            continue
        m = load_run_metrics(run_dir)
        rows.append({
            "encoder": model,
            "mse":    round(float(m.get("mse",  float("nan"))), 4),
            "rmse":   round(float(m.get("rmse", float("nan"))), 4),
            "mae":    round(float(m.get("mae",  float("nan"))), 4),
            "mape":   round(float(m.get("mape", float("nan"))), 4),
            "r2":     round(float(m.get("r2",   float("nan"))), 4),
            "n_test": int(m.get("num_test_edges", 0)),
        })
    return rows


def build_config_comparison_table(
    model: str = "gatv2",
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """tab:metricas_configs — encoder evaluado sobre eval_master agrupado por tol_prox.

    Carga las predicciones de eval_master/predictions.csv y agrupa por bin de
    tol_prox (km), calculando RMSE, MAE y R² por cada bin. Esta tabla reemplaza
    la comparación artificial A/B/C por agrupación metodológica por kilometraje.

    Returns:
        List de dicts: [{tol_prox_km, n_eval, rmse, mae, mape, r2}, ...]
    """
    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    run_dir = discover_latest_run(base_dir, model)
    rows = []

    if run_dir is None:
        return [{"error": "run_not_found", "model": model}]

    preds_df = load_subset_predictions(run_dir, "master")
    if preds_df is None:
        return [{"error": "eval_master_not_found_or_skipped", "model": model}]

    preds_df = preds_df.dropna(subset=["pred", "target"])
    if "tol_prox" not in preds_df.columns:
        return [{"error": "tol_prox_column_missing", "model": model}]

    for km_bin in KM_BINS:
        sub = preds_df[preds_df["tol_prox"] == km_bin]
        if sub.empty:
            continue
        residuals = (sub["pred"].values - sub["target"].values) ** 2
        rmse = float(np.sqrt(residuals.mean()))
        mae  = float(np.abs(sub["pred"].values - sub["target"].values).mean())
        ss_res = residuals.sum()
        ss_tot = ((sub["target"].values - sub["target"].values.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # MAPE: ignorar targets cero
        tgt = sub["target"].values
        prd = sub["pred"].values
        nonzero = tgt != 0
        mape = float(np.abs((prd[nonzero] - tgt[nonzero]) / tgt[nonzero]).mean() * 100) if nonzero.any() else float("nan")
        rows.append({
            "tol_prox_km": km_bin,
            "n_eval":      len(sub),
            "rmse":        round(rmse, 4),
            "mae":         round(mae,  4),
            "mape":        round(mape, 4),
            "r2":          round(r2,   4),
        })
    return rows


def build_encoder_comparison_straight(
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """tab:metricas_rectas — 3 encoders evaluados sobre el subconjunto de rutas rectas.

    Returns:
        List de dicts: [{encoder, mse, rmse, mae, mape, r2, n_eval}, ...]
    """
    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    rows = []
    for model in _MODELS_ORDERED:
        run_dir = discover_latest_run(base_dir, model)
        entry: Dict = {"encoder": model}
        if run_dir is None:
            entry["error"] = "run_not_found"
            rows.append(entry)
            continue
        m = load_subset_metrics(run_dir, "straight")
        if m is None:
            entry["error"] = "eval_not_found_or_skipped"
            rows.append(entry)
            continue
        entry.update({
            "n_eval": int(m.get("n_eval", 0)),
            "mse":    round(float(m.get("mse",  float("nan"))), 4),
            "rmse":   round(float(m.get("rmse", float("nan"))), 4),
            "mae":    round(float(m.get("mae",  float("nan"))), 4),
            "mape":   round(float(m.get("mape", float("nan"))), 4),
            "r2":     round(float(m.get("r2",   float("nan"))), 4),
        })
        rows.append(entry)
    return rows


def build_straightness_breakdown_table(
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """tab:rmse_por_penalidad — RMSE por rango de r para los 3 encoders en eval_master.

    r = 1 / straightness_index  (r=1 = perfectamente recto, r>1 = más curvo)

    Rangos: r≤1.05 | (1.05,1.10] | (1.10,1.20] | >1.20

    Returns:
        List de dicts: [{r_range, r_min, r_max, n, rmse_graphsage, rmse_gat, rmse_gatv2}, ...]
    """
    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    subset_name = "master"

    # Cargar predicciones de los 3 encoders
    model_dfs: Dict[str, Optional[pd.DataFrame]] = {}
    for model in _MODELS_ORDERED:
        run_dir = discover_latest_run(base_dir, model)
        model_dfs[model] = load_subset_predictions(run_dir, subset_name) if run_dir else None

    rows = []
    for r_label, r_min, r_max in _R_BINS:
        entry: Dict = {
            "r_range": r_label,
            "r_min": r_min,
            "r_max": r_max if not math.isinf(r_max) else None,
        }
        n_ref = None
        for model in _MODELS_ORDERED:
            df = model_dfs[model]
            if df is None or "straightness_index" not in df.columns:
                entry[f"rmse_{model}"] = None
                continue
            df = df.copy()
            df["_r"] = 1.0 / df["straightness_index"].replace(0, np.nan)
            mask = (df["_r"] > r_min) & (df["_r"] <= r_max) if r_max != math.inf \
                   else (df["_r"] > r_min)
            if r_min == 0.0:
                mask = df["_r"] <= r_max
            sub = df[mask].dropna(subset=["pred", "target"])
            if sub.empty:
                entry[f"rmse_{model}"] = None
                continue
            residuals = (sub["pred"].values - sub["target"].values) ** 2
            entry[f"rmse_{model}"] = round(float(np.sqrt(residuals.mean())), 4)
            if n_ref is None:
                n_ref = len(sub)
        entry["n"] = n_ref or 0
        rows.append(entry)
    return rows


def build_rmse_vs_r_data(
    r_step: float = 0.02,
    base_dir: Optional[Path] = None,
) -> List[Dict]:
    """fig:rmse_vs_r — RMSE vs r en buckets finos para los 3 encoders en eval_master.

    Returns:
        List de dicts: [{r_center, n, rmse_graphsage, rmse_gat, rmse_gatv2}, ...]
        ordenados por r_center ascendente.
    """
    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    subset_name = "master"

    model_dfs: Dict[str, Optional[pd.DataFrame]] = {}
    for model in _MODELS_ORDERED:
        run_dir = discover_latest_run(base_dir, model)
        model_dfs[model] = load_subset_predictions(run_dir, subset_name) if run_dir else None

    # Determinar rango de r desde datos disponibles
    all_r_vals = []
    for model, df in model_dfs.items():
        if df is not None and "straightness_index" in df.columns:
            r_vals = (1.0 / df["straightness_index"].replace(0, np.nan)).dropna()
            all_r_vals.extend(r_vals.tolist())
    if not all_r_vals:
        return []

    r_min_data = max(1.0, math.floor(min(all_r_vals) * 20) / 20)
    r_max_data = math.ceil(max(all_r_vals) * 20) / 20
    buckets = np.arange(r_min_data, r_max_data + r_step, r_step)

    rows = []
    for b_start in buckets[:-1]:
        b_end = b_start + r_step
        b_center = round((b_start + b_end) / 2, 4)
        entry: Dict = {"r_center": b_center}
        n_ref = None
        for model in _MODELS_ORDERED:
            df = model_dfs[model]
            if df is None or "straightness_index" not in df.columns:
                entry[f"rmse_{model}"] = None
                continue
            df = df.copy()
            df["_r"] = 1.0 / df["straightness_index"].replace(0, np.nan)
            sub = df[(df["_r"] >= b_start) & (df["_r"] < b_end)].dropna(subset=["pred", "target"])
            if sub.empty:
                entry[f"rmse_{model}"] = None
                continue
            residuals = (sub["pred"].values - sub["target"].values) ** 2
            entry[f"rmse_{model}"] = round(float(np.sqrt(residuals.mean())), 4)
            if n_ref is None:
                n_ref = len(sub)
        entry["n"] = n_ref or 0
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Generador de reporte completo
# ---------------------------------------------------------------------------

def generate_tex_data_report(
    output_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Genera src/outputs/reports/tex_data.json con todos los datos para el LaTeX.

    Claves del JSON:
        generated_at                   — timestamp ISO
        ideal_params                   — IDEAL_CURVE_PENALTY, IDEAL_KM_TOLERANCE
        encoder_comparison_general     — tabla tab:metricas_general
        config_comparison_gatv2        — tabla tab:metricas_configs (por tol_prox bin)
        encoder_comparison_straight    — tabla tab:metricas_rectas
        straightness_breakdown_master  — tabla tab:rmse_por_penalidad (eval_master)
        rmse_vs_r_master               — datos para fig:rmse_vs_r (eval_master)

    Returns:
        Path al archivo JSON generado.
    """
    import datetime

    base_dir = Path(base_dir) if base_dir is not None else _TRAINING_BASE
    output_path = Path(output_path) if output_path is not None else (_REPORTS_DIR / "tex_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ideal_params": {
            "curve_penalty": IDEAL_CURVE_PENALTY,
            "km_tolerance":  IDEAL_KM_TOLERANCE,
        },
        "encoder_comparison_general":  build_encoder_comparison_table(base_dir),
        "config_comparison_gatv2":     build_config_comparison_table("gatv2", base_dir),
        "encoder_comparison_straight": build_encoder_comparison_straight(base_dir),
        "straightness_breakdown_master": build_straightness_breakdown_table(base_dir),
        "rmse_vs_r_master":              build_rmse_vs_r_data(base_dir=base_dir),
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    print(f"[report_builder] tex_data.json generado en: {output_path}")
    return output_path
