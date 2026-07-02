"""Trainer unificado para modelos de predicción de tiempos de viaje.

Este módulo carga el grafo procesado desde los CSV exportados en
`src/data/processed/graph/`, lo convierte a `torch_geometric.data.Data`,
crea los masks train/val/test y entrena el encoder seleccionado.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx
from shapely import wkt
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from src.config import Config, EVAL_SCENARIOS, STRAIGHT_THRESHOLD
from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2
from src.models.graphsage_model import TravelTimeGraphSAGE
from src.utils.metrics import compute_all_metrics, travel_time_stats
from src.utils.others import get_logger, set_seed

logger = get_logger("trainer")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_GRAPH_DIR = PROJECT_ROOT / "src" / "data" / "processed" / "graph"

TARGET_CANDIDATES = ("travel_time_s", "target", "y")
_EPS = 1e-9  # evitar división por cero en ratios km


def _coerce_scalar(v):
    """Convierte valores leídos desde CSV a tipos útiles."""
    if pd.isna(v):
        return None

    if isinstance(v, (bool, np.bool_)):
        return bool(v)

    if isinstance(v, (int, np.integer)):
        return int(v)

    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None

        low = s.lower()
        if low in {"true", "false"}:
            return low == "true"
        if low in {"yes", "no"}:
            return low == "yes"

        num = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
        if pd.notna(num):
            return float(num)

        return s

    return v


def _as_bool(v) -> bool:
    """Normaliza campos tipo bool que pueden venir como 0/1, True/False, yes/no, strings, etc."""
    if pd.isna(v):
        return False

    if isinstance(v, (bool, np.bool_)):
        return bool(v)

    if isinstance(v, (int, np.integer, float, np.floating)):
        return float(v) != 0.0

    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _edge_role_from_row(row: pd.Series) -> str:
    """Clasifica la arista para distinguir espacial vs observada."""
    if _as_bool(row.get("observed")):
        return "observed"
    if _as_bool(row.get("spatial")):
        return "spatial"
    return "other"


def _load_multidigraph_from_csv(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    load_geometry: bool = True,
) -> nx.MultiDiGraph:
    """Reconstruye el MultiDiGraph respetando u, v, key y atributos por arista.

    Args:
        load_geometry: Si es False, omite la reconstrucción de geometrías Shapely
            (WKT). Usar False para entrenamiento donde las geometrías no se usan;
            usar True (default) para visualización y análisis espacial.
    """
    G = nx.MultiDiGraph()

    # Nodos
    for _, row in nodes_df.iterrows():
        node_id = str(row["GTFS Stop ID"])
        attrs = {}

        for col, val in row.items():
            if col in {"GTFS Stop ID", "geometry", "geometry_wkt"}:
                continue
            attrs[col] = _coerce_scalar(val)

        if load_geometry and "geometry_wkt" in row and pd.notna(row["geometry_wkt"]) and str(row["geometry_wkt"]).strip():
            attrs["geometry"] = wkt.loads(str(row["geometry_wkt"]))

        G.add_node(node_id, **attrs)

    # Aristas multigrafo
    for _, row in edges_df.iterrows():
        u = str(row["u"])
        v = str(row["v"])

        key = None
        if "key" in row.index and pd.notna(row["key"]):
            key = row["key"]

        attrs = {}
        for col, val in row.items():
            if col in {"u", "v", "key"}:
                continue
            if col == "geometry_wkt":
                if load_geometry and pd.notna(val) and str(val).strip():
                    attrs["geometry"] = wkt.loads(str(val))
            else:
                attrs[col] = _coerce_scalar(val)

        attrs["edge_role"] = _edge_role_from_row(row)
        G.add_edge(u, v, key=key, **attrs)

    return G


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_csv(path, low_memory=False)


def _normalize_column_name(col: str) -> str:
    return str(col).strip().lower()


def _choose_target_column(edges_df: pd.DataFrame) -> str:
    normalized = {_normalize_column_name(c): c for c in edges_df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "No se encontró una columna objetivo en edges.csv. "
        "Se espera algo como 'travel_time_s'."
    )


def _build_feature_frame(
    df: pd.DataFrame,
    exclude_cols: Optional[set[str]] = None,
    max_cardinality: int = 20,
) -> pd.DataFrame:
    """Construye un marco de features mixtas: numéricas + categóricas de baja cardinalidad."""
    exclude_cols = {c.lower() for c in (exclude_cols or set())}
    parts = []

    for col in df.columns:
        col_norm = _normalize_column_name(col)
        if col_norm in exclude_cols:
            continue

        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            parts.append(pd.DataFrame({col: s.astype(float)}))
            continue

        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.DataFrame({col: pd.to_numeric(s, errors="coerce")}))
            continue

        if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            s_str = s.fillna("missing").astype(str)
            nunique = s_str.nunique(dropna=True)
            if 1 < nunique <= max_cardinality:
                dummies = pd.get_dummies(s_str, prefix=col, dtype=float)
                parts.append(dummies)
            continue

    if parts:
        feat_df = pd.concat(parts, axis=1)
    else:
        feat_df = pd.DataFrame(index=df.index)

    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if feat_df.shape[1] == 0:
        feat_df = pd.DataFrame({"bias": np.ones(len(df), dtype=float)}, index=df.index)

    return feat_df.astype(np.float32)


def _build_masks(
    num_items: int,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Genera máscaras booleanas train/val/test con al menos 1 elemento de train si es posible."""
    if num_items <= 0:
        raise ValueError("No hay elementos para crear masks.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)

    n_test = int(round(num_items * test_ratio))
    n_val = int(round(num_items * val_ratio))

    if num_items >= 3:
        n_test = max(1, min(n_test, num_items - 2))
    else:
        n_test = max(0, min(n_test, num_items - 1))

    remaining_after_test = num_items - n_test
    if remaining_after_test >= 3:
        n_val = max(1, min(n_val, remaining_after_test - 1))
    else:
        n_val = max(0, min(n_val, remaining_after_test - 1))

    n_train = num_items - n_test - n_val
    if n_train < 1 and num_items >= 2:
        deficit = 1 - n_train
        take_val = min(deficit, max(0, n_val))
        n_val -= take_val
        deficit -= take_val
        if deficit > 0:
            n_test = max(0, n_test - deficit)
        n_train = num_items - n_test - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_items, dtype=torch.bool)
    val_mask = torch.zeros(num_items, dtype=torch.bool)
    test_mask = torch.zeros(num_items, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def _load_processed_graph_as_pyg(
    processed_dir: Path,
    config: Config,
) -> Tuple[Data, Dict[str, object], pd.DataFrame]:
    """
    Carga nodes.csv y edges.csv, reconstruye el MultiDiGraph y lo convierte
    a torch_geometric.data.Data sin colapsar aristas paralelas.

    Returns:
        (data, metadata, edge_graph_df)
        - data      : objeto Data de PyG listo para entrenamiento.
        - metadata  : dict ligero y JSON-serializable con estadísticas del grafo.
        - edge_graph_df : DataFrame con una fila por arista; incluye columna 'key'
          para identificar aristas paralelas (u, v, key) de forma unívoca.
    """
    nodes_path = processed_dir / "nodes.csv"
    edges_path = processed_dir / "edges.csv"
    lines_path = processed_dir / "lines.csv"

    nodes_df = _safe_read_csv(nodes_path)
    edges_df = _safe_read_csv(edges_path)
    lines_df = _safe_read_csv(lines_path) if lines_path.exists() else pd.DataFrame()

    if "GTFS Stop ID" not in nodes_df.columns:
        raise ValueError(
            "nodes.csv no contiene la columna 'GTFS Stop ID'. "
            "No se puede mapear el grafo a índices enteros."
        )

    if "u" not in edges_df.columns or "v" not in edges_df.columns:
        raise ValueError("edges.csv debe contener las columnas 'u' y 'v'.")

    # Reconstrucción fiel del MultiDiGraph.
    # NetworkX se usa aquí para preservar las keys de aristas paralelas (u,v,key)
    # y el orden original de iteración. La geometría no se carga para ahorrar
    # memoria y tiempo durante el entrenamiento.
    G = _load_multidigraph_from_csv(nodes_df, edges_df, load_geometry=False)

    node_ids = nodes_df["GTFS Stop ID"].astype(str).tolist()
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    # Features de nodo
    node_feat_exclude = {"gtfs stop id", "geometry", "geometry_wkt"}
    node_feature_df = _build_feature_frame(nodes_df, exclude_cols=node_feat_exclude)

    if node_feature_df.shape[1] == 1 and "bias" in node_feature_df.columns:
        preferred_node_cols = []
        for candidate in ["GTFS Latitude", "GTFS Longitude", "_x", "_y"]:
            if candidate in nodes_df.columns:
                preferred_node_cols.append(candidate)
        if preferred_node_cols:
            node_feature_df = (
                nodes_df[preferred_node_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )

    # Recolectar aristas del multigrafo sin perder paralelas ni keys
    edge_rows = []
    for u, v, k, edata in G.edges(keys=True, data=True):
        rec = {"u": u, "v": v, "key": k}
        rec.update(edata)
        edge_rows.append(rec)

    edge_graph_df = pd.DataFrame(edge_rows)

    target_col = _choose_target_column(edge_graph_df)

    # Features de arista: excluir identificadores, target y columnas de roles obsoletos
    edge_feat_exclude = {
        "u",
        "v",
        "key",
        "geometry",
        "geometry_wkt",
        "spatial",
        "observed",
        "is_spatial",
        "is_observed",
        "edge_role",
        "route_ids",
        target_col,
    }
    edge_feature_df = _build_feature_frame(edge_graph_df, exclude_cols=edge_feat_exclude)

    # Filtrar aristas que sí conectan nodos existentes
    valid_edge_rows = []
    for i, row in edge_graph_df.iterrows():
        u = str(row["u"])
        v = str(row["v"])
        if u in node_to_idx and v in node_to_idx:
            valid_edge_rows.append(i)

    if not valid_edge_rows:
        raise ValueError(
            "No se encontró ninguna arista válida que conecte nodos presentes en nodes.csv."
        )

    edge_graph_df = edge_graph_df.loc[valid_edge_rows].reset_index(drop=True)
    edge_feature_df = edge_feature_df.loc[valid_edge_rows].reset_index(drop=True)

    # Target solo para aristas observadas; las espaciales quedan sin supervisión
    y_raw = pd.to_numeric(edge_graph_df[target_col], errors="coerce")
    labeled_mask = ~y_raw.isna()
    labeled_idx = np.where(labeled_mask.to_numpy())[0]

    if labeled_idx.size == 0:
        raise ValueError(
            f"La columna objetivo '{target_col}' no tiene valores numéricos válidos en edges.csv."
        )

    if edge_feature_df.shape[1] == 0:
        edge_feature_df = pd.DataFrame(
            {"bias": np.ones(len(edge_graph_df), dtype=float)},
            index=edge_graph_df.index,
        )

    src = [node_to_idx[str(u)] for u in edge_graph_df["u"].astype(str)]
    dst = [node_to_idx[str(v)] for v in edge_graph_df["v"].astype(str)]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.tensor(node_feature_df.to_numpy(dtype=np.float32), dtype=torch.float)
    edge_attr = torch.tensor(edge_feature_df.to_numpy(dtype=np.float32), dtype=torch.float)

    # Aclaración sobre target: NaN para aristas sin observación (no cero, que es un valor físico válido).
    y_norm = np.full(len(y_raw), np.nan, dtype=np.float32)
    scaler = StandardScaler()
    y_norm[labeled_mask] = scaler.fit_transform(
        y_raw[labeled_mask].to_numpy().reshape(-1, 1)
    ).ravel()
    y = torch.tensor(
        np.nan_to_num(y_norm),
        dtype=torch.float32,
    )

    train_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)
    val_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)
    test_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)

    labeled_train, labeled_val, labeled_test = _build_masks(
        num_items=len(labeled_idx),
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    train_mask[labeled_idx[labeled_train.numpy()]] = True
    val_mask[labeled_idx[labeled_val.numpy()]] = True
    test_mask[labeled_idx[labeled_test.numpy()]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        target_scaler=scaler,
    )

    metadata: Dict[str, object] = {
        "processed_dir": str(processed_dir),
        "num_nodes": int(x.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "num_labeled_edges": int(labeled_idx.size),
        "num_train_edges": int(train_mask.sum().item()),
        "num_val_edges": int(val_mask.sum().item()),
        "num_test_edges": int(test_mask.sum().item()),
        "node_feature_dim": int(x.shape[1]),
        "edge_attr_dim": int(edge_attr.shape[1]),
        "target_col": target_col,
        "target_scaler": scaler,
        "lines_rows": int(len(lines_df)) if isinstance(lines_df, pd.DataFrame) else 0,
    }

    # Detecta patrones repetidos útiles para debug
    duplicate_uv = int(edge_graph_df.duplicated(subset=["u", "v"], keep=False).sum())
    metadata["duplicate_uv_edges"] = duplicate_uv

    return data, metadata, edge_graph_df

def predict_routes(
    model: nn.Module,
    data: Data,
    edge_graph_df: pd.DataFrame,
    route_candidates_csv: Path,
    run_dir: Path,
    device: torch.device,
    straightness_threshold: float = STRAIGHT_THRESHOLD,
) -> Dict[str, object]:
    """Predice tiempos de viaje para rutas multi-hop desde route_candidates.csv.

    Flujo por ruta en route_candidates.csv:
        1. Si la ruta es 1-hop (columnas start_stop_id/end_stop_id definen
           directamente una arista), se busca su posición en edge_graph_df.
        2. Si la ruta es multi-hop (columna 'hops' con lista de stop IDs
           intermedios), se descompone en hops elementales y se suma.
        3. Si algún hop no está en el grafo, la ruta se marca como no-cobertura
           y queda con pred_time=NaN.

    Después de predecir todas las rutas, aplica los filtros de escenarios A/B/C
    (km_tolerance + curve_penalty sobre straightness_index y km_offset) y guarda:
        run_dir/route_predictions/all_routes.csv
        run_dir/route_predictions/scenario_A.csv
        run_dir/route_predictions/scenario_B.csv
        run_dir/route_predictions/scenario_C.csv
        run_dir/route_predictions/summary.json

    Returns:
        Dict con conteos y rutas de archivos generados.
    """
    import json as _json

    out_dir = run_dir / "route_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Cargar route_candidates y obtener todas las predicciones del modelo
    # ------------------------------------------------------------------ #
    try:
        cand_df = pd.read_csv(str(route_candidates_csv), low_memory=False)
    except Exception as exc:
        logger.warning("predict_routes: no se pudo leer %s — %s", route_candidates_csv, exc)
        return {"error": str(exc)}

    if cand_df.empty:
        logger.warning("predict_routes: route_candidates.csv está vacío")
        return {"n_routes": 0}

    # Obtener predicciones del modelo para TODAS las aristas del grafo
    model.eval()
    with torch.no_grad():
        all_preds_norm = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        ).squeeze(-1).detach().cpu()

    y_raw = data.y.to(device)
    # Desnormalizar si hay scaler disponible en data
    scaler = getattr(data, "target_scaler", None)
    if scaler is not None:
        all_preds_sec = scaler.inverse_transform(
            all_preds_norm.numpy().reshape(-1, 1)
        ).ravel()
        y_denorm = scaler.inverse_transform(
            y_raw.numpy().reshape(-1, 1)
        ).ravel()
    else:
        all_preds_sec = all_preds_norm.numpy()
        y_denorm = y_raw.numpy()

    # ------------------------------------------------------------------ #
    # 2. Construir lookup (u_str, v_str) → target_time y pred_time
    #    Para aristas paralelas se promedia (misma pareja u,v en distinto key)
    # ------------------------------------------------------------------ #
    uv_to_preds: Dict[str, list] = {}
    for pos_idx, row in edge_graph_df.iterrows():
        key = str(row["u"]) + "__" + str(row["v"])
        uv_to_preds.setdefault(key, []).append(float(all_preds_sec[pos_idx]))

    uv_to_targets: Dict[str, list] = {}
    for pos_idx, row in edge_graph_df.iterrows():
        t_val = float(y_denorm[pos_idx])
        if not np.isnan(t_val):
            key = str(row["u"]) + "__" + str(row["v"])
            uv_to_targets.setdefault(key, []).append(t_val)

    # Para cada par (u,v), usar la predicción media entre aristas paralelas
    uv_pred: Dict[str, float] = {
        uv: float(np.mean(vals)) for uv, vals in uv_to_preds.items()
    }
    uv_target: Dict[str, float] = {
        uv: float(np.mean(vals)) for uv, vals in uv_to_targets.items()
    }


    # ------------------------------------------------------------------ #
    # 3. Predecir cada ruta en route_candidates.csv
    # ------------------------------------------------------------------ #
    # Determinar si la ruta viene en formato 1-hop (start_stop_id/end_stop_id)
    # o multi-hop (columna 'hops' con lista JSON de stop IDs consecutivos).
    has_hops_col = "hops" in cand_df.columns
    has_stop_ids = "start_stop_id" in cand_df.columns and "end_stop_id" in cand_df.columns

    result_rows = []
    n_covered = 0
    n_missing_hop = 0

    for _, cand_row in cand_df.iterrows():
        # Construir lista de hops elementales para esta ruta
        if has_hops_col and pd.notna(cand_row.get("hops", None)):
            try:
                hop_ids: list = _json.loads(str(cand_row["hops"]))
                # hop_ids es lista de stop IDs: [A, B, C, D]
                # → hops elementales: [(A,B), (B,C), (C,D)]
                hop_pairs = [(str(hop_ids[k]), str(hop_ids[k + 1])) for k in range(len(hop_ids) - 1)]
            except Exception:
                hop_pairs = []
        elif has_stop_ids:
            u_str = str(cand_row["start_stop_id"])
            v_str = str(cand_row["end_stop_id"])
            hop_pairs = [(u_str, v_str)]
        else:
            hop_pairs = []

        if not hop_pairs:
            pred_time = float("nan")
            covered = False
        else:
            hop_times = []
            hop_targets = []
            all_found = True
            for u_str, v_str in hop_pairs:
                uv = str(u_str) + "__" + str(v_str)
                vu = str(v_str) + "__" + str(u_str)
                t_pred = uv_pred.get(uv) or uv_pred.get(vu)
                if t_pred is None:
                    all_found = False
                    break
                hop_times.append(t_pred)
                # target puede no existir (arista espacial sin observación) → None
                t_tgt = uv_target.get(uv) or uv_target.get(vu)
                hop_targets.append(t_tgt)

            if all_found:
                pred_time = float(np.sum(hop_times))
                # target solo si todos los hops tienen ground truth
                target_time = (
                    float(np.sum(hop_targets))
                    if all(t is not None for t in hop_targets)
                    else float("nan")
                )
                covered = True
            else:
                pred_time = float("nan")
                target_time = float("nan")
                covered = False
                n_missing_hop += 1

        row_out = dict(cand_row)
        row_out["pred"] = pred_time
        row_out["target"] = target_time
        row_out["has_target"] = not np.isnan(target_time) and not target_time == 0
        row_out["n_hops"] = len(hop_pairs)
        row_out["covered"] = covered
        result_rows.append(row_out)

    all_routes_df = pd.DataFrame(result_rows)
    all_routes_df.to_csv(out_dir / "all_routes.csv", index=False)

    logger.info(
        "predict_routes: %d rutas totales | %d cubiertas por el grafo | %d con hops faltantes",
        len(all_routes_df), n_covered, n_missing_hop,
    )

    # ------------------------------------------------------------------ #
    # 4. Aplicar escenarios A/B/C por km_tolerance + curve_penalty
    # ------------------------------------------------------------------ #
    scenario_results: Dict[str, object] = {}
    bin_results: Dict[str, object] = {}

    required_for_scenarios = {"straightness_index", "tol_prox", "km_offset"}
    can_run_scenarios = required_for_scenarios.issubset(set(all_routes_df.columns))

    covered_df = all_routes_df[all_routes_df["covered"]].copy()


    for km_bin, bin_group in covered_df.groupby("tol_prox"):
        bin_times = bin_group["pred"].dropna()
        bin_results[str(km_bin)] = {
            "n_routes": len(bin_group),
            "mean_pred_sec": float(bin_times.mean()) if len(bin_times) else None,
            "median_pred_sec": float(bin_times.median()) if len(bin_times) else None,
            "std_pred_sec": float(bin_times.std()) if len(bin_times) else None,
        }
        bin_targets = bin_group["target"].dropna()
        # errors
        bin_metrics = compute_all_metrics(
            bin_times.values,
            bin_targets.values,
        )
        bin_results[str(km_bin)].update(
            {k: float(v) for k, v in bin_metrics.items()}
        )


    for letter, params in EVAL_SCENARIOS.items():
        km_tol    = float(params["km_tolerance"])
        curve_pen = float(params["curve_penalty"])

        if not can_run_scenarios or covered_df.empty:
            logger.warning("predict_routes: no hay columnas para escenario %s o no hay rutas cubiertas", letter)
            scenario_results[letter] = {"n_routes": 0, "skipped": True}
            continue

        _eps = 1e-9
        scenario_score = (
            covered_df["straightness_index"]
            - curve_pen * covered_df["km_offset"].abs() / (covered_df["tol_prox"] + _eps)
        )
        score_threshold = straightness_threshold - curve_pen * km_tol
        km_filter    = covered_df["km_offset"].abs() / (covered_df["tol_prox"] + _eps) <= km_tol
        score_filter = scenario_score >= score_threshold

        scen_df = covered_df[km_filter & score_filter].copy()
        scen_df["scenario_score"] = scenario_score[km_filter & score_filter].values

        # Agrupar por bin de km para ver distribución de predicciones
        summary_by_bin: Dict[str, object] = {}
        if "tol_prox" in scen_df.columns and not scen_df.empty:
            for km_bin, bin_group in scen_df.groupby("tol_prox"):
                bin_times = bin_group["pred"].dropna()
                summary_by_bin[str(km_bin)] = {
                    "n_routes": len(bin_group),
                    "mean_pred_sec": float(bin_times.mean()) if len(bin_times) else None,
                    "median_pred_sec": float(bin_times.median()) if len(bin_times) else None,
                    "std_pred_sec": float(bin_times.std()) if len(bin_times) else None,
                }
                bin_targets = bin_group["target"].dropna()


        scenario_results[letter] = {
            "n_routes": len(scen_df),
            "km_tolerance": km_tol,
            "curve_penalty": curve_pen,
            "score_threshold": score_threshold,
            "by_km_bin": summary_by_bin,
        }
        eval_rows = scen_df[scen_df["has_target"]].copy()
        if not eval_rows.empty:
            scen_metrics = compute_all_metrics(
                eval_rows["pred"].values,
                eval_rows["target"].values,
            )
            scenario_results[letter].update({k: float(v) for k, v in scen_metrics.items()})
            scenario_results[letter]["n_with_target"] = len(eval_rows)

        logger.info(
            "predict_routes escenario [%s]: %d rutas | km_tol=%.2f | score_thr=%.3f",
            letter, len(scen_df), km_tol, score_threshold,
        )

    # ------------------------------------------------------------------ #
    # 5. Guardar resumen global
    # ------------------------------------------------------------------ #
    summary = {
        "n_routes_total": len(all_routes_df),
        "n_routes_covered": n_covered,
        "n_routes_missing_hop": n_missing_hop,
        "scenarios": scenario_results,
        "bin_results": bin_results
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        _json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Evaluación de escenarios A/B/C derivados de route_candidates
# ---------------------------------------------------------------------------

def evaluate_scenarios_from_routes(
    run_dir: Path,
    straightness_threshold: float = 0.9,
) -> Dict[str, Optional[Dict]]:
    """Evalúa escenarios A/B/C filtrando desde eval_candidates/predictions.csv.

    No depende de CSVs externos: aplica los filtros de cada escenario sobre
    las predicciones ya calculadas del subconjunto maestro (eval_candidates).

    Para cada escenario (km_tolerance, curve_penalty) calcula:
        scenario_score   = straightness_index − curve_penalty · |km_offset| / (tol_prox + ε)
        score_threshold  = straightness_threshold − curve_penalty · km_tolerance
        km_filter        = abs(km_offset) / (tol_prox + ε) ≤ km_tolerance

    Guarda en run_dir:
        eval_scenario_A/metrics.json + predictions.csv
        eval_scenario_B/...
        eval_scenario_C/...

    Returns:
        Dict {letra: metrics_dict_o_None}
    """
    import json as _json

    master_preds_path = run_dir / "route_predictions" / "all_routes.csv"
    results: Dict[str, Optional[Dict]] = {}

    if not master_preds_path.exists():
        logger.warning(
            "evaluate_scenarios_from_routes: route_predictions/all_routes.csv no encontrado en %s — "
            "asegúrate de incluir 'candidates' en los subsets de evaluación",
            run_dir,
        )
        return results

    try:
        df = pd.read_csv(str(master_preds_path), low_memory=False)
    except Exception as exc:
        logger.warning("evaluate_scenarios_from_routes: error leyendo predictions.csv — %s", exc)
        return results

    required_cols = {"pred", "target", "tol_prox", "km_offset", "straightness_index"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(
            "evaluate_scenarios_from_routes: columnas faltantes en eval_candidates/predictions.csv — %s. "
            "El CSV maestro debe incluir tol_prox, km_offset y straightness_index.",
            missing,
        )
        return results

    df = df.dropna(subset=["pred", "target"])

    for letter, params in EVAL_SCENARIOS.items():
        km_tol   = float(params["km_tolerance"])
        curve_pen = float(params["curve_penalty"])
        out_dir   = run_dir / f"eval_scenario_{letter}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Recomputar score con los parámetros específicos del escenario
        scenario_score  = df["straightness_index"] - curve_pen * df["km_offset"].abs() / (df["tol_prox"] + _EPS)
        score_threshold = straightness_threshold - curve_pen * km_tol
        km_filter       = df["km_offset"].abs() / (df["tol_prox"] + _EPS) <= km_tol
        score_filter    = scenario_score >= score_threshold

        sub = df[km_filter & score_filter].copy()
        sub["scenario_score"] = scenario_score[km_filter & score_filter].values

        skipped = sub.empty
        metrics_out: Dict[str, object] = {
            "subset_name":     f"scenario_{letter}",
            "km_tolerance":    km_tol,
            "curve_penalty":   curve_pen,
            "score_threshold": score_threshold,
            "n_eval":          len(sub),
            "skipped":         skipped,
        }

        if not skipped:
            metrics = compute_all_metrics(sub["pred"].values, sub["target"].values)
            metrics_out.update({k: float(v) for k, v in metrics.items()})
            sub.to_csv(out_dir / "predictions.csv", index=False)
            logger.info(
                "Scenario eval [%s] | n_eval: %d | RMSE: %.4f | R²: %.4f",
                letter, len(sub),
                metrics.get("rmse", float("nan")),
                metrics.get("r2",   float("nan")),
            )
        else:
            logger.warning("Scenario eval [%s]: ninguna arista pasó los filtros", letter)

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
            _json.dump(metrics_out, fh, indent=2, ensure_ascii=False)

        results[letter] = metrics_out

    return results


# ---------------------------------------------------------------------------
# Plots de entrenamiento
# ---------------------------------------------------------------------------

def _save_plot(fig, path_run: Path, path_reports: Path) -> None:
    """Guarda una figura en run_dir y en src/outputs/reports/plots/."""
    for dest in (path_run, path_reports):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(dest), dpi=120, bbox_inches="tight")
        except Exception as exc:
            logger.warning("No se pudo guardar plot en %s: %s", dest, exc)


def plot_training_results(
    run_dir: Path,
    history: Dict[str, list],
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    model_name: str = "",
) -> None:

    reports_plot_dir = PROJECT_ROOT / "src" / "outputs" / "reports" / "plots"
    ts = run_dir.name  # timestamp del run, e.g. 20260625_143022

    def _report_path(name: str) -> Path:
        return reports_plot_dir / f"{model_name}_{ts}_{name}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- 1. Curvas de entrenamiento ---
    if history.get("epoch"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Curvas de entrenamiento — {model_name.upper()}", fontsize=13)

        axes[0].plot(history["epoch"], history["train_loss"], color="steelblue")
        axes[0].set_xlabel("Época"); axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Train Loss"); axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 20)

        axes[1].plot(history["epoch"], history["val_rmse"], color="darkorange")
        axes[1].set_xlabel("Época"); axes[1].set_ylabel("RMSE (s)")
        axes[1].set_title("Val RMSE"); axes[1].grid(alpha=0.3)
        axes[1].set_ylim(50, 300)

        axes[2].plot(history["epoch"], history["val_r2"], color="seagreen")
        axes[2].set_xlabel("Época"); axes[2].set_ylabel("R²")
        axes[2].set_title("Val R²"); axes[2].grid(alpha=0.3)
        axes[2].set_ylim(-1, 1)

        plt.tight_layout()
        _save_plot(fig, run_dir / "training_curves.png", _report_path("training_curves.png"))
        plt.close(fig)

    # --- 2. Scatter pred vs target ---
    if len(test_preds) > 0:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(test_targets, test_preds, alpha=0.4, s=12, color="steelblue")
        lo = min(float(test_targets.min()), float(test_preds.min())) * 0.95
        hi = max(float(test_targets.max()), float(test_preds.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Ideal")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_xlabel("Target (s)"); ax.set_ylabel("Predicción (s)")
        ax.set_title(f"Predicciones vs Targets — {model_name.upper()}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_plot(fig, run_dir / "predictions_scatter.png", _report_path("predictions_scatter.png"))
        plt.close(fig)

    # Distribución de errores (points)
    if len(test_preds) > 0:
        errors = test_preds - test_targets
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(errors, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Error = 0")
        ax.set_xlabel("Error (pred − target) [s]"); ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de errores — {model_name.upper()}")
        ax.set_xlim(-400, 400)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        _save_plot(fig, run_dir / "error_distribution.png", _report_path("error_distribution.png"))
        plt.close(fig)

    # --- 4. RMSE por bin tol_prox (all routes) ---
    master_preds_path = run_dir / "route_predictions" / "all_routes.csv"
    if master_preds_path.exists():
        try:
            df_m = pd.read_csv(str(master_preds_path), low_memory=False)
            if {"pred", "target", "tol_prox"}.issubset(df_m.columns):
                df_m = df_m.dropna(subset=["pred", "target"])
                bins_rmse = (
                    df_m.groupby("tol_prox")
                    .apply(lambda g: float(np.sqrt(((g["pred"].values - g["target"].values) ** 2).mean())))
                    .reset_index(name="rmse")
                )
                if not bins_rmse.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(bins_rmse["tol_prox"].astype(str), bins_rmse["rmse"],
                           color="steelblue", alpha=0.85, edgecolor="white")
                    ax.set_xlabel("tol_prox (km)"); ax.set_ylabel("RMSE (s)")
                    ax.set_title(f"RMSE por bin de km — {model_name.upper()}")
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    _save_plot(fig, run_dir / "rmse_by_km_bin.png", _report_path("rmse_by_km_bin.png"))
                    plt.close(fig)
        except Exception as exc:
            logger.warning("No se pudo generar rmse_by_km_bin.png: %s", exc)


def build_model(config: Config, in_channels: int, edge_attr_dim: int) -> nn.Module:
    """Crea el encoder/decoder elegido."""
    if config.hidden_dim < 2:
        raise ValueError("hidden_dim debe ser >= 2 para que el decoder sea válido.")
    if config.num_layers < 1:
        raise ValueError("num_layers debe ser >= 1.")

    if config.model in ("gat", "gatv2") and config.num_layers < 2:
        raise ValueError(
            f"{config.model} requiere num_layers >= 2 para que la dimensión final del encoder sea estable."
        )

    if config.model == "graphsage":
        return TravelTimeGraphSAGE(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )

    if config.model == "gat":
        return TravelTimeGAT(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )

    if config.model == "gatv2":
        return TravelTimeGATv2(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )

    raise ValueError(f"Modelo desconocido: {config.model}")


def _masked_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=preds.device)
    return criterion(preds[mask], targets[mask])


def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Ejecuta una época de entrenamiento usando solo train_mask."""
    model.train()
    optimizer.zero_grad()
    preds = model(data.x, data.edge_index, data.edge_attr)
    loss = _masked_loss(preds, data.y, data.train_mask, criterion)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: Data,
    mask_name: str = "val",
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Evalúa el modelo sobre una máscara específica: train, val o test."""
    model.eval()
    preds = model(data.x, data.edge_index, data.edge_attr)

    if not hasattr(data, f"{mask_name}_mask"):
        raise ValueError(f"La data no contiene la máscara '{mask_name}_mask'.")

    mask = getattr(data, f"{mask_name}_mask")
    pred_np = preds[mask].detach().cpu().numpy()
    target_np = data.y[mask].detach().cpu().numpy()
    
    scaler = getattr(data, "target_scaler", None)
    if scaler is not None:
        pred_np = scaler.inverse_transform(
            pred_np.reshape(-1, 1)
        ).ravel()
        target_np = scaler.inverse_transform(
            target_np.reshape(-1, 1)
        ).ravel()
    metrics = compute_all_metrics(pred_np, target_np)
    return metrics, preds


def fit_model(model: nn.Module, data: Data, config: Config, run_dir: Optional[Path] = None) -> Dict[str, object]:
    """Entrena el modelo y conserva el mejor checkpoint según validación.

    Args:
        run_dir: Si se indica, guarda `best_model.pt` en esa carpeta cada vez
            que se encuentra un nuevo mejor modelo.
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
    }
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, data, optimizer, criterion)
        if epoch % config.eval_every == 0:
            val_metrics, _ = evaluate_model(
                model, data,
                mask_name="val"
            )
            val_mse = val_metrics["mse"]
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_mse)
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_r2"].append(val_metrics["r2"])

            logger.info(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                # f"Val MSE: {val_metrics['mse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val R²: {val_metrics['r2']:.4f}"
            )
            if val_mse < best_val:
                best_val = val_mse
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
                logger.info(
                    f"   ✓ Nuevo mejor modelo "
                    f"(val_mse={best_val:.4f})"
                )
                if run_dir is not None:
                    try:
                        import torch as _torch
                        _torch.save(best_state, run_dir / "best_model.pt")
                    except Exception as _e:
                        logger.warning(f"No se pudo guardar best_model.pt: {_e}")
            else:
                patience_counter += 1
                logger.info(
                    f"   patience "
                    f"{patience_counter}/{config.patience}"
                )
                if patience_counter >= config.patience:
                    logger.info(
                        "\nEarly stopping activado."
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history

def train_and_evaluate(
    config: Config,
    processed_dir: Optional[Path] = None,
    evaluate: bool = False,
) -> Dict[str, object]:
    """Pipeline completo: carga datos, entrena, evalúa en test y guarda artefactos.

    Crea una carpeta de corrida en:
        src/outputs/training/<model>/<YYYYMMDD_HHMMSS>/
    y guarda dentro:
        training.log, history.csv, final_metrics.json,
        test_predictions.csv, best_model.pt
        training_curves.png, predictions_scatter.png,
        error_distribution.png, rmse_by_km_bin.png

    Evaluación de subconjuntos (en orden):
    1. eval_master/      — subconjunto maestro de master_segments.csv
    2. eval_scenario_A/  — derivado de eval_master filtrando por EVAL_SCENARIOS["A"]
       eval_scenario_B/  — ídem para B
       eval_scenario_C/  — ídem para C
       Los escenarios A/B/C no requieren CSVs externos: se calculan sobre
       las predicciones de eval_master usando km_offset y curve_penalty_score.

    Args:
        master_csv:   Ruta a master_segments.csv. Si es None, se busca en la
                      ruta por defecto src/topsegments/master_segments.csv.
        straight_csv: Ruta a straight_routes.csv. Si es None, se busca en la
                      ruta por defecto src/outputs/routes/straight_routes.csv.
    """
    import datetime

    set_seed(config.seed)
    device = torch.device(config.device)

    # Crear carpeta de corrida
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        PROJECT_ROOT / "src" / "outputs" / "training" / config.model / run_timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Activar log a archivo para esta corrida
    run_logger = get_logger("trainer", log_file=str(run_dir / "training.log"))

    processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_GRAPH_DIR
    run_logger.info(f"Cargando grafo procesado desde: {processed_dir}")

    data, metadata, edge_graph_df = _load_processed_graph_as_pyg(processed_dir, config)
    nodes_path = processed_dir / "nodes.csv"
    data = data.to(device)

    run_logger.info(
        f"Grafo cargado | nodos: {metadata['num_nodes']} | aristas: {metadata['num_edges']} | "
        f"etiquetadas: {metadata['num_labeled_edges']} | train/val/test: "
        f"{metadata['num_train_edges']}/{metadata['num_val_edges']}/{metadata['num_test_edges']}"
    )
    run_logger.info(
        f"Features | node_dim: {metadata['node_feature_dim']} | edge_attr_dim: {metadata['edge_attr_dim']} | "
        f"target: {metadata['target_col']}"
    )

    model = build_model(config, data.x.shape[1], data.edge_attr.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_logger.info(f"Modelo: {config.model} | Parámetros entrenables: {num_params:,}")

    run_logger.info(f"Entrenando durante {config.epochs} épocas...")
    start_time = time.time()
    train_info = fit_model(model, data, config, run_dir=run_dir)
    elapsed = time.time() - start_time
    run_logger.info(f"Entrenamiento completado en {elapsed:.1f}s")

    final_metrics, preds = evaluate_model(model, data, mask_name="test")
    test_preds = preds[data.test_mask].detach().cpu().numpy()
    test_targets = data.y[data.test_mask].detach().cpu().numpy()

    scaler = getattr(data, "target_scaler", None)
    if scaler is not None:
        test_preds = scaler.inverse_transform(
            test_preds.reshape(-1,1)
        ).ravel()
        test_targets = scaler.inverse_transform(
            test_targets.reshape(-1,1)
        ).ravel()
    stats = travel_time_stats(test_preds, test_targets)

    run_logger.info("=" * 60)
    run_logger.info(f"RESULTADOS FINALES — Modelo: {config.model.upper()}")
    run_logger.info("=" * 60)
    run_logger.info(f"MSE:  {final_metrics['mse']:.4f}")
    run_logger.info(f"RMSE: {final_metrics['rmse']:.4f}")
    run_logger.info(f"MAE:  {final_metrics['mae']:.4f}")
    run_logger.info(f"MAPE: {final_metrics['mape']:.2f}%")
    run_logger.info(f"R²:   {final_metrics['r2']:.4f}")
    run_logger.info("-" * 60)
    run_logger.info("Estadísticas de predicciones (test):")
    run_logger.info(
        f"Media pred: {stats.get('pred_mean', 0.0):.2f} | Media target: {stats.get('target_mean', 0.0):.2f}"
    )
    run_logger.info(
        f"Mediana pred: {stats.get('pred_median', 0.0):.2f} | Mediana target: {stats.get('target_median', 0.0):.2f}"
    )
    run_logger.info(
        f"Std pred: {stats.get('pred_std', 0.0):.2f} | Std target: {stats.get('target_std', 0.0):.2f}"
    )
    run_logger.info(f"P25 error: {stats.get('error_p25', 0.0):.2f} | P75 error: {stats.get('error_p75', 0.0):.2f}")
    run_logger.info(f"P90 error: {stats.get('error_p90', 0.0):.2f}")
    run_logger.info("=" * 60)

    # --- Guardar artefactos de la corrida ---
    try:
        # history.csv
        history_df = pd.DataFrame(train_info)
        history_df.to_csv(run_dir / "history.csv", index=False)

        # final_metrics.json
        artifacts_metrics = {
            **final_metrics,
            "model": config.model,
            "elapsed_seconds": elapsed,
            "num_params": num_params,
            "num_train_edges": int(metadata["num_train_edges"]),
            "num_val_edges": int(metadata["num_val_edges"]),
            "num_test_edges": int(metadata["num_test_edges"]),
            "stats": stats,
        }
        import json as _json
        with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as fh:
            _json.dump(artifacts_metrics, fh, indent=2, ensure_ascii=False, default=str)

        # test_predictions.csv
        test_edge_indices = data.test_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
        pd.DataFrame({
            "edge_idx": test_edge_indices,
            "pred": test_preds,
            "target": test_targets,
        }).to_csv(run_dir / "test_predictions.csv", index=False)

        run_logger.info(f"Artefactos guardados en {run_dir}")
    except Exception as _e:
        run_logger.warning(f"No se pudieron guardar algunos artefactos: {_e}")

    # --- Evaluación sobre route_candidates ---
    if evaluate:
        # --- Predicción sobre rutas multi-hop de route_candidates.csv ---
        _default_candidates = PROJECT_ROOT / "src" / "outputs" / "routes" / "route_candidates.csv"
        if _default_candidates.exists():
            run_logger.info("Prediciendo tiempos de rutas multi-hop desde route_candidates.csv...")
            route_pred_summary = predict_routes(
                model=model,
                data=data,
                edge_graph_df=edge_graph_df,
                route_candidates_csv=_default_candidates,
                run_dir=run_dir,
                device=device,
                straightness_threshold=STRAIGHT_THRESHOLD,
            )
            run_logger.info(
                "predict_routes: %d rutas cubiertas de %d totales",
                route_pred_summary.get("n_routes_covered", 0),
                route_pred_summary.get("n_routes_total", 0),
            )
        else:
            run_logger.warning(
                "route_candidates.csv no encontrado en %s — predicción multi-hop omitida.",
                _default_candidates,
            )

    # --- Plots de entrenamiento ---
    run_logger.info("Generando plots de entrenamiento...")
    plot_training_results(
        run_dir=run_dir,
        history=train_info,
        test_preds=test_preds,
        test_targets=test_targets,
        model_name=config.model,
    )

    result = {
        "model": config.model,
        "run_dir": str(run_dir),
        "processed_dir": str(processed_dir),
        "metadata": metadata,
        "num_params": num_params,
        "elapsed_seconds": elapsed,
        "train_info": train_info,
        "final_metrics": final_metrics,
        "stats": stats
    }
    return result