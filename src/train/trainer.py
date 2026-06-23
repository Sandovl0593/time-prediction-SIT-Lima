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

from src.config import Config
from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2
from src.models.graphsage_model import TravelTimeGraphSAGE
from src.utils.metrics import compute_all_metrics, travel_time_stats
from src.utils.others import get_logger, set_seed


logger = get_logger("trainer")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_GRAPH_DIR = PROJECT_ROOT / "src" / "data" / "processed" / "graph"

TARGET_CANDIDATES = ("travel_time_s", "target", "y")


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


def _load_multidigraph_from_csv(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.MultiDiGraph:
    """Reconstruye el MultiDiGraph respetando u, v, key y atributos por arista."""
    G = nx.MultiDiGraph()

    # Nodos
    for _, row in nodes_df.iterrows():
        node_id = str(row["GTFS Stop ID"])
        attrs = {}

        for col, val in row.items():
            if col in {"GTFS Stop ID", "geometry", "geometry_wkt"}:
                continue
            attrs[col] = _coerce_scalar(val)

        if "geometry_wkt" in row and pd.notna(row["geometry_wkt"]) and str(row["geometry_wkt"]).strip():
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
                if pd.notna(val) and str(val).strip():
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
        "Se esperaba algo como 'travel_time_s'."
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
) -> Tuple[Data, Dict[str, object]]:
    """
    Carga nodes.csv y edges.csv, reconstruye el MultiDiGraph y lo convierte
    a torch_geometric.data.Data sin colapsar aristas paralelas.
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

    # Reconstrucción fiel del MultiDiGraph
    G = _load_multidigraph_from_csv(nodes_df, edges_df)

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

    # Flags explícitos para ayudar al modelo a distinguir tipos de arista
    if "spatial" in edge_graph_df.columns:
        edge_graph_df["is_spatial"] = edge_graph_df["spatial"].apply(_as_bool).astype(float)
    else:
        edge_graph_df["is_spatial"] = 0.0

    if "observed" in edge_graph_df.columns:
        edge_graph_df["is_observed"] = edge_graph_df["observed"].apply(_as_bool).astype(float)
    else:
        edge_graph_df["is_observed"] = 0.0

    edge_graph_df["edge_role"] = edge_graph_df.apply(_edge_role_from_row, axis=1)

    # Features de arista: excluir identificadores y target
    edge_feat_exclude = {
        "u",
        "v",
        "key",
        "geometry",
        "geometry_wkt",
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
    labeled_idx = np.where(~y_raw.isna().to_numpy())[0]

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
    y = torch.tensor(y_raw.fillna(0.0).to_numpy(dtype=np.float32), dtype=torch.float)

    train_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)
    val_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)
    test_mask = torch.zeros(len(edge_graph_df), dtype=torch.bool)

    labeled_train, labeled_val, labeled_test = _build_masks(
        num_items=len(labeled_idx),
        test_ratio=config.test_ratio,
        val_ratio=0.1,
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
        "lines_rows": int(len(lines_df)) if isinstance(lines_df, pd.DataFrame) else 0,
    }

    # Detecta patrones repetidos útiles para debug
    duplicate_uv = int(edge_graph_df.duplicated(subset=["u", "v"], keep=False).sum())
    metadata["duplicate_uv_edges"] = duplicate_uv

    return data, metadata


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
    metrics = compute_all_metrics(pred_np, target_np)
    return metrics, preds


def fit_model(model: nn.Module, data: Data, config: Config) -> Dict[str, object]:
    """Entrena el modelo y conserva el mejor checkpoint según validación."""
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_mse": []
    }
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, data, optimizer, criterion)
        history["train_loss"].append(train_loss)
        if epoch % config.eval_every == 0:
            val_metrics, _ = evaluate_model(
                model, data,
                mask_name="val"
            )
            val_mse = val_metrics["mse"]
            history["val_mse"].append(val_mse)

            logger.info(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val MSE: {val_metrics['mse']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val R²: {val_metrics['r2']:.4f}"
            )
            if val_mse < best_val:
                best_val = val_mse
                best_state = deepcopy( model.state_dict())
                patience_counter = 0
                logger.info(
                    f"   ✓ Nuevo mejor modelo "
                    f"(val_mse={best_val:.4f})"
                )
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
) -> Dict[str, object]:
    """Pipeline completo: carga datos, entrena y evalúa en test."""
    set_seed(config.seed)
    device = torch.device(config.device)

    processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_GRAPH_DIR
    logger.info(f"Cargando grafo procesado desde: {processed_dir}")

    data, metadata = _load_processed_graph_as_pyg(processed_dir, config)
    data = data.to(device)

    logger.info(
        f"Grafo cargado | nodos: {metadata['num_nodes']} | aristas: {metadata['num_edges']} | "
        f"etiquetadas: {metadata['num_labeled_edges']} | train/val/test: "
        f"{metadata['num_train_edges']}/{metadata['num_val_edges']}/{metadata['num_test_edges']}"
    )
    logger.info(
        f"Features | node_dim: {metadata['node_feature_dim']} | edge_attr_dim: {metadata['edge_attr_dim']} | "
        f"target: {metadata['target_col']}"
    )

    model = build_model(config, data.x.shape[1], data.edge_attr.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo: {config.model} | Parámetros entrenables: {num_params:,}")

    logger.info(f"Entrenando durante {config.epochs} épocas...")
    start_time = time.time()
    train_info = fit_model(model, data, config)
    elapsed = time.time() - start_time
    logger.info(f"Entrenamiento completado en {elapsed:.1f}s")

    final_metrics, preds = evaluate_model(model, data, mask_name="test")
    test_preds = preds[data.test_mask].detach().cpu().numpy()
    test_targets = data.y[data.test_mask].detach().cpu().numpy()
    stats = travel_time_stats(test_preds, test_targets)

    logger.info("=" * 60)
    logger.info(f"RESULTADOS FINALES — Modelo: {config.model.upper()}")
    logger.info("=" * 60)
    logger.info(f"MSE:  {final_metrics['mse']:.4f}")
    logger.info(f"RMSE: {final_metrics['rmse']:.4f}")
    logger.info(f"MAE:  {final_metrics['mae']:.4f}")
    logger.info(f"MAPE: {final_metrics['mape']:.2f}%")
    logger.info(f"R²:   {final_metrics['r2']:.4f}")
    logger.info("-" * 60)
    logger.info("Estadísticas de predicciones (test):")
    logger.info(
        f"Media pred: {stats.get('pred_mean', 0.0):.2f} | Media target: {stats.get('target_mean', 0.0):.2f}"
    )
    logger.info(
        f"Mediana pred: {stats.get('pred_median', 0.0):.2f} | Mediana target: {stats.get('target_median', 0.0):.2f}"
    )
    logger.info(
        f"Std pred: {stats.get('pred_std', 0.0):.2f} | Std target: {stats.get('target_std', 0.0):.2f}"
    )
    logger.info(f"P25 error: {stats.get('error_p25', 0.0):.2f} | P75 error: {stats.get('error_p75', 0.0):.2f}")
    logger.info(f"P90 error: {stats.get('error_p90', 0.0):.2f}")
    logger.info("=" * 60)

    return {
        "model": config.model,
        "processed_dir": str(processed_dir),
        "metadata": metadata,
        "num_params": num_params,
        "elapsed_seconds": elapsed,
        "train_info": train_info,
        "final_metrics": final_metrics,
        "stats": stats,
    }
