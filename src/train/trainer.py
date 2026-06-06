"""Trainer unificado para todos los modelos de predicción de tiempos de viaje.

Soporta 8 modelos:
- gcn: GCN puro (estático)
- graphsage: GraphSAGE puro (estático)
- gat: GAT con multi-head attention (estático)
- gatv2: GATv2 con atención dinámica (estático)
"""

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.config import Config
from src.models.graphsage_model import TravelTimeGraphSAGE
from src.models.gat_model import TravelTimeGAT
from src.models.gatv2_model import TravelTimeGATv2
from src.utils.others import get_logger, set_seed
from src.utils.metrics import compute_all_metrics, travel_time_stats


logger = get_logger("trainer")


def build_model(config: Config, in_channels: int, edge_attr_dim: int) -> nn.Module:
    """Construye el modelo según la configuración.

    Args:
        config: Configuración del experimento.
        in_channels: Dimensión de features de nodo.
        edge_attr_dim: Dimensión de features de arista.

    Returns:
        Modelo instanciado.
    """
    if config.model == "graphsage":
        return TravelTimeGraphSAGE(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )
    elif config.model == "gat":
        return TravelTimeGAT(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )
    elif config.model == "gatv2":
        return TravelTimeGATv2(
            in_channels=in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            heads=config.heads,
            edge_attr_dim=edge_attr_dim,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Modelo desconocido: {config.model}")


# def _is_hybrid(model_name: str) -> bool:
#     """Retorna True si el modelo usa features temporales (GRU)."""
#     return model_name in ("gcn_gru", "graphsage_gru", "gat_gru", "gatv2_gru")


def train_epoch(
    model: nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
) -> float:
    """Ejecuta una época de entrenamiento.

    Returns:
        Loss promedio de entrenamiento.
    """
    model.train()
    optimizer.zero_grad()

    # if _is_hybrid(config.model):
    #     preds = model(data.x_temporal, data.edge_index, data.edge_attr)
    # else:
    #     preds = model(data.x, data.edge_index, data.edge_attr)

    preds = model(data.x, data.edge_index, data.edge_attr)

    loss = criterion(preds[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data,
    config: Config,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Evalúa el modelo en los datos de test.

    Returns:
        metrics: Dict con todas las métricas.
        preds: Tensor de predicciones sobre aristas de test.
    """
    model.eval()

    # if _is_hybrid(config.model):
    #     preds = model(data.x_temporal, data.edge_index, data.edge_attr)
    # else:
    #     preds = model(data.x, data.edge_index, data.edge_attr)

    preds = model(data.x, data.edge_index, data.edge_attr)

    test_preds = preds[data.test_mask].cpu().numpy()
    test_targets = data.y[data.test_mask].cpu().numpy()

    metrics = compute_all_metrics(test_preds, test_targets)
    return metrics, preds


def train_and_evaluate(config: Config) -> Dict:
    """Pipeline completo: genera datos, entrena y evalúa.

    Args:
        config: Configuración del experimento.

    Returns:
        Dict con métricas finales, estadísticas e historial de loss.
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    # ----- Datos -----
    logger.info("Generando grafo mock de Lima...")
    data, metadata = mock_lima_graph(
        seed=config.seed,
        num_time_steps=config.num_time_steps,
    )
    data = data.to(device)
    logger.info(
        f"Grafo: {metadata['num_nodes']} nodos, {metadata['num_edges']} aristas "
        f"(train: {metadata['train_edges']}, test: {metadata['test_edges']})"
    )

    # ----- Modelo -----
    in_channels = data.x.shape[1]
    edge_attr_dim = data.edge_attr.shape[1]
    model = build_model(config, in_channels, edge_attr_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo: {config.model} | Parámetros: {num_params:,}")

    # ----- Entrenamiento -----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    loss_history = []
    best_loss = float("inf")
    best_metrics = {}

    logger.info(f"Iniciando entrenamiento por {config.epochs} épocas...")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, data, optimizer, criterion, config)
        loss_history.append(train_loss)

        if epoch % max(1, config.epochs // 10) == 0 or epoch == 1:
            metrics, _ = evaluate_model(model, data, config)
            if metrics["mse"] < best_loss:
                best_loss = metrics["mse"]
                best_metrics = metrics.copy()
            logger.info(
                f"Epoch {epoch:4d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test MSE: {metrics['mse']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"R²: {metrics['r2']:.4f}"
            )

    elapsed = time.time() - start_time
    logger.info(f"Entrenamiento completado en {elapsed:.1f}s")

    # ----- Evaluación final -----
    final_metrics, preds = evaluate_model(model, data, config)
    test_preds = preds[data.test_mask].cpu().numpy()
    test_targets = data.y[data.test_mask].cpu().numpy()
    stats = travel_time_stats(test_preds, test_targets)

    logger.info("=" * 60)
    logger.info(f"RESULTADOS FINALES — Modelo: {config.model.upper()}")
    logger.info("=" * 60)
    logger.info(f"  MSE:  {final_metrics['mse']:.4f}")
    logger.info(f"  RMSE: {final_metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {final_metrics['mae']:.4f}")
    logger.info(f"  MAPE: {final_metrics['mape']:.2f}%")
    logger.info(f"  R²:   {final_metrics['r2']:.4f}")
    logger.info("-" * 60)
    logger.info("Estadísticas de predicciones (minutos):")
    logger.info(f"  Media pred:   {stats['pred_mean']:.2f} | Media target: {stats['target_mean']:.2f}")
    logger.info(f"  Mediana pred: {stats['pred_median']:.2f} | Mediana target: {stats['target_median']:.2f}")
    logger.info(f"  Std pred:     {stats['pred_std']:.2f} | Std target:   {stats['target_std']:.2f}")
    logger.info(f"  P25 error:    {stats['error_p25']:.2f} | P75 error:   {stats['error_p75']:.2f}")
    logger.info(f"  P90 error:    {stats['error_p90']:.2f}")
    logger.info("=" * 60)

    return {
        "model": config.model,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "stats": stats,
        "loss_history": loss_history,
        "metadata": metadata,
        "num_params": num_params,
        "elapsed_seconds": elapsed,
    }
