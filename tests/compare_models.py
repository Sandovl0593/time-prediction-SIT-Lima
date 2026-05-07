"""Tests comparativos: GCN+GRU vs GraphSAGE+GRU usando mock_lima_tensor.

Este test ejecuta un número pequeño de repeticiones (REPEATS) con pocas
épocas para recoger métricas y tiempos, luego resume (mean, std, median,
min, max) por métrica y determina el "mejor" modelo por MSE (media).

Notas:
- Diseñado para ser razonablemente rápido en CI/CPU (REPEATS=3, EPOCHS=3).
- La prueba no fuerza que un modelo gane; sólo valida y reporta estadísticas
  y marca cuál tiene menor MSE medio.
"""

import time
import math

import numpy as np
import pytest
import torch

from src.data.mock_lima_tensor import mock_lima_graph
from src.config import Config
from src.train.trainer import build_model, train_epoch, evaluate_model
from src.utils.seed import set_seed


REPEATS = 3
EPOCHS = 3
NUM_TIME_STEPS = 6
MODELS = ["gcn_gru", "graphsage_gru"]
DEVICE = "cpu"


def _ensure_tensor_attr(data, name, dtype=torch.float32):
    val = getattr(data, name, None)
    if val is None:
        return
    if not torch.is_tensor(val):
        # Masks may be boolean arrays
        if dtype is torch.bool:
            setattr(data, name, torch.tensor(val, dtype=torch.bool))
        else:
            setattr(data, name, torch.tensor(val, dtype=dtype))


def _summarize_list(vals):
    a = np.array(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def test_compare_gru_models_collect_stats():
    """Ejecuta varias corridas cortas, recopila métricas y tiempos.

    - Valida que las métricas son numéricas y finitas
    - Devuelve un `pytest.summary` con el informe para inspección manual
    """

    results = {m: {"metrics": [], "train_times": [], "eval_times": [], "num_params": []} for m in MODELS}

    for r in range(REPEATS):
        seed = 1000 + r
        set_seed(seed)
        data, metadata = mock_lima_graph(seed=seed, num_time_steps=NUM_TIME_STEPS)

        # Asegurar tipos torch (mock_lima_tensor debería devolver tensores,
        # pero convertir por seguridad)
        _ensure_tensor_attr(data, "x", torch.float32)
        _ensure_tensor_attr(data, "y", torch.float32)
        _ensure_tensor_attr(data, "x_temporal", torch.float32)
        _ensure_tensor_attr(data, "edge_attr", torch.float32)
        _ensure_tensor_attr(data, "train_mask", torch.bool)
        _ensure_tensor_attr(data, "test_mask", torch.bool)

        data = data.to(DEVICE)

        for model_name in MODELS:
            config = Config(
                model=model_name,
                hidden_dim=32,
                num_layers=2,
                gru_hidden_dim=32,
                gru_num_layers=1,
                epochs=EPOCHS,
                num_time_steps=NUM_TIME_STEPS,
                lr=0.01,
                device=DEVICE,
                seed=seed,
            )

            model = build_model(config, data.x.shape[1], data.edge_attr.shape[1]).to(DEVICE)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            criterion = torch.nn.MSELoss()

            # Entrenamiento (timed)
            t0 = time.perf_counter()
            for epoch in range(config.epochs):
                loss = train_epoch(model, data, optimizer, criterion, config)
            train_time = time.perf_counter() - t0

            # Evaluación (timed)
            t1 = time.perf_counter()
            metrics, preds = evaluate_model(model, data, config)
            eval_time = time.perf_counter() - t1

            # Sanity checks básicos
            assert isinstance(metrics, dict)
            assert set(metrics.keys()) == {"mse", "rmse", "mae", "mape", "r2"}

            results[model_name]["metrics"].append(metrics)
            results[model_name]["train_times"].append(train_time)
            results[model_name]["eval_times"].append(eval_time)
            results[model_name]["num_params"].append(num_params)

    # Resumir estadísticas
    summary = {}
    for m in MODELS:
        metric_keys = results[m]["metrics"][0].keys()
        metrics_summary = {}
        for k in metric_keys:
            vals = [d[k] for d in results[m]["metrics"]]
            metrics_summary[k] = _summarize_list(vals)

        summary[m] = {
            "metrics": metrics_summary,
            "train_time": _summarize_list(results[m]["train_times"]),
            "eval_time": _summarize_list(results[m]["eval_times"]),
            "num_params": _summarize_list(results[m]["num_params"]),
            "repeats": len(results[m]["metrics"]),
        }

    # Mejor modelo por MSE (media más baja)
    mse_means = {m: summary[m]["metrics"]["mse"]["mean"] for m in MODELS}
    best_model = min(mse_means, key=mse_means.get)
    summary["best_model"] = best_model

    # Asserts: todas las estadísticas son finitas y std >= 0
    for m in MODELS:
        for metric_name, stats in summary[m]["metrics"].items():
            assert math.isfinite(stats["mean"])
            assert stats["std"] >= 0
        assert math.isfinite(summary[m]["train_time"]["mean"])
        assert summary[m]["train_time"]["std"] >= 0
        assert math.isfinite(summary[m]["eval_time"]["mean"])
        assert summary[m]["eval_time"]["std"] >= 0
        assert isinstance(summary[m]["repeats"], int) and summary[m]["repeats"] == REPEATS

    # Mejor modelo calculado debe ser uno de los dos
    assert summary["best_model"] in MODELS

    # Guardar resumen en pytest para inspección manual opcional
    pytest.summary = summary
