"""Configuración unificada para todos los modelos."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuración central del experimento.

    Attributes:
        model: Nombre del modelo a entrenar.
            Opciones: 'gcn', 'graphsage', 'gat', 'gatv2',
                      'gcn_gru', 'graphsage_gru', 'gat_gru', 'gatv2_gru'
        hidden_dim: Dimensión de las capas ocultas.
        num_layers: Número de capas del encoder de grafos.
        heads: Número de heads de atención (solo para GAT/GATv2).
        gru_hidden_dim: Dimensión oculta del GRU (solo para modelos híbridos).
        gru_num_layers: Número de capas GRU (solo para modelos híbridos).
        dropout: Tasa de dropout.
        lr: Learning rate.
        weight_decay: Regularización L2.
        epochs: Número de épocas de entrenamiento.
        seed: Semilla para reproducibilidad.
        test_ratio: Proporción de aristas para test.
        num_time_steps: Pasos temporales para modelos híbridos (GRU).
        device: Dispositivo de cómputo ('cpu' o 'cuda').
    """

    model: str = "gcn"
    hidden_dim: int = 64
    num_layers: int = 2
    heads: int = 4
    gru_hidden_dim: int = 64
    gru_num_layers: int = 1
    dropout: float = 0.1
    lr: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 100
    seed: int = 42
    test_ratio: float = 0.2
    num_time_steps: int = 12
    device: str = "cpu"

    VALID_MODELS = (
        "gcn", "graphsage", "gat", "gatv2",
        "gcn_gru", "graphsage_gru", "gat_gru", "gatv2_gru",
    )

    def __post_init__(self):
        if self.model not in self.VALID_MODELS:
            raise ValueError(
                f"Modelo '{self.model}' no válido. "
                f"Opciones: {self.VALID_MODELS}"
            )

    @classmethod
    def from_args(cls, args) -> "Config":
        """Crear Config desde un argparse.Namespace."""
        return cls(**{k: v for k, v in vars(args).items() if v is not None})
