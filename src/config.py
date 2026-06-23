"""Configuración unificada para todos los modelos."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuración central del experimento.

    Attributes:
        model: Nombre del modelo a entrenar.
            Opciones: 'gat', 'gatv2', 'graphsage'
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

    # ---------- Paths ----------
    data_dir = Path("src/data/processed")
    graph_dir = data_dir / "graph"
    # ---------- Model ----------
    model: str = "gatv2"      # graphsage | gat | gatv2
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    # ---------- Training ----------
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 1
    seed: int = 42

    # ---------- Early stopping ----------
    patience: int = 25
    # ---------- Logs ----------
    eval_every: int = 5
    print_every: int = 1

    device: str = "cuda"

    VALID_MODELS = (
        "graphsage",
        "gat",
        "gatv2"
    )

    def __post_init__(self):
        self.model = self.model.lower()
        if self.model not in self.VALID_MODELS:
            raise ValueError(
                f"Modelo inválido {self.model}"
            )
        if self.model in ["gat", "gatv2"] and self.num_layers < 2:
            raise ValueError(
                "GAT y GATv2 requieren num_layers>=2"
            )

    @classmethod
    def from_args(cls, args) -> "Config":
        """Crear Config desde un argparse.Namespace."""
        return cls(**{k: v for k, v in vars(args).items() if v is not None})
