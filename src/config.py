"""Configuración unificada para todos los modelos."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

KM_BINS_DEFAULT: List[float] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

# Rangos de parámetros: (ideal/estricto, peor caso realista/permisivo)
# curve_penalty  ideal=0.5 → penaliza fuertemente las curvas; mínimo realista=0.1
# km_tolerance   ideal=0.2 → ±20% del bin; máximo realista=0.5 → ±50%
CURVE_PENALTY_RANGE = (0.5, 0.1)
KM_TOLERANCE_RANGE  = (0.2, 0.5)

# ---------- Route configurations A/B/C ----------
# Cada config define un par (km_tolerance, curve_penalty) dentro de los rangos.
# A: estricta en km, alta penalidad → rutas rectas bien ajustadas al bin.
# B: tolerancia media, penalidad media → balance entre cobertura y rectitud.
# C: tolerancia amplia, baja penalidad → máxima cobertura de candidatos.
ROUTE_CONFIGS = {
    "A": {"km_tolerance": 0.2, "curve_penalty": 0.5},
    "B": {"km_tolerance": 0.3, "curve_penalty": 0.3},
    "C": {"km_tolerance": 0.4, "curve_penalty": 0.2},
}

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
        straightness_threshold: Umbral mínimo de índice de rectitud para seleccionar rutas rectas.
    """

    # ---------- Paths ----------
    data_dir = Path("src/data/processed")
    graph_dir = data_dir / "graph"
    routes_output_dir = Path("src/outputs/routes")
    training_output_dir = Path("src/outputs/training")

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

    # ---------- Routes ----------
    straightness_threshold: float = 0.9
    base_curve_penalty: float = 0.5
    base_km_tolerance_ratio: float = 0.3

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
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio debe sumar 1.0 "
                f"(actual: {self.train_ratio + self.val_ratio + self.test_ratio:.6f})"
            )
        if self.epochs <= 0:
            raise ValueError("epochs debe ser > 0")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim debe ser >= 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser > 0")
        if self.patience <= 0:
            raise ValueError("patience debe ser > 0")
        if not isinstance(self.seed, int):
            raise ValueError("seed debe ser un entero")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device debe ser 'cpu' o 'cuda'")

    @classmethod
    def from_args(cls, args) -> "Config":
        """Crear Config desde un argparse.Namespace."""
        return cls(**{k: v for k, v in vars(args).items() if v is not None})
