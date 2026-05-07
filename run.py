"""Entry point para entrenar y evaluar modelos de predicción de tiempos de viaje.

Uso:
    python run.py --model gcn --epochs 50
    python run.py --model graphsage --epochs 50
    python run.py --model gcn_gru --epochs 50
    python run.py --model graphsage_gru --epochs 50
    python run.py --model gcn --hidden_dim 128 --num_layers 3
"""

import argparse
import sys

from src.config import Config
from src.train.trainer import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predicción de tiempos de viaje — Red de transporte de Lima",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=Config.VALID_MODELS,
        help="Modelo a entrenar",
    )
    parser.add_argument("--hidden_dim", type=int, default=None, help="Dimensión oculta")
    parser.add_argument("--num_layers", type=int, default=None, help="Capas del encoder de grafos")
    parser.add_argument("--gru_hidden_dim", type=int, default=None, help="Dimensión oculta del GRU")
    parser.add_argument("--gru_num_layers", type=int, default=None, help="Capas del GRU")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="L2 regularization")
    parser.add_argument("--epochs", type=int, default=None, help="Número de épocas")
    parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria")
    parser.add_argument("--num_time_steps", type=int, default=None, help="Pasos temporales (GRU)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo de cómputo",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Construir config, solo sobrescribir valores explícitos
    config_kwargs = {"model": args.model}
    for key in [
        "hidden_dim", "num_layers", "gru_hidden_dim", "gru_num_layers",
        "dropout", "lr", "weight_decay", "epochs", "seed", "num_time_steps",
        "device",
    ]:
        val = getattr(args, key)
        if val is not None:
            config_kwargs[key] = val

    config = Config(**config_kwargs)

    print(f"\n{'='*60}")
    print(f"  Predicción de Tiempos de Viaje — Lima Transport Network")
    print(f"  Modelo: {config.model.upper()}")
    print(f"{'='*60}\n")

    results = train_and_evaluate(config)

    print(f"\n{'='*60}")
    print(f"  Resumen: {results['model'].upper()}")
    print(f"  Parámetros: {results['num_params']:,}")
    print(f"  Tiempo: {results['elapsed_seconds']:.1f}s")
    print(f"  Best Test MSE: {results['best_metrics'].get('mse', 'N/A')}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
