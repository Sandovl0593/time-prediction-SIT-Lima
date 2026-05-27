"""Entry point para entrenar y evaluar modelos de predicción de tiempos de viaje.

Uso:
    python run.py --model gcn --epochs 50
    python run.py --model graphsage --epochs 50
    python run.py --model gcn_gru --epochs 50
    python run.py --model graphsage_gru --epochs 50
    python run.py --model gcn --hidden_dim 128 --num_layers 3
    python run.py --model gat --epochs 50
    python run.py --model gatv2 --epochs 50
    python run.py --model gat --hidden_dim 128 --num_layers 3
    python run.py --test
"""

import argparse
import sys
import os

from src.data.load_nyc_mta import load_nyc_mta, load_processed_gdfs, visualize_nodes_edges

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
           default="gat",
        choices=Config.VALID_MODELS,
        help="Modelo a entrenar",
    )
    parser.add_argument("--process-nyc", action="store_true", help="Procesar datos RAW de NYC y guardar en processed/")
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

    # Si se solicita procesar los datos raw de NYC, ejecutar el pipeline y terminar
    if args.process_nyc:
        base = os.path.join("src", "data")
        base_raw = os.path.join(base, "rawNYC")
        stations = os.path.join(base_raw, "MTA_Subway_Stations.csv")
        stop_times = os.path.join(base_raw, "stop_times.csv")
        trips = os.path.join(base_raw, "trip_times.csv")

        path_processed = os.path.join(base, "processed")

        print("[run.py] Processing raw NYC data and saving processed CSVs...")
        
        processed_dir = os.path.join(base, "processed")
        if not os.path.exists(path_processed):
            load_nyc_mta(stations, stop_times_path=stop_times, trips_path=trips)
            # print(f"[run.py] Processed files saved to: {processed_dir}")

        print("[run.py] Loading processed CSVs for visualization...")
        gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)

        print("[run.py] Visualizing processed data (loaded from CSV)...")
        visualize_nodes_edges(gdf_nodes, gdf_lines, show_labels=False, node_size=15, edge_color="black")
        return 0

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
