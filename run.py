"""Entry point para entrenar y evaluar modelos de predicción de tiempos de viaje.

Uso:
    python run.py --model graphsage --epochs 50
    python run.py --model gat --epochs 50
    python run.py --model gatv2 --epochs 50
    python run.py --model gat --hidden_dim 128 --num_layers 3
    python run.py --test
"""

import argparse
import sys
import os
import json

from src.main_process.general_pipeline import general_pipeline, load_processed_gdfs
from src.main_process.visualize import visualize_segments_csv, visualize_nodes_edges

from src.config import Config

def _sanitize(val: str) -> str:
    return str(val).replace(".", "p").replace(",", "_").replace(" ", "_")

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
    parser.add_argument("--show-processed", action="store_true", help="Cargar y mostrar visualización desde src/data/processed/graph sin parámetros")
    parser.add_argument("--cleaning-config", type=str, default=None, help="Ruta a JSON con umbrales/configuración de limpieza")
    parser.add_argument("--clean-outputs", action="store_true", help="Eliminar la carpeta src/outputs (limpieza de artefactos) antes de ejecutar")
    
    parser.add_argument("-v", "--view-file", nargs="?", const=True, default=None, help="Ruta a un CSV precomputado para visualizar. Si se llama sin argumento, usa -k/-t/-p para localizar el CSV.")
    
    parser.add_argument("-k", "--target", type=float, default=10.0, help="Longitud objetivo en km (solo un valor)")
    parser.add_argument("-p", "--curve-penalty", type=float, default=0.5, help="Penalidad por curvas (mayor -> prefiere más rectitud)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.25, help="Tolerancia relativa para distancia objetivo (fracción)")
    parser.add_argument("-n", "--n-segments", type=int, default=None, help="Número de tramos del top a visualizar (solo para --view-file)")
    
    parser.add_argument("--hidden_dim", type=int, default=None, help="Dimensión oculta")
    parser.add_argument("--num_layers", type=int, default=None, help="Capas del encoder de grafos")
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
        "hidden_dim", "num_layers",
        "dropout", "lr", "weight_decay", "epochs", "seed", "num_time_steps",
        "device",
    ]:
        val = getattr(args, key)
        if val is not None:
            config_kwargs[key] = val

    config = Config(**config_kwargs)

    # Si se solicita procesar los datos raw de NYC, ejecutar el pipeline y opcionalmente análisis de rutas
    if args.process_nyc:
        base = os.path.join("src", "data")
        base_raw = os.path.join(base, "rawNYC")
        stations = os.path.join(base_raw, "MTA_Subway_Stations.csv")
        stop_times = os.path.join(base_raw, "stop_times.csv")
        trips = os.path.join(base_raw, "trip_times.csv")

        processed_dir = os.path.join(base, "processed")

        print("[run.py] Processing raw NYC data and saving processed CSVs...")

        # cargar configuración de limpieza si se proporcionó
        cleaning_thresholds = None
        if args.cleaning_config:
            try:
                with open(args.cleaning_config, "r", encoding="utf-8") as fh:
                    cleaning_thresholds = json.load(fh)
            except Exception as e:
                print(f"[run.py] Warning: failed loading cleaning config {args.cleaning_config}: {e}")

        # Opcional: limpiar outputs si se pidió
        if args.clean_outputs:
            import shutil
            out_root = os.path.join("src", "outputs")
            if os.path.exists(out_root):
                print(f"[run.py] Removing outputs directory {out_root}")
                shutil.rmtree(out_root)
            else:
                print(f"[run.py] No outputs directory to remove at {out_root}")

        km_label = str(int(float(args.target))) if float(args.target).is_integer() else _sanitize(str(args.target))
        scenario_tag =f"target_{km_label}km_pen{_sanitize(str(args.curve_penalty))}_tol{_sanitize(str(args.tolerance))}"

        # Ejecutar pipeline parametrizado (scenario_tag se usa como scenario_id internamente)
        general_pipeline(
            stations,
            stop_times_path=stop_times,
            trips_path=trips,
            scenario_id=scenario_tag,
            target_km=args.target,
            curve_penalty=args.curve_penalty,
            tolerance=args.tolerance,
            cleaning_thresholds=cleaning_thresholds,
        )

        return 0

    # Visualizar si se pidió
    if args.view_file is True:
        km_label = str(int(float(args.target))) if float(args.target).is_integer() else _sanitize(str(args.target))
        scenario_tag =f"target_{km_label}km_pen{_sanitize(str(args.curve_penalty))}_tol{_sanitize(str(args.tolerance))}"

        print("[run.py] Loading processed CSVs for visualization...")
        # Prefer visualizing top-k segments from the generated route.csv if available
        out_dir = os.path.join("src", "outputs", scenario_tag)
        route_csv_path = os.path.join(out_dir, "route.csv")
        if os.path.exists(route_csv_path):
            try:
                # Export top-k using the exporter and visualize those segments
                from src.routes.export_top_segments import export_top_k_from_route_csv
                print(f"[run.py] Found route.csv at {route_csv_path}; exporting top segments...")
                k = int(args.n_segments) if args.n_segments is not None else 50
                exported = export_top_k_from_route_csv(route_csv_path, k=k)
                print(f"[run.py] Visualizing exported top-{k} segments: {exported}")
                visualize_segments_csv(exported, processed_dir=scenario_tag, show_nodes=True, figsize=(10,10))
            except Exception as e:
                print(f"[run.py] Warning: failed exporting/visualizing top segments: {e}")
                try:
                    gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
                    visualize_nodes_edges(gdf_nodes, gdf_lines, show_labels=False, node_size=5)
                except Exception as e2:
                    print(f"[run.py] Failed fallback visualization: {e2}")
        else:
            # fallback: visualize processed graph (lines + nodes)
            try:
                gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
                print("[run.py] Visualizing processed data (loaded from CSV)...")
                visualize_nodes_edges(gdf_nodes, gdf_lines, show_labels=False, node_size=5)
            except Exception as e:
                print(f"[run.py] Failed to visualize processed graph: {e}")

        return 0

    # Mostrar processed/graph si se pide explícitamente
    if args.show_processed:
        processed_dir = os.path.join("src", "data", "processed", "graph")
        print("[run.py] Showing processed graph visualization from src/data/processed/graph")
        try:
            visualize_nodes_edges(processed_dir=processed_dir, show_labels=False, node_size=5)
        except Exception as e:
            print(f"[run.py] Failed to visualize processed graph: {e}")
            return 1
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
