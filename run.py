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

from src.data.load_nyc_mta import load_nyc_mta, load_processed_gdfs, visualize_nodes_edges
from src.main_process.route_analysis import analyze_and_select_routes, visualize_segments_csv

from src.config import Config



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
    parser.add_argument("--scenario-id", type=str, default="default", help="Identificador de escenario para nombres de salida")
    parser.add_argument("--cleaning-config", type=str, default=None, help="Ruta a JSON con umbrales/configuración de limpieza")
    
    parser.add_argument("-r", "--select-routes", action="store_true", help="Seleccionar rutas 'rectas' desde processed/ y exportar CSVs")
    parser.add_argument("-k", "--target", type=float, default=10.0, help="Longitud objetivo en km (solo un valor)")
    parser.add_argument("-p", "--curve-penalty", type=float, default=0.5, help="Penalidad por curvas (mayor -> prefiere más rectitud)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.25, help="Tolerancia relativa para distancia objetivo (fracción)")
    parser.add_argument("-v", "--view-file", nargs="?", const=True, default=None, help="Ruta a un CSV precomputado para visualizar. Si se llama sin argumento, usa -k/-t/-p para localizar el CSV.")
    
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

        # Ejecutar pipeline parametrizado
        load_nyc_mta(
            stations,
            stop_times_path=stop_times,
            trips_path=trips,
            scenario_id=args.scenario_id,
            select_routes=None,
            target_km=args.target,
            curve_penalty=args.curve_penalty,
            tolerance=args.tolerance,
            cleaning_thresholds=cleaning_thresholds,
        )

        # Si el usuario pidió seleccionar rutas inmediatamente después, ejecutarlo
        if args.select_routes:
            print(f"[run.py] Running route analysis for target={args.target} km (tolerance={args.tolerance})")
            result = analyze_and_select_routes(
                processed_dir=processed_dir,
                target_km=args.target,
                tolerance=args.tolerance,
                curvature_penalty=args.curve_penalty,
            )
            out_dir = result.get("out_dir")
            file_path = result.get("file")
            print(f"[run.py] Route analysis finished. Outputs in: {out_dir}\n    target {args.target} km -> {file_path}")

        # Visualizar si se pidió
        if args.view_file is True:
            print("[run.py] Loading processed CSVs for visualization...")
            gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
            print("[run.py] Visualizing processed data (loaded from CSV)...")
            visualize_nodes_edges(gdf_nodes, gdf_lines, show_labels=False, node_size=5)

        return 0

    # Si se solicita seleccionar rutas (análisis del grafo), ejecutar y terminar
    if args.select_routes:
        processed_dir = os.path.join("src", "data", "processed")
        if not os.path.exists(processed_dir):
            print(f"[run.py] processed dir not found at {processed_dir}. Run --process-nyc first.")
            return 1

        target_km = float(args.target)
        print(f"[run.py] Running route analysis for target={target_km} km (tolerance={args.tolerance})")
        result = analyze_and_select_routes(
            processed_dir=processed_dir,
            target_km=target_km,
            tolerance=args.tolerance,
            curvature_penalty=args.curve_penalty,
        )

        out_dir = result.get("out_dir")
        file_path = result.get("file")

        print(f"\n{'='*60}")
        print(f"  Route analysis finished. Outputs in: {out_dir}")
        print(f"    target {target_km} km -> {file_path}")
        print(f"{'='*60}\n")

        # No visualizar aquí: usar --view-file para mostrar un CSV precomputado
        return 0

    # Si se solicita visualizar un CSV precomputado, mostrarlo y terminar
    if args.view_file is not None:
        # Modo 1: --view-file sin argumento -> construir ruta desde -k/-t/-p
        if args.view_file is True:
            target_km = float(args.target)
            tolerance = args.tolerance
            curvature_penalty = args.curve_penalty

            def _sanitize(val: str) -> str:
                return str(val).replace(".", "p").replace(",", "_").replace(" ", "_")

            km_label = str(int(float(target_km))) if float(target_km).is_integer() else _sanitize(str(target_km))
            cfg_tag = f"target_{km_label}km_pen{_sanitize(str(curvature_penalty))}_tol{_sanitize(str(tolerance))}"
            out_dir = os.path.join("src/outputs", cfg_tag)
            fname = os.path.join(out_dir, f"segments_{km_label}km.csv")

            if not os.path.exists(fname):
                print(f"[run.py] No precomputed CSV found at {fname}")
                if os.path.exists(out_dir):
                    print("[run.py] Files in the directory:")
                    for f in os.listdir(out_dir):
                        print("  " + f)
                return 1

            processed_dir = os.path.join("src", "data", "processed")
            print(f"[run.py] Visualizing precomputed CSV: {fname}")
            visualize_segments_csv(fname, processed_dir=processed_dir)
            return 0

        # Modo 2: --view-file <path>
        vf = args.view_file
        if not os.path.exists(vf):
            print(f"[run.py] view-file not found: {vf}")
            return 1
        processed_dir = os.path.join("src", "data", "processed")
        print(f"[run.py] Visualizing precomputed CSV: {vf}")
        visualize_segments_csv(vf, processed_dir=processed_dir)
        return 0

    print(f"\n{'='*60}")
    print(f"  Predicción de Tiempos de Viaje — Lima Transport Network")
    print(f"  Modelo: {config.model.upper()}")
    print(f"{'='*60}\n")

    # Importar el módulo de entrenamiento aquí para evitar dependencias
    # al ejecutar herramientas de preprocesado o análisis solamente.
    from src.train.trainer import train_and_evaluate

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
