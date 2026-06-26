"""Entry point para entrenar y evaluar modelos de predicción de tiempos de viaje.

Uso:
    python run.py --model graphsage --epochs 50
    python run.py --model gat --epochs 50
    python run.py --model gatv2 --epochs 50
    python run.py --model gat --hidden_dim 128 --num_layers 3
    python run.py --process-nyc
"""

import argparse
import sys
import os
import json

from src.main_process.general_pipeline import general_pipeline, load_processed_gdfs
from src.main_process.visualize import visualize_segments_csv, visualize_nodes_edges
from src.train.trainer import train_and_evaluate

from src.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predicción de tiempos de viaje — Red de transporte de Lima",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=Config.VALID_MODELS,
        help="Modelo a entrenar",
    )
    parser.add_argument("-p", "--process-nyc", action="store_true", help="Procesar datos RAW de NYC y guardar en processed/")
    parser.add_argument("-s", "--show-processed", action="store_true", help="Cargar y mostrar visualización desde src/data/processed/graph sin parámetros")
    parser.add_argument("--cleaning-config", type=str, default=None, help="Ruta a JSON con umbrales/configuración de limpieza")
    parser.add_argument("--clean-outputs", action="store_true", help="Eliminar la carpeta src/outputs (limpieza de artefactos) antes de ejecutar")
    parser.add_argument("-r", "--route-analysis", action="store_true", help="Ejecutar análisis de rutas (estadísticas, distribución, rutas rectas)")

    parser.add_argument(
        "--eval-subsets",
        nargs="*",
        default=None,
        metavar="SUBSET",
        help=(
            "Subconjuntos a evaluar tras el entrenamiento. "
            "Opciones: config_A config_B config_C straight. "
            "Sin valor: incluye todos los que tengan CSV disponible. "
            "Ejemplo: --eval-subsets config_A config_B"
        ),
    )
    parser.add_argument(
        "--gen-report",
        action="store_true",
        help="Generar src/outputs/reports/tex_data.json a partir de los artefactos de entrenamiento existentes",
    )

    parser.add_argument("-v", "--view-file", action="store_true", default=False, help="Visualizar top-k segmentos del CSV maestro de rutas")
    parser.add_argument("-c", "--config", type=str, default="A", help="Ruta a CSV de configuración de top-k segmentos (por defecto config_A.csv)")
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

    if args.model:
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

        # Process the model training and evaluation if not in a special mode
        if not (args.process_nyc or args.view_file or args.show_processed):
            # master_segments.csv y straight_routes.csv se buscan en las rutas
            # por defecto; si no existen, train_and_evaluate los omite sin error.
            # Los escenarios A/B/C se derivan automáticamente de eval_master.
            train_and_evaluate(config)

    # Si se solicita procesar los datos raw de NYC, ejecutar el pipeline
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

        general_pipeline(
            stations,
            stop_times_path=stop_times,
            trips_path=trips,
            scenario_id="base",
            cleaning_thresholds=cleaning_thresholds,
        )

        return 0

    # Visualizar si se pidió
    if args.view_file:
        # Usar config_A.csv como vista por defecto (configuración más estricta)
        set_config = args.config if args.config else "A"
        config_csv = os.path.join("src", "topsegments", f"config_{set_config}.csv")
        print("[run.py] Loading config_A segments for visualization...")
        if os.path.exists(config_csv):
            try:
                visualize_segments_csv(config_csv, show_nodes=True, figsize=(10, 10))
            except Exception as e:
                print(f"[run.py] Warning: failed visualizing config segments: {e}")
                processed_dir = os.path.join("src", "data", "processed", "graph")
                try:
                    gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
                    visualize_nodes_edges(gdf_nodes, gdf_lines, show_labels=False, node_size=5)
                except Exception as e2:
                    print(f"[run.py] Failed fallback visualization: {e2}")
        else:
            print(f"[run.py] Config CSV not found at {config_csv}; run --route-analysis first")

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
    
    # Ejecutar análisis de rutas si se pide
    if args.route_analysis:
        from src.routes.route_analysis import run_route_analysis
        print("[run.py] Running route analysis...")
        try:
            result = run_route_analysis()
            print("Artefactos generados:")
            for key, path in result.items():
                print(f"  {key}: {path}")
        except Exception as e:
            print(f"[run.py] Failed to run route analysis: {e}")
            return 1

    # Generar reporte tex_data.json si se pide
    if args.gen_report:
        from src.reports.report_builder import generate_tex_data_report
        print("[run.py] Generando tex_data.json...")
        try:
            out_path = generate_tex_data_report()
            print(f"[run.py] Reporte generado: {out_path}")
        except Exception as e:
            print(f"[run.py] Error al generar reporte: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
