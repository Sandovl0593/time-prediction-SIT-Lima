"""Proyecta etiquetas temporales de escenario desde NYC hacia el grafo de Lima.

Las tasas de referencia se calculan exclusivamente con aristas observadas de
NYC (``travel_time_s / length_m``). El resultado no modifica el CSV crudo de
Lima: escribe etiquetas proyectadas y trazables en ``processedLima``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-9


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalise_structure(value: object) -> str:
    return str(value).strip().casefold() if pd.notna(value) else "unknown"


def _reference_rates(nyc_graph_dir: Path) -> pd.DataFrame:
    """Load valid observed NYC rates and annotate source-node structure."""
    edges = pd.read_csv(nyc_graph_dir / "edges.csv", low_memory=False)
    nodes = pd.read_csv(nyc_graph_dir / "nodes.csv", low_memory=False)
    required = {"u", "v", "length_m", "travel_time_s"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"Aristas NYC sin columnas requeridas: {sorted(missing)}")

    edges = edges.copy()
    if "observed" in edges:
        edges = edges[edges["observed"].map(_as_bool)]
    edges["length_m"] = pd.to_numeric(edges["length_m"], errors="coerce")
    edges["travel_time_s"] = pd.to_numeric(edges["travel_time_s"], errors="coerce")
    edges = edges[
        edges["length_m"].gt(0)
        & edges["travel_time_s"].gt(0)
        & np.isfinite(edges["length_m"])
        & np.isfinite(edges["travel_time_s"])
    ].copy()
    if edges.empty:
        raise ValueError("No hay aristas observadas con tiempo y longitud válidos en NYC.")

    if "Structure" not in nodes.columns:
        edges["reference_structure"] = "unknown"
    else:
        node_structure = dict(
            zip(nodes["GTFS Stop ID"].astype(str), nodes["Structure"].map(_normalise_structure))
        )
        edges["reference_structure"] = edges["u"].astype(str).map(node_structure)
    edges["reference_structure"] = edges["reference_structure"].fillna("unknown")
    edges["rate_s_per_m"] = edges["travel_time_s"] / edges["length_m"]
    return edges[["length_m", "rate_s_per_m", "reference_structure"]].reset_index(drop=True)


def _line_mode(line: object) -> str:
    return "brt" if str(line).strip().upper() == "BRT" else "rail"


def _edge_direction(edge: pd.Series, nodes: pd.DataFrame) -> int | None:
    """Infer direction from stop order for trip metadata; transfers have none."""
    if str(edge.get("edge_type", "in_line")) == "transfer":
        return None
    if "Stop Sequence" not in nodes.columns:
        return None
    sequence = dict(zip(nodes["GTFS Stop ID"].astype(str), nodes["Stop Sequence"]))
    first = pd.to_numeric(pd.Series([sequence.get(str(edge["u"]))]), errors="coerce").iloc[0]
    second = pd.to_numeric(pd.Series([sequence.get(str(edge["v"]))]), errors="coerce").iloc[0]
    if pd.isna(first) or pd.isna(second):
        return None
    return int(second < first)


def project_lima_times(
    lima_graph_dir: Path,
    nyc_graph_dir: Path,
    *,
    k_neighbors: int = 12,
    brt_scale: float = 1.15,
    rail_scale: float = 1.0,
    transfer_fixed_s: float = 45.0,
    walking_speed_mps: float = 1.2,
) -> dict[str, object]:
    """Project ``travel_time_s`` for every Lima edge and save CSV artifacts.

    In-line edges use inverse-distance weighting over the ``k_neighbors`` NYC
    reference rates. Distances compare log-length and source structure. BRT
    receives a scenario scale because the reference network is rail-based.
    Transfer edges use a transparent walking rule: fixed access time plus
    walking distance divided by ``walking_speed_mps``.
    """
    if k_neighbors < 1:
        raise ValueError("k_neighbors debe ser al menos 1.")
    if brt_scale <= 0 or rail_scale <= 0 or walking_speed_mps <= 0:
        raise ValueError("Los factores y la velocidad deben ser positivos.")

    lima_graph_dir = Path(lima_graph_dir)
    nyc_graph_dir = Path(nyc_graph_dir)
    edges_path = lima_graph_dir / "edges.csv"
    nodes_path = lima_graph_dir / "nodes.csv"
    edges = pd.read_csv(edges_path, low_memory=False)
    nodes = pd.read_csv(nodes_path, low_memory=False)
    references = _reference_rates(nyc_graph_dir)

    required_lima = {"u", "v", "key", "length_m"}
    missing = required_lima.difference(edges.columns)
    if missing:
        raise ValueError(f"Aristas Lima sin columnas requeridas: {sorted(missing)}")
    if "edge_type" not in edges:
        edges["edge_type"] = "in_line"

    edges = edges.copy()
    edges["length_m"] = pd.to_numeric(edges["length_m"], errors="coerce")
    if edges["length_m"].isna().any() or edges["length_m"].le(0).any():
        raise ValueError("Todas las aristas de Lima requieren length_m positivo.")

    global_rate = float(references["rate_s_per_m"].median())
    projected_rows: list[dict[str, object]] = []
    for _, edge in edges.iterrows():
        line = edge.get("source_line") if str(edge["edge_type"]) == "transfer" else edge.get("line")
        mode = _line_mode(line)
        length_m = float(edge["length_m"])

        if str(edge["edge_type"]) == "transfer":
            time_s = float(transfer_fixed_s + length_m / walking_speed_mps)
            projected_rows.append(
                {
                    "travel_time_s": time_s,
                    "time_source": "projected_lima_scenario",
                    "projection_method": "transfer_walk_rule",
                    "projection_distance": 0.0,
                    "projection_quality": "rule_based_transfer",
                    "reference_count": 0,
                    "temporal_rate_s_per_m": time_s / length_m,
                }
            )
            continue

        target_structure = _normalise_structure(edge.get("structure"))
        length_distance = np.abs(np.log(references["length_m"].to_numpy() / length_m))
        structure_penalty = np.where(
            references["reference_structure"].to_numpy() == target_structure,
            0.0,
            0.5,
        )
        distance = length_distance + structure_penalty
        k = min(k_neighbors, len(references))
        selected = np.argpartition(distance, k - 1)[:k]
        selected_distance = distance[selected]
        weights = 1.0 / (selected_distance + EPS)
        rate = float(np.average(references["rate_s_per_m"].to_numpy()[selected], weights=weights))
        scale = brt_scale if mode == "brt" else rail_scale
        quality = "matched_structure" if np.any(structure_penalty[selected] == 0.0) else "mode_fallback"
        projected_rows.append(
            {
                "travel_time_s": float(length_m * rate * scale),
                "time_source": "projected_lima_scenario",
                "projection_method": "knn_inverse_distance_rate",
                "projection_distance": float(np.average(selected_distance, weights=weights)),
                "projection_quality": quality,
                "reference_count": int(k),
                "temporal_rate_s_per_m": float(rate * scale),
            }
        )

    projected = pd.DataFrame(projected_rows)
    for column in projected.columns:
        edges[column] = projected[column]
    edges["observed"] = False
    edges.to_csv(edges_path, index=False)

    trip_rows = []
    for _, edge in edges.iterrows():
        edge_type = str(edge["edge_type"])
        route_id = "TRANSFER" if edge_type == "transfer" else str(edge.get("line", ""))
        token = f"LIMA_P0_{route_id}_{edge['u']}_{edge['v']}_{edge['key']}"
        trip_rows.append(
            {
                "trip_uid": token,
                "trip_id": token,
                "route_id": route_id,
                "direction_id": _edge_direction(edge, nodes),
                "start_time": pd.NA,
                "vehicle_id": pd.NA,
                "u": edge["u"],
                "v": edge["v"],
                "key": edge["key"],
                "travel_time_s": edge["travel_time_s"],
                "time_source": edge["time_source"],
                "projection_method": edge["projection_method"],
                "projection_quality": edge["projection_quality"],
            }
        )
    projected_trips_path = lima_graph_dir / "projected_trip_times.csv"
    pd.DataFrame(trip_rows).to_csv(projected_trips_path, index=False)

    report = {
        "lima_edges": int(len(edges)),
        "in_line_edges": int((edges["edge_type"] != "transfer").sum()),
        "transfer_edges": int((edges["edge_type"] == "transfer").sum()),
        "nyc_reference_edges": int(len(references)),
        "global_reference_rate_s_per_m": global_rate,
        "k_neighbors": k_neighbors,
        "brt_scale": brt_scale,
        "rail_scale": rail_scale,
        "transfer_fixed_s": transfer_fixed_s,
        "walking_speed_mps": walking_speed_mps,
        "edges_path": str(edges_path),
        "projected_trips_path": str(projected_trips_path),
    }
    report_path = lima_graph_dir / "lima_time_projection_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lima-graph-dir", type=Path, default=Path("src/data/processedLima"))
    parser.add_argument("--nyc-graph-dir", type=Path, default=Path("src/data/processed/graph"))
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--brt-scale", type=float, default=1.15)
    parser.add_argument("--rail-scale", type=float, default=1.0)
    parser.add_argument("--transfer-fixed-s", type=float, default=45.0)
    parser.add_argument("--walking-speed-mps", type=float, default=1.2)
    args = parser.parse_args()
    report = project_lima_times(
        args.lima_graph_dir,
        args.nyc_graph_dir,
        k_neighbors=args.k_neighbors,
        brt_scale=args.brt_scale,
        rail_scale=args.rail_scale,
        transfer_fixed_s=args.transfer_fixed_s,
        walking_speed_mps=args.walking_speed_mps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
