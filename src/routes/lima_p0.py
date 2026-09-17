"""Construye rutas candidatas P0 a partir del grafo de Lima ya etiquetado."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from src.config import KM_BINS
except ModuleNotFoundError:  # permite ejecutar el archivo directamente
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import KM_BINS


def _nearest_bin(length_km: float) -> float:
    return float(min(KM_BINS, key=lambda value: abs(value - length_km)))


def _edge_lookup(edges: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, edge in edges.iterrows():
        key = (str(edge["u"]), str(edge["v"]))
        # The Lima structural graph has one directed edge per pair. Keep the
        # first one defensively if a future source emits parallel variants.
        lookup.setdefault(key, edge)
    return lookup


def build_lima_p0_candidates(processed_dir: Path, output_path: Path) -> pd.DataFrame:
    """Write bidirectional in-line multi-hop routes and one-hop transfers for P0."""
    processed_dir = Path(processed_dir)
    nodes = pd.read_csv(processed_dir / "nodes.csv", low_memory=False)
    edges = pd.read_csv(processed_dir / "edges.csv", low_memory=False)
    required = {"GTFS Stop ID", "Line", "Stop Sequence"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"nodes.csv sin columnas requeridas: {sorted(missing)}")
    if "travel_time_s" not in edges:
        raise ValueError("Ejecute la proyección de tiempos antes de construir rutas P0.")

    edge_type = edges.get("edge_type", pd.Series("in_line", index=edges.index)).fillna("in_line")
    in_line_edges = edges[edge_type != "transfer"].copy()
    lookup = _edge_lookup(in_line_edges)
    rows: list[dict[str, object]] = []

    for line, group in nodes.groupby("Line", sort=False):
        stop_ids = group.sort_values("Stop Sequence")["GTFS Stop ID"].astype(str).tolist()
        for ordered_ids in (stop_ids, list(reversed(stop_ids))):
            for first in range(len(ordered_ids) - 1):
                for last in range(first + 1, len(ordered_ids)):
                    hops = ordered_ids[first : last + 1]
                    hop_edges = [lookup.get((hops[i], hops[i + 1])) for i in range(len(hops) - 1)]
                    if any(edge is None for edge in hop_edges):
                        continue
                    length_m = float(sum(float(edge["length_m"]) for edge in hop_edges if edge is not None))
                    target_s = float(sum(float(edge["travel_time_s"]) for edge in hop_edges if edge is not None))
                    length_km = length_m / 1000.0
                    bin_km = _nearest_bin(length_km)
                    rows.append(
                        {
                            "scenario_id": "P0_Lima",
                            "route_type": "in_line",
                            "line": str(line),
                            "start_stop_id": hops[0],
                            "end_stop_id": hops[-1],
                            "hops": json.dumps(hops),
                            "n_hops": len(hops) - 1,
                            "length_real_km": length_km,
                            "tol_prox": bin_km,
                            "km_offset": length_km - bin_km,
                            "target": target_s,
                        }
                    )

    transfer_edges = edges[edge_type == "transfer"]
    for _, edge in transfer_edges.iterrows():
        length_km = float(edge["length_m"]) / 1000.0
        rows.append(
            {
                "scenario_id": "P0_Lima",
                "route_type": "transfer",
                "line": "TRANSFER",
                "start_stop_id": str(edge["u"]),
                "end_stop_id": str(edge["v"]),
                "hops": json.dumps([str(edge["u"]), str(edge["v"])]),
                "n_hops": 1,
                "length_real_km": length_km,
                "tol_prox": _nearest_bin(length_km),
                "km_offset": length_km - _nearest_bin(length_km),
                "target": float(edge["travel_time_s"]),
            }
        )

    candidates = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("src/data/processedLima"))
    parser.add_argument("--output", type=Path, default=Path("src/outputs/lima/p0_route_candidates.csv"))
    args = parser.parse_args()
    result = build_lima_p0_candidates(args.processed_dir, args.output)
    print(f"Rutas P0 Lima escritas en {args.output}: {len(result)} filas")


if __name__ == "__main__":
    main()
