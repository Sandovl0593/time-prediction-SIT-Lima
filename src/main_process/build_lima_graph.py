"""Construye el grafo estructural de Lima desde ``stations_adapted.csv``."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


REQUIRED_COLUMNS = {
    "Station ID",
    "Station Name",
    "Line",
    "Stop Sequence",
    "GTFS Latitude",
    "GTFS Longitude",
}


def _haversine_meters(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    radius_m = 6_371_000.0
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = math.radians(latitude_b - latitude_a)
    delta_lon = math.radians(longitude_b - longitude_a)
    hav = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(hav))


def build_lima_graph(stations_path: Path, output_dir: Path) -> nx.MultiDiGraph:
    """Crea un MultiDiGraph dirigido con conexiones entre paradas consecutivas."""
    stations = pd.read_csv(stations_path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(stations.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    stations = stations.copy()
    stations["Station ID"] = stations["Station ID"].astype(str).str.strip()
    stations["Line"] = stations["Line"].astype(str).str.strip()
    stations["Stop Sequence"] = pd.to_numeric(stations["Stop Sequence"], errors="raise")
    stations["GTFS Latitude"] = pd.to_numeric(stations["GTFS Latitude"], errors="raise")
    stations["GTFS Longitude"] = pd.to_numeric(stations["GTFS Longitude"], errors="raise")
    if stations["Station ID"].duplicated().any():
        raise ValueError("Station ID debe ser único para construir el grafo")

    graph = nx.MultiDiGraph(name="Lima structural graph")
    for _, row in stations.iterrows():
        attributes = row.to_dict()
        station_id = str(attributes.pop("Station ID"))
        attributes["GTFS Stop ID"] = station_id
        attributes["lat"] = float(attributes["GTFS Latitude"])
        attributes["lon"] = float(attributes["GTFS Longitude"])
        graph.add_node(station_id, **attributes)

    for line, line_stations in stations.groupby("Line", sort=False):
        ordered = line_stations.sort_values("Stop Sequence")
        consecutive = ordered.to_dict("records")
        for first, second in zip(consecutive, consecutive[1:]):
            first_id = str(first["Station ID"])
            second_id = str(second["Station ID"])
            length_m = _haversine_meters(
                float(first["GTFS Latitude"]),
                float(first["GTFS Longitude"]),
                float(second["GTFS Latitude"]),
                float(second["GTFS Longitude"]),
            )
            edge_geometry = LineString(
                [
                    (float(first["GTFS Longitude"]), float(first["GTFS Latitude"])),
                    (float(second["GTFS Longitude"]), float(second["GTFS Latitude"])),
                ]
            )
            edge_attributes = {
                "line": line,
                "length_m": length_m,
                "spatial": True,
                "structure": "At Grade",
                "geometry": edge_geometry,
            }
            graph.add_edge(first_id, second_id, **edge_attributes)
            graph.add_edge(
                second_id,
                first_id,
                **{**edge_attributes, "geometry": LineString(list(edge_geometry.coords)[::-1])},
            )

    _save_graph(graph, stations, output_dir)
    return graph


def _save_graph(graph: nx.MultiDiGraph, stations: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = []
    for station_id, attributes in graph.nodes(data=True):
        node_record = {"GTFS Stop ID": station_id, **attributes}
        node_record["geometry_wkt"] = Point(
            node_record["lon"], node_record["lat"]
        ).wkt
        nodes.append(node_record)
    pd.DataFrame(nodes).to_csv(output_dir / "nodes.csv", index=False)

    edges = []
    for source, target, key, attributes in graph.edges(keys=True, data=True):
        edge_record = {"u": source, "v": target, "key": key, **attributes}
        edge_record["geometry_wkt"] = edge_record.pop("geometry").wkt
        edges.append(edge_record)
    pd.DataFrame(edges).to_csv(output_dir / "edges.csv", index=False)

    lines = []
    for line, line_stations in stations.groupby("Line", sort=False):
        ordered = line_stations.sort_values("Stop Sequence")
        lines.append(
            {
                "Line": line,
                "num_stations": len(ordered),
                "station_ids": "|".join(ordered["Station ID"].astype(str)),
                "geometry_wkt": LineString(
                    list(zip(ordered["GTFS Longitude"], ordered["GTFS Latitude"]))
                ).wkt,
            }
        )
    pd.DataFrame(lines).to_csv(output_dir / "lines.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "rawLima" / "stations_adapted.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "data" / "processedLima",
    )
    args = parser.parse_args()
    graph = build_lima_graph(args.stations, args.output_dir)
    print(f"Grafo escrito en {args.output_dir}: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas")


if __name__ == "__main__":
    main()