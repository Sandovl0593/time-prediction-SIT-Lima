"""Adapta las estaciones crudas de Lima al esquema usado por el proyecto.

Uso:
    python -m src.data.rawLima.adapt_stations
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "Station ID",
    "Station Name",
    "Line",
    "Stop Sequence",
    "Mode",
    "Structure",
    "GTFS Latitude",
    "GTFS Longitude",
    "Density Workday",
    "Density Workday Night",
    "Density Holiday Day",
    "Density Holiday Night",
]


def _parse_coordinates(value: object) -> tuple[float, float]:
    """Devuelve latitud y longitud desde un valor tipo ``(lat, lon)``."""
    try:
        coordinates = ast.literal_eval(str(value).strip())
        if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
            raise ValueError
        return float(coordinates[0]), float(coordinates[1])
    except (ValueError, SyntaxError, TypeError, IndexError) as exc:
        raise ValueError(f"Coordenadas inválidas: {value!r}") from exc


def adapt_stations(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Convierte el CSV crudo de Lima y escribe el esquema normalizado."""
    source = pd.read_csv(input_path, sep=";", encoding="utf-8-sig")
    required = {"estacion", "servicio", "coordenadas"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    coordinates = source["coordenadas"].map(_parse_coordinates)
    adapted = pd.DataFrame(
        {
            "Station ID": [f"LIM-{index:04d}" for index in range(1, len(source) + 1)],
            "Station Name": source["estacion"].astype(str).str.strip(),
            "Line": source["servicio"].astype(str).str.strip(),
            "Stop Sequence": source.groupby("servicio", sort=False).cumcount() + 1,
            "Mode": "Sub",
            "Structure": "At Grade",
            "GTFS Latitude": [latitude for latitude, _ in coordinates],
            "GTFS Longitude": [longitude for _, longitude in coordinates],
            "Density Workday": 0,
            "Density Workday Night": 0,
            "Density Holiday Day": 0,
            "Density Holiday Night": 0,
        },
        columns=OUTPUT_COLUMNS,
    )

    if adapted["Station ID"].duplicated().any():
        raise ValueError("Se generaron Station ID duplicados")
    if adapted[["GTFS Latitude", "GTFS Longitude"]].isna().any().any():
        raise ValueError("Hay estaciones sin coordenadas válidas")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_csv(output_path, index=False, encoding="utf-8")
    return adapted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("stations.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("stations_adapted.csv"),
    )
    args = parser.parse_args()
    adapted = adapt_stations(args.input, args.output)
    print(f"Escritas {len(adapted)} estaciones en {args.output}")


if __name__ == "__main__":
    main()