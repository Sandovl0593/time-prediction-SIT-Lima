"""Visualiza el grafo estructural de Lima guardado en CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt


def _load_geometry_column(dataframe: pd.DataFrame, column: str) -> list:
    if column not in dataframe.columns:
        raise ValueError(f"El CSV no contiene la columna '{column}'")
    return [
        wkt.loads(value)
        for value in dataframe[column]
        if pd.notna(value) and str(value).strip()
    ]


def visualize_lima(
    processed_dir: str | Path = "src/data/processedLima",
    show_labels: bool = False,
    figsize: tuple[int, int] = (12, 10),
    save_path: str | Path | None = None,
):
    """Lee nodes.csv, edges.csv y lines.csv y dibuja el grafo de Lima."""
    processed_dir = Path(processed_dir)
    nodes = pd.read_csv(processed_dir / "nodes.csv")
    edges = pd.read_csv(processed_dir / "edges.csv")
    lines = pd.read_csv(processed_dir / "lines.csv")

    node_points = _load_geometry_column(nodes, "geometry_wkt")
    edge_lines = _load_geometry_column(edges, "geometry_wkt")
    line_geometries = _load_geometry_column(lines, "geometry_wkt")

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")
    for index, geometry in enumerate(line_geometries):
        x_values, y_values = geometry.xy
        ax.plot(
            x_values,
            y_values,
            color=colors(index % 10),
            linewidth=2.2,
            alpha=0.85,
            label=str(lines.iloc[index]["Line"]),
            zorder=1,
        )

    for geometry in edge_lines:
        x_values, y_values = geometry.xy
        ax.plot(x_values, y_values, color="gray", linewidth=0.7, alpha=0.25, zorder=2)

    ax.scatter(
        [point.x for point in node_points],
        [point.y for point in node_points],
        s=18,
        color="#202020",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )

    if show_labels:
        for _, row in nodes.iterrows():
            point = wkt.loads(row["geometry_wkt"])
            ax.annotate(
                str(row.get("Station Name", row["GTFS Stop ID"])),
                (point.x, point.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                zorder=4,
            )

    ax.set_title("Grafo estructural de estaciones de Lima")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    if len(line_geometries) <= 10:
        ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("src/data/processedLima"))
    parser.add_argument("--labels", action="store_true", help="Mostrar nombre de cada estación")
    parser.add_argument("--save", type=Path, help="Guardar también la imagen en esta ruta")
    args = parser.parse_args()
    visualize_lima(args.processed_dir, show_labels=args.labels, save_path=args.save)


if __name__ == "__main__":
    main()