"""Análisis de rutas 'rectas' sobre los datos procesados de NYC.

Este módulo carga los GeoDataFrames procesados desde `src/data/processed/`
y extrae tramos (sub-linestrings) de las `Line` existentes. Cada tramo
recibe una puntuación basada en su rectitud (chord/path length) y una
penalización por desviación respecto a una longitud objetivo.

Funciones principales:
- `analyze_and_select_routes(...)` — interfaz principal, retornando un
  GeoDataFrame con los tramos seleccionados y mostrando la visualización.

El algoritmo es sencillo y está pensado para ser explicable y ajustable
por parámetros desde `run.py`.
"""

from typing import List, Optional, Tuple, Dict
import os

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt

from src.main_process.general_pipeline import load_processed_gdfs

def visualize_nodes_edges(
    processed_dir: Optional[str] = None,
    show_labels: bool = False,
    figsize: Tuple[int, int] = (10, 10),
    node_size: int = 30,
    node_color: str = "tab:red",
    edge_color: str = "gray",
    edge_width: float = 1.0,
    edge_alpha: float = 0.7,
    colormap_name: str = "tab20",
) -> plt.Axes:
    """Visualiza nodos y aristas del grafo.

    Cada geometría en `gdf_lines` se colorea según su valor en la columna `Line`.

    Args:
        line_colormap: Optional mapping {line_name: color} para usar colores personalizados.
        colormap_name: Nombre de colormap matplotlib a usar si no se aporta `line_colormap`.
        show_legend: Si True, muestra una leyenda con las líneas.
    """

    # Preparar ejes
    _, ax = plt.subplots(figsize=figsize)

    proc_dir = processed_dir or os.path.join("src", "data", "processed", "graph")
    loaded = load_processed_gdfs(proc_dir)
    # load_processed_gdfs devuelve (gdf_nodes, gdf_lines) (posiblemente + edges_df)
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        gdf_nodes, gdf_lines = loaded[0], loaded[1]
    else:
        gdf_nodes, gdf_lines = loaded

    # Dibujar líneas (servicios) coloreadas por 'Line' si existe la columna
    if gdf_lines is not None and not gdf_lines.empty and "Line" in gdf_lines.columns:
        unique_lines = list(gdf_lines["Line"].astype(str).fillna("").unique())

        # # Generar mapping de colores si no se proporciona uno
        # if line_colormap is None:
        cmap = plt.get_cmap(colormap_name)
        n = len(unique_lines)
        if n <= 1:
            colors = [cmap(0)]
        else:
            colors = [cmap(i / max(1, n - 1)) for i in range(n)]
        import matplotlib.colors as mcolors
        color_hex = [mcolors.to_hex(c) for c in colors]
        line_colormap = dict(zip(unique_lines, color_hex))

        for ln in unique_lines:
            subset = gdf_lines[gdf_lines["Line"].astype(str) == ln]
            color = line_colormap.get(ln, edge_color)
            if not subset.empty:
                subset.plot(ax=ax, linewidth=edge_width, alpha=edge_alpha, color=color, zorder=1)

        # Leyenda
        # if show_legend:
        #     import matplotlib.patches as mpatches
        #     handles = [mpatches.Patch(color=line_colormap.get(ln, edge_color), label=str(ln)) for ln in unique_lines]
        #     ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.9)
    else:
        # Fallback: dibujar todas las líneas con el mismo color
        if gdf_lines is not None and not gdf_lines.empty:
            gdf_lines.plot(ax=ax, linewidth=edge_width, alpha=edge_alpha, color=edge_color, zorder=1)

    # Dibujar nodos encima
    if gdf_nodes is not None and not gdf_nodes.empty:
        gdf_nodes.plot(ax=ax, markersize=node_size, color=node_color, zorder=3)
        # Etiquetas opcionales
        if show_labels:
            for _, row in gdf_nodes.iterrows():
                if row.geometry is None:
                    continue
                x, y = row.geometry.x, row.geometry.y
                label = row.get("GTFS Stop ID") or row.get("node_id") or ""
                ax.text(x, y, str(label), fontsize=6, zorder=4)

        # Ajustar límites del plot según el bbox del grafo (gdf_nodes)
        xs = gdf_nodes.geometry.x.dropna()
        ys = gdf_nodes.geometry.y.dropna()
        if not xs.empty and not ys.empty:
            minx, maxx = float(xs.min()), float(xs.max())
            miny, maxy = float(ys.min()), float(ys.max())
            dx = maxx - minx
            dy = maxy - miny
            margin_x = dx * 0.01 if dx != 0 else 0.005
            margin_y = dy * 0.01 if dy != 0 else 0.005
            ax.set_xlim(minx - margin_x, maxx + margin_x)
            ax.set_ylim(miny - margin_y, maxy + margin_y)

    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# def visualize_topk_csv(topk_csv_path: str, processed_dir: Optional[str] = None, show_nodes: bool = True, figsize: Tuple[int, int] = (10, 10)) -> None:
#     """Convenience wrapper to visualize a top-k CSV produced by `src/routes/export_top_segments.py`.

#     This function simply forwards parameters to `visualize_segments_csv`, and
#     exists to make intent explicit when visualizing exported top-k files.
#     """
#     return visualize_segments_csv(topk_csv_path, processed_dir=processed_dir, show_nodes=show_nodes, figsize=figsize)


def visualize_segments_csv(
    csv_path: str,
    processed_dir: Optional[str] = None,
    show_nodes: bool = True, 
    figsize: Tuple[int, int] = (10, 10),
    # top_segments: Optional[int] = 10
) -> None:
    """Visualiza un CSV generado por `analyze_and_select_routes` sin re-ejecutar la búsqueda.

    Args:
        csv_path: Ruta al CSV con `geometry_wkt` producido por `analyze_and_select_routes`.
        processed_dir: Opcional carpeta `src/data/processed` para cargar líneas/nodos de fondo.
        show_nodes: Si True, muestra nodos (si están disponibles en processed_dir).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[route_analysis] CSV vacío: {csv_path}")
        return

    # reconstruir geometría
    if "geometry_wkt" not in df.columns:
        raise ValueError("CSV must contain a 'geometry_wkt' column to visualize")

    df["geometry"] = df["geometry_wkt"].apply(lambda s: wkt.loads(s) if pd.notna(s) and s != "" else None)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.reset_index(drop=True)

    # # Si se pide limitar el número de tramos a visualizar, muestre una muestra aleatoria
    # if top_segments is not None and top_segments > 0:
    #     if len(gdf) > top_segments:
    #         gdf = gdf.sample(n=top_segments).reset_index(drop=True)
    #     else:

    fig, ax = plt.subplots(figsize=figsize)
    # fondo: líneas procesadas si se dispone
    if processed_dir:
        try:
            gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
            if gdf_lines is not None and not gdf_lines.empty:
                gdf_lines.plot(ax=ax, linewidth=0.6, alpha=0.4, color="lightgray", zorder=1)
            if show_nodes and gdf_nodes is not None and not gdf_nodes.empty:
                gdf_nodes.plot(ax=ax, markersize=4, color="black", alpha=0.6, zorder=2)
        except Exception:
            pass

    # Colorear por fila (cada tramo) en lugar de agrupar por línea completa.
    n = len(gdf)
    if n > 0:
        import matplotlib.colors as mcolors

        if n <= 10:
            cmap = plt.get_cmap("tab10")
            palette_size = 10
        elif n <= 20:
            cmap = plt.get_cmap("tab20")
            palette_size = 20
        else:
            cmap = plt.get_cmap("hsv")
            palette_size = n

        colors = [mcolors.to_hex(cmap(i % max(1, palette_size - 1) / max(1, palette_size - 1))) for i in range(n)]

        # Dibujar cada fila por separado con su color
        for i in range(n):
            subset = gdf.iloc[[i]]
            color = colors[i]
            subset.plot(ax=ax, linewidth=2.5, color=color, zorder=3)
        
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()