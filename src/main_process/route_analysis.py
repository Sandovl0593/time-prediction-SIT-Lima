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
import math
from datetime import datetime

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import wkt

from src.data.load_nyc_mta import load_processed_gdfs


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _extract_segments_from_line(line_geom: LineString, line_geom_proj: LineString):
    """Devuelve listas de coordenadas lonlat y proyectadas para una línea.

    Retorna: (coords_lonlat, coords_proj)
    """
    coords_lonlat = list(line_geom.coords)
    coords_proj = list(line_geom_proj.coords)
    return coords_lonlat, coords_proj


def _cumulative_distances(coords: List[Tuple[float, float]]) -> List[float]:
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + _euclid(coords[i], coords[i - 1]))
    return cum


def _segment_geometry_from_coords(coords_lonlat: List[Tuple[float, float]], i: int, j: int) -> LineString:
    return LineString(coords_lonlat[i : j + 1])


def _collect_candidate_segments(gdf_lines: gpd.GeoDataFrame, max_target_m: float) -> List[dict]:
    """Recorre `gdf_lines` y genera candidatos (i,j) con métricas.

    `max_target_m` se usa como umbral para evitar generar segmentos excesivamente largos.
    """
    gdf_proj = gdf_lines.to_crs(epsg=3857)
    candidates = []
    for idx, row in gdf_lines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geom_proj = gdf_proj.loc[idx].geometry
        coords_lonlat, coords_proj = _extract_segments_from_line(geom, geom_proj)
        if len(coords_proj) < 2:
            continue
        cum = _cumulative_distances(coords_proj)
        n = len(coords_proj)
        # Generar todos los sub-tramos i<j (combinatoria, típicamente pequeña)
        for i in range(n - 1):
            for j in range(i + 1, n):
                path_len = cum[j] - cum[i]
                if path_len <= 0:
                    continue
                # opcional: descartar tramos mucho mayores que la mayor objetivo
                if path_len > max_target_m * 1.5:
                    continue
                chord = _euclid(coords_proj[j], coords_proj[i])
                straightness = chord / path_len if path_len > 0 else 0.0
                seg_geom = _segment_geometry_from_coords(coords_lonlat, i, j)
                candidates.append(
                    {
                        "line": row.get("Line", ""),
                        "line_idx": idx,
                        "start_idx": i,
                        "end_idx": j,
                        "path_length_m": float(path_len),
                        "chord_length_m": float(chord),
                        "straightness": float(straightness),
                        "geometry": seg_geom,
                    }
                )
    return candidates


def analyze_and_select_routes(
    processed_dir: str,
    target_km: Optional[float] = None,
    tolerance: float = 0.25,
    curvature_penalty: float = 0.5,
    out_base_dir: Optional[str] = "src/outputs",
    max_factor: float = 1.5,
) -> Dict[str, str]:
    """Ejecuta la búsqueda de tramos candidatos y exporta un CSV para un único objetivo.

    - `target_km` es un único valor (float). Se generan TODOS los tramos cuya
        longitud se encuentra dentro de `tolerance * target_km`.
    - Los CSV se guardan en un subdirectorio determinista de `out_base_dir`
        separado por la configuración (target, penalty, tolerance). Si el CSV
        ya existe para esa combinación, la función no re-ejecuta la búsqueda y
        retorna inmediatamente la ruta existente.

    Retorna:
            mapping: dict con claves: 'out_dir' y 'file' (ruta al CSV generado/existente)
    """
    if target_km is None:
        target_km = 1.0

    target_m = float(target_km) * 1000.0
    max_target_m = target_m

    # Preparar carpeta/archivo esperado por parámetros y evitar cómputo si ya existe
    def _sanitize_local(val: str) -> str:
        return str(val).replace(".", "p").replace(",", "_").replace(" ", "_")

    km_label = str(int(float(target_km))) if float(target_km).is_integer() else _sanitize_local(str(target_km))
    cfg_tag = f"target_{km_label}km_pen{_sanitize_local(str(curvature_penalty))}_tol{_sanitize_local(str(tolerance))}"
    out_dir = os.path.join(out_base_dir, cfg_tag)
    fname = os.path.join(out_dir, f"segments_{km_label}km.csv")

    # Si ya existe el CSV para esta configuración, no re-ejecutar el cómputo
    if os.path.exists(fname):
        try:
            existing = pd.read_csv(fname)
            print(f"[route_analysis] Found existing export for target {target_km} km at {fname} ({len(existing)} rows). Skipping computation.")
        except Exception:
            print(f"[route_analysis] Found existing file at {fname}. Skipping computation.")
        return {"out_dir": out_dir, "file": fname}

    # Cargar datos procesados
    gdf_nodes, gdf_lines = load_processed_gdfs(processed_dir)
    if gdf_lines is None or gdf_lines.empty:
        raise FileNotFoundError(f"No se encontraron líneas procesadas en {processed_dir}")

    print(f"[route_analysis] Generating segment candidates from {len(gdf_lines)} lines...")
    candidates = _collect_candidate_segments(gdf_lines, max_target_m=max_target_m)
    if not candidates:
        raise RuntimeError("No candidate segments generated from lines")

    print(f"[route_analysis] {len(candidates)} candidate segments generated")

    # Estimar segundos por metro usando edges.csv si está disponible
    edges_path = os.path.join(processed_dir, "edges.csv")
    seconds_per_m = None
    if os.path.exists(edges_path):
        try:
            edf = pd.read_csv(edges_path)
            valid = edf[edf["travel_time_s"].notna() & edf["length_m"].notna() & (edf["length_m"] > 0)]
            if not valid.empty:
                # tiempo (s) por metro
                seconds_per_m = (valid["travel_time_s"] / valid["length_m"]).median()
        except Exception:
            seconds_per_m = None

    # out_dir/fname ya se prepararon más arriba (evitamos re-ejecutar si existía)

    # Generar filas para el único target
    rows = []
    for c in candidates:
        # mantener solo tramos dentro de la tolerancia relativa al objetivo
        if abs(c["path_length_m"] - target_m) > (tolerance * target_m):
            continue
        length_dev = abs(c["path_length_m"] - target_m)
        score = c["straightness"] - curvature_penalty * (length_dev / (target_m + 1e-9))
        best_time = float(c["path_length_m"]) * float(seconds_per_m) if seconds_per_m is not None else None
        rows.append(
            {
                "line": c.get("line", ""),
                "line_idx": c.get("line_idx"),
                "start_idx": c.get("start_idx"),
                "end_idx": c.get("end_idx"),
                "path_length_m": c.get("path_length_m"),
                "path_length_km": c.get("path_length_m") / 1000.0,
                "chord_length_m": c.get("chord_length_m"),
                "straightness": c.get("straightness"),
                "score": float(score),
                "best_time_s": best_time,
                "geometry_wkt": c.get("geometry").wkt if c.get("geometry") is not None else "",
            }
        )

    df_out = pd.DataFrame(rows)
    if not df_out.empty and "score" in df_out.columns:
        df_out = df_out.sort_values("score", ascending=False)

    # Asegurar carpeta de salida y escribir CSV
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(fname, index=False)
    print(f"[route_analysis] Exported {len(df_out)} segments for target {target_km} km -> {fname}")

    return {"out_dir": out_dir, "file": fname}


def visualize_segments_csv(csv_path: str, processed_dir: Optional[str] = None, show_nodes: bool = True, figsize: Tuple[int, int] = (10, 10)) -> None:
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

    # Preferir colorear por `line` (ruta). Si no existe, usar `target_km`,
    # y en último caso pintar todo de rojo.
    plotted = False
    # if "line" in gdf.columns:
    uniques = list(gdf["line"].dropna().unique())
    n = len(uniques)
    # Elegir un mapa con suficientes colores; tab20 cubre hasta 20 rutas.
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        palette_size = 10
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        palette_size = 20
    else:
        cmap = plt.get_cmap("hsv")
        palette_size = n

    mapping = {u: cmap(i % palette_size / max(1, palette_size - 1)) for i, u in enumerate(uniques)}
    for u, color in mapping.items():
        subset = gdf[gdf["line"] == u]
        if not subset.empty:
            subset.plot(ax=ax, linewidth=2.5, color=color, zorder=3, label=str(u))
            plotted = True
    if plotted:
        ax.legend(title="line", fontsize="small", markerscale=2, loc="best")
    # elif "target_km" in gdf.columns:
    #     uniques = sorted(gdf["target_km"].dropna().unique())
    #     cmap = plt.get_cmap("tab10")
    #     mapping = {u: cmap(i % 10) for i, u in enumerate(uniques)}
    #     for u, color in mapping.items():
    #         subset = gdf[gdf["target_km"] == u]
    #         if not subset.empty:
    #             subset.plot(ax=ax, linewidth=2.5, color=color, zorder=3)
    # else:
    #     gdf.plot(ax=ax, linewidth=2.5, color="red", zorder=3)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
