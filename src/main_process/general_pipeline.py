"""Carga y prepara datos de estaciones MTA (NYC) separando la parte espacial
de la estructural.

Flujo:
1. GeoPandas: crea geometrías puntuales y líneas por servicio, proyecta para
   obtener distancias métricas y extrae variables espaciales fiables.
2. NetworkX: construye el grafo (nodos + aristas) a partir de la información
   espacial, asignando atributos a nodos y aristas pensados para un encoder
   GAT y un decoder MLP.

La función devuelve un diccionario con el grafo (`networkx.Graph`), los
GeoDataFrames de nodos y líneas y metadatos básicos.
"""

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import os
import re
from collections import defaultdict
import logging

# configurar logger simple para información durante el preprocesado
logger = logging.getLogger("data_cleaning")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[cleaning] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely import wkt
from .cleaning import (
    ensure_processed_dirs,
    clean_stations,
    clean_stop_times,
    clean_trips,
    write_manifest,
    write_quality_report
)

def _rustic_1d_projection(coords: np.ndarray) -> np.ndarray:
    """Proyecta coordenadas 2D a valores 1D de forma rústica y rápida.

    Estrategia: comparar la varianza en X e Y y usar la coordenada dominante
    como valor de proyección. Es rápido (O(n)) y suficiente para trayectos
    mayoritariamente lineales; la mejora local 2-opt posterior reduce zigzags.
    """
    if coords.size == 0:
        return np.array([])
    x = coords[:, 0]
    y = coords[:, 1]
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    # elegir la coordenada con mayor varianza
    return x if var_x >= var_y else y


def _order_points_via_rustic_and_2opt(
    coords: np.ndarray,
    max_iter: int = 3
) -> np.ndarray:
    """Ordena índices de puntos 2D usando una proyección rústica + 2-opt.

    Args:
        coords: array (N,2) con coordenadas proyectadas (métricas).
        max_iter: número máximo de iteraciones de mejora 2-opt.

    Returns:
        array de índices (longitud N) que indica el nuevo orden.
    """
    n = coords.shape[0]
    if n <= 2:
        return np.arange(n)

    # Inicializar con proyección rústica (coordenada dominante)
    proj = _rustic_1d_projection(coords)
    asc = np.argsort(proj)
    desc = asc[::-1]

    def path_length(order_idx: np.ndarray) -> float:
        pts = coords[order_idx]
        dif = np.diff(pts, axis=0)
        return float(np.sum(np.sqrt((dif ** 2).sum(axis=1))))

    # Escoger la orientación (asc/desc) con menor longitud inicial
    best = asc
    best_len = path_length(asc)
    rev_len = path_length(desc)
    if rev_len < best_len:
        best = desc
        best_len = rev_len

    order = list(best)

    # Mejora local 2-opt (open path)
    it = 0
    improved = True
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                # Consider edges (i,i+1) and (j,j+1)
                a = coords[order[i]]
                b = coords[order[i + 1]]
                c = coords[order[j]]
                d = coords[order[j + 1]]
                cur = np.linalg.norm(a - b) + np.linalg.norm(c - d)
                new = np.linalg.norm(a - c) + np.linalg.norm(b - d)
                if new + 1e-9 < cur:
                    # Revertir segmento (i+1 .. j) inclusive
                    order[i + 1 : j + 1] = list(reversed(order[i + 1 : j + 1]))
                    improved = True
                    break
            if improved:
                break

    return np.array(order, dtype=int)


# La implementación anterior basada en PCA fue eliminada; usar
# _order_points_via_rustic_and_2opt junto con _rustic_1d_projection.

def _load_nodes_from_csv(path: str) -> gpd.GeoDataFrame:
    print(f"[gen_pipeline] Reading stations CSV from {path}")
    df = pd.read_csv(path)
    # Eliminar completamente las columnas 'CBD' y 'Borough' por petición del usuario.
    for c in ["CBD", "Borough"]:
        if c in df.columns:
            print(f"[gen_pipeline] Dropping column '{c}' as requested by user")
            df = df.drop(columns=[c])
    stop_id_col = "GTFS Stop ID"
    lat_col = "GTFS Latitude"
    lon_col = "GTFS Longitude"

    gdf_nodes = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
    )

    print("[gen_pipeline] Projecting to metric CRS for distance calculations")
    gdf_proj = gdf_nodes.to_crs(epsg=3857)
    gdf_nodes["_x"] = gdf_proj.geometry.x
    gdf_nodes["_y"] = gdf_proj.geometry.y

    # 'CBD' and 'Borough' are intentionally discarded; no parsing required.

    return gdf_nodes


def _build_lines_from_nodes(gdf_nodes: gpd.GeoDataFrame):
    print("[gen_pipeline] Building LineString geometries per 'Line' (assume header exists)")
    lines = []
    line_orders = []  # list of (line_name, ordered_gdf)
    for line_name, sub in gdf_nodes.groupby("Line"):
        sub = sub.dropna(subset=["_x", "_y"]) if len(sub) > 0 else sub
        if len(sub) < 2:
            continue
        coords = np.vstack([sub["_x"].values, sub["_y"].values]).T

        # Orden inicial rústico + refinamiento 2-opt para evitar zigzags
        try:
            order = _order_points_via_rustic_and_2opt(coords, max_iter=3)
            ordered = sub.iloc[order]
        except Exception:
            # Fallback simple si algo falla: ordenar por coordenada dominante
            proj = _rustic_1d_projection(coords)
            order = np.argsort(proj)
            ordered = sub.iloc[order]
        line_orders.append((line_name, ordered))

        # crear LineString en WGS84 (lon, lat)
        line_coords = list(zip(ordered["GTFS Longitude"].values, ordered["GTFS Latitude"].values))
        line_geom = LineString(line_coords)
        lines.append({"Line": line_name, "num_stations": len(ordered), "geometry": line_geom})

    gdf_lines = gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326")
    return gdf_lines, line_orders


def _build_graph_from_nodes_and_lines(gdf_nodes: gpd.GeoDataFrame, line_orders):
    print("[gen_pipeline] Building NetworkX MultiDiGraph with node attributes")
    G = nx.MultiDiGraph()

    node_meta = {}
    for _, row in gdf_nodes.iterrows():
        node_id = str(row["GTFS Stop ID"])
        attrs = {
            "lon": float(row["GTFS Longitude"]) if pd.notna(row["GTFS Longitude"]) else None,
            "lat": float(row["GTFS Latitude"]) if pd.notna(row["GTFS Latitude"]) else None,
            "x": float(row["_x"]),
            "y": float(row["_y"]),
            # 'Borough' and 'CBD' fueron descartados antes de la ingestión
            # y por tanto no se incluyen como atributos de nodo.
            "line": row.get("Line"),
            "structure": row.get("Structure", "unknown"),
            "geometry": row.geometry,
        }
        G.add_node(node_id, **attrs)
        node_meta[node_id] = attrs

    print("[gen_pipeline] Adding spatial edges between consecutive stations per line")
    for line_name, ordered in line_orders:
        prev_id = None
        prev_row = None
        for _, row in ordered.iterrows():
            curr_id = str(row["GTFS Stop ID"])
            if prev_id is None:
                prev_id = curr_id
                prev_row = row
                continue
            dx = row["_x"] - prev_row["_x"]
            dy = row["_y"] - prev_row["_y"]
            dist_m = float(np.sqrt(dx * dx + dy * dy))
            geom_ab = LineString([
                (prev_row["GTFS Longitude"], prev_row["GTFS Latitude"]),
                (row["GTFS Longitude"], row["GTFS Latitude"]),
            ])
            geom_ba = LineString([
                (row["GTFS Longitude"], row["GTFS Latitude"]),
                (prev_row["GTFS Longitude"], prev_row["GTFS Latitude"]),
            ])
            G.add_edge(prev_id, curr_id, line=line_name, length_m=dist_m, geometry=geom_ab, spatial=True)
            G.add_edge(curr_id, prev_id, line=line_name, length_m=dist_m, geometry=geom_ba, spatial=True)
            prev_id = curr_id
            prev_row = row

    return G, node_meta


def _process_stop_times(
    stop_times_path: str, 
    G: nx.MultiDiGraph,
    node_meta: dict,
    gdf_nodes: gpd.GeoDataFrame,
    trips_path: Optional[str] = None
):
    if stop_times_path is None:
        print("[gen_pipeline] No stop_times file provided; skipping observed travel-time processing")
        return

    if not os.path.exists(stop_times_path):
        print(f"[gen_pipeline] stop_times file not found at {stop_times_path}; skipping")
        return

    print(f"[gen_pipeline] Processing stop_times from {stop_times_path}")
    st = pd.read_csv(stop_times_path, dtype={"stop_id": str})
    st_cols = {c.lower(): c for c in st.columns}
    trip_col = st_cols.get("trip_uid") or st_cols.get("trip_id")
    stop_col = st_cols.get("stop_id")
    arr_col = st_cols.get("arrival_time") or st_cols.get("arrival")
    dep_col = st_cols.get("departure_time") or st_cols.get("departure")
    last_obs_col = st_cols.get("last_observed")
    mark_col = st_cols.get("marked_past")

    if trip_col is None or stop_col is None:
        raise ValueError("stop_times file must contain a trip id column and a stop_id column")

    for c in [arr_col, dep_col, last_obs_col, mark_col]:
        if c and c in st.columns:
            st[c] = pd.to_numeric(st[c], errors="coerce")

    station_ids = set(gdf_nodes["GTFS Stop ID"].astype(str).values)

    def map_stop_safe(s: str) -> Optional[str]:
        if pd.isna(s):
            return None
        s = str(s).strip()
        if s in station_ids:
            return s
        m = re.match(r"^(\d+)[A-Za-z]+$", s)
        if m:
            base = m.group(1)
            if base in station_ids:
                return base
        return None

    trips_map = {}
    if trips_path and os.path.exists(trips_path):
        print(f"[gen_pipeline] Reading trips file from {trips_path}")
        trips_df = pd.read_csv(trips_path, dtype={})
        tcols = {c.lower(): c for c in trips_df.columns}
        t_trip_col = tcols.get("trip_uid") or tcols.get("trip_id")
        route_col = tcols.get("route_id")
        if t_trip_col and route_col:
            trips_map = dict(zip(trips_df[t_trip_col].astype(str), trips_df[route_col].astype(str)))

    if arr_col and arr_col in st.columns:
        st["_order_time"] = st[arr_col]
    elif dep_col and dep_col in st.columns:
        st["_order_time"] = st[dep_col]
    else:
        st["_order_time"] = np.nan

    st_order = st[st["_order_time"].notna()].copy()
    print(f"[gen_pipeline] Found {len(st_order)} stop_time records with valid times for processing")

    travel_acc = defaultdict(list)

    for trip_id, trip_group in st_order.groupby(trip_col):
        trip_group = trip_group.sort_values("_order_time")
        route_id_val = trips_map.get(str(trip_id)) if trips_map else None
        prev = None
        prev_times = None
        for _, r in trip_group.iterrows():
            curr_stop = map_stop_safe(r[stop_col])
            if curr_stop is None:
                prev = None
                prev_times = None
                continue
            curr_arr = r[arr_col] if arr_col in r.index else np.nan
            curr_dep = r[dep_col] if dep_col in r.index else np.nan
            curr_last = r[last_obs_col] if (last_obs_col and last_obs_col in r.index) else np.nan

            if prev is not None and prev_times is not None:
                prev_dep = prev_times.get("dep")
                prev_arr = prev_times.get("arr")
                start_time = prev_dep if prev_dep is not None and not pd.isna(prev_dep) else prev_arr
                end_time = curr_arr if not pd.isna(curr_arr) else (curr_dep if not pd.isna(curr_dep) else (curr_last if not pd.isna(curr_last) else np.nan))
                if pd.notna(start_time) and pd.notna(end_time):
                    delta = float(end_time - start_time)
                    if delta > 0:
                        key = (prev, curr_stop, route_id_val)
                        travel_acc[key].append(delta)

            prev = curr_stop
            prev_times = {"arr": curr_arr if not pd.isna(curr_arr) else None, "dep": curr_dep if not pd.isna(curr_dep) else None}
    print(f"[gen_pipeline] Processed stop_times for {len(travel_acc)} observed edges")

    for (u, v, route_id_val), deltas in travel_acc.items():
        if not deltas:
            continue
        avg = float(np.mean(deltas))
        u_meta = node_meta.get(u)
        v_meta = node_meta.get(v)
        if u_meta and v_meta:
            dx = u_meta["x"] - v_meta["x"]
            dy = u_meta["y"] - v_meta["y"]
            dist_m = float(np.sqrt(dx * dx + dy * dy))
            geom = LineString([(u_meta["lon"], u_meta["lat"]), (v_meta["lon"], v_meta["lat"])])
            G.add_edge(u, v, route_id=route_id_val, travel_time_s=avg, length_m=dist_m, geometry=geom, observed=True)
    print(f"[gen_pipeline] Added {len(travel_acc)} observed edges to graph")


def save_processed_graph(
    G: nx.MultiDiGraph,
    gdf_nodes: gpd.GeoDataFrame,
    gdf_lines: gpd.GeoDataFrame,
    out_dir: str
):
    """Guarda nodos, aristas y líneas procesadas en CSV dentro de `out_dir`.

    - nodes.csv: contiene atributos de nodos (incluye lon/lat y geometry_wkt)
    - edges.csv: contiene aristas con atributos (u,v,key, atributos..., geometry_wkt)
    - lines.csv: contiene geometrías de línea por servicio (Line, num_stations, geometry_wkt)
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[save_processed_graph] Saving processed files to {out_dir}")

    # nodes
    nodes_df = gdf_nodes.copy()
    if "geometry" in nodes_df.columns:
        nodes_df["geometry_wkt"] = nodes_df.geometry.apply(lambda g: g.wkt if g is not None else "")
    nodes_out = os.path.join(out_dir, "nodes.csv")
    nodes_df.to_csv(nodes_out, index=False)

    # edges
    edges_list = []
    if G.is_multigraph():
        for u, v, k, edata in G.edges(keys=True, data=True):
            rec = {"u": u, "v": v, "key": k}
            for ak, av in edata.items():
                if ak == "geometry":
                    rec["geometry_wkt"] = av.wkt if av is not None else ""
                else:
                    rec[ak] = av
            edges_list.append(rec)
    else:
        for u, v, edata in G.edges(data=True):
            rec = {"u": u, "v": v}
            for ak, av in edata.items():
                if ak == "geometry":
                    rec["geometry_wkt"] = av.wkt if av is not None else ""
                else:
                    rec[ak] = av
            edges_list.append(rec)

    edges_out = os.path.join(out_dir, "edges.csv")
    edges_df = pd.DataFrame(edges_list)
    edges_df.to_csv(edges_out, index=False)
    
    # lines
    lines_df = gdf_lines.copy()
    if "geometry" in lines_df.columns:
        lines_df["geometry_wkt"] = lines_df.geometry.apply(lambda g: g.wkt if g is not None else "")
    lines_out = os.path.join(out_dir, "lines.csv")
    lines_df.to_csv(lines_out, index=False)


def load_processed_gdfs(processed_dir: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Optional[pd.DataFrame]]:
    """Carga `nodes.csv`, `lines.csv` y `edges.csv` desde `processed_dir` y devuelve GeoDataFrames.

    Retorna: (gdf_nodes, gdf_lines, edges_df)
    """
    nodes_path = os.path.join(processed_dir, "nodes.csv")
    lines_path = os.path.join(processed_dir, "lines.csv")
    edges_path = os.path.join(processed_dir, "edges.csv")

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"nodes.csv not found in {processed_dir}")

    nodes_df = pd.read_csv(nodes_path)
    # reconstruir geometría
    if "geometry_wkt" in nodes_df.columns:
        nodes_df["geometry"] = nodes_df["geometry_wkt"].apply(lambda s: wkt.loads(s) if pd.notna(s) and s != "" else None)
    else:
        # intentar con lon/lat
        if "GTFS Longitude" in nodes_df.columns and "GTFS Latitude" in nodes_df.columns:
            nodes_df["geometry"] = gpd.points_from_xy(nodes_df["GTFS Longitude"], nodes_df["GTFS Latitude"])
        else:
            nodes_df["geometry"] = None

    gdf_nodes = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs="EPSG:4326")

    if os.path.exists(lines_path):
        lines_df = pd.read_csv(lines_path)
        if "geometry_wkt" in lines_df.columns:
            lines_df["geometry"] = lines_df["geometry_wkt"].apply(lambda s: wkt.loads(s) if pd.notna(s) and s != "" else None)
        gdf_lines = gpd.GeoDataFrame(lines_df, geometry="geometry", crs="EPSG:4326")
    else:
        gdf_lines = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # edges_df = pd.read_csv(edges_path) if os.path.exists(edges_path) else None
    
    G = nx.MultiDiGraph()
    # load nodes
    for _, row in gdf_nodes.iterrows():
        node_id = str(row["GTFS Stop ID"])
        attrs = {k: row[k] for k in row.index if k not in {"GTFS Stop ID", "geometry", "geometry_wkt"}}
        G.add_node(node_id, **attrs)

    edges_df = pd.read_csv(edges_path)
    for _, row in edges_df.iterrows():
        u = row["u"]
        v = row["v"]
        key = row.get("key", 0)
        edata = {k: v for k, v in row.items() if k not in {"u", "v", "key", "geometry_wkt"}}
        if "geometry_wkt" in row and pd.notna(row["geometry_wkt"]) and row["geometry_wkt"] != "":
            edata["geometry"] = wkt.loads(row["geometry_wkt"])
        G.add_edge(u, v, key=key, **edata)
    
    return gdf_nodes, gdf_lines


def compute_and_save_metrics(
    G: nx.MultiDiGraph,
    gdf_lines: gpd.GeoDataFrame,
    scenario_id: str = "default",
    target_km: Optional[float] = None,
    curve_penalty: Optional[float] = None,
    tolerance: Optional[float] = None
) -> Dict[str, Any]:
    """Computa métricas simples de preprocesado y guarda CSVs resumidos."""
    metrics = {
        "num_nodes": int(G.number_of_nodes()) if G is not None else 0,
        "num_edges": int(G.number_of_edges()) if G is not None else 0,
        "processing_date": pd.Timestamp.now().isoformat(),
    }

    # Generar métricas de rutas por escenario (sin particionado por día)
    outputs_root = Path("src") / "outputs"
    out_dir = outputs_root / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)

    route_metrics_rows = []

    # Si tenemos líneas procesadas, generar candidatos y calcular métricas por segmento
    if gdf_lines is not None and not gdf_lines.empty:
        gdf_proj = gdf_lines.to_crs(epsg=3857)
        for idx, row in gdf_lines.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geom_proj = gdf_proj.loc[idx].geometry
            coords_lonlat = list(geom.coords)
            coords_proj = list(geom_proj.coords)
            if len(coords_proj) < 2:
                continue
            # cumulative distances using projected coords
            cum = [0.0]
            for i in range(1, len(coords_proj)):
                a = coords_proj[i-1]
                b = coords_proj[i]
                dist = float(np.hypot(b[0]-a[0], b[1]-a[1]))
                cum.append(cum[-1] + dist)
            n = len(coords_proj)
            # generar subtramos i<j
            for i in range(n-1):
                for j in range(i+1, n):
                    path_len = cum[j] - cum[i]
                    if path_len <= 0:
                        continue
                    chord = float(np.hypot(coords_proj[j][0]-coords_proj[i][0], coords_proj[j][1]-coords_proj[i][1]))
                    # convertir a km
                    length_real_km = path_len / 1000.0
                    length_straight_km = chord / 1000.0 if chord > 0 else 0.0
                    straightness_index = length_real_km / length_straight_km if length_straight_km > 0 else None
                    # si target_km proporcionado, filtrar por tolerancia
                    accept_by_length = True
                    abs_error_km = None
                    relative_error = None
                    accept_within_tol = None
                    if target_km is not None and tolerance is not None:
                        abs_error_km = abs(length_real_km - float(target_km))
                        relative_error = abs_error_km / float(target_km) if float(target_km) != 0 else None
                        accept_within_tol = (relative_error is not None and relative_error <= float(tolerance))
                        accept_by_length = accept_within_tol

                    straightness_c = (chord / path_len) if path_len > 0 else 0.0
                    score = straightness_c - (float(curve_penalty) * (abs(length_real_km - float(target_km)) / (float(target_km) + 1e-9))) if target_km is not None and curve_penalty is not None else straightness_c
                    
                    seg_geom = LineString(coords_lonlat[i:j+1])
                    route_metrics_rows.append({
                        "line": row.get("Line", ""),
                        "line_idx": idx,
                        "start_idx": i,
                        "end_idx": j,
                        "length_real_km": length_real_km,
                        "length_straight_km": length_straight_km,
                        "straightness_index": straightness_index,
                        "abs_error_km": abs_error_km,
                        "relative_error": relative_error,
                        "acceptance_within_tolerance": bool(accept_within_tol) if accept_within_tol is not None else None,
                        "score": float(score) if score is not None else None,
                        "geometry_wkt": seg_geom.wkt,
                    })

    df_route_metrics = pd.DataFrame(route_metrics_rows)
    # Si se especificó target_km y tolerance, filtrar filas aceptadas
    if not df_route_metrics.empty and target_km is not None and tolerance is not None:
        df_route_metrics["accepted"] = df_route_metrics["acceptance_within_tolerance"].fillna(False)
    else:
        df_route_metrics["accepted"] = False

    # ordenar y rankear por cercanía al objetivo si existe
    if not df_route_metrics.empty and "abs_error_km" in df_route_metrics.columns:
        df_route_metrics["rank_abs_error"] = df_route_metrics["abs_error_km"].rank(method="min")

    # Guardar route_metrics_summary en outputs/<scenario_tag>
    route_metrics_path = out_dir / f"route.csv"
    df_route_metrics.to_csv(route_metrics_path, index=False)

    # Actualizar índice general de escenarios
    scenario_record = {
        "scenario_id": scenario_id,
        "target_km": target_km,
        "curve_penalty": curve_penalty,
        "tolerance": tolerance,
        "num_candidates": len(df_route_metrics),
        "num_accepted": int(df_route_metrics["accepted"].sum()) if not df_route_metrics.empty else 0,
        "route_metrics_path": str(route_metrics_path),
    }

    metrics.update({
        "route_metrics_path": str(route_metrics_path),
        "route_count_by_scenario": int(scenario_record["num_candidates"]),
        "accepted_count": int(scenario_record["num_accepted"]),
    })

    return metrics



def general_pipeline(
    path: str,
    stop_times_path: Optional[str] = None,
    trips_path: Optional[str] = None,
    scenario_id: str = "default",
    target_km: Optional[float] = None,
    curve_penalty: Optional[float] = None,
    tolerance: Optional[float] = None,
    cleaning_thresholds: Optional[Dict[str, Any]] = None,
):
    """Carga CSV de estaciones (GTFS-like) y construye GeoDataFrames y grafo.

    Args:
        path: Ruta al CSV de estaciones. Debe contener columnas al menos:
            'GTFS Stop ID', 'GTFS Latitude', 'GTFS Longitude'. Opcionalmente
            'Line', 'Complex ID', 'Daytime Routes', 'Borough', 'CBD'.
    """
 
    print(f"[gen_pipeline] Starting load for {path}")

    # Preparar directorios procesados
    processed_root = Path("src") / "data" / "processed"
    dirs = ensure_processed_dirs(processed_root)

    # 0) etapa de limpieza de estaciones (produce cleaned_stations.csv)
    cleaned_dir = dirs["cleaned"]
    graph_dir = dirs["graph"]

    cleaned_stations_path = cleaned_dir / "cleaned_stations.csv"
    cleaned_stop_times_path = cleaned_dir / "cleaned_stop_times.csv"
    cleaned_trips_path = cleaned_dir / "cleaned_trip_times.csv"

    reports = []

    # Si ya existen los CSV limpios, evitar re-ejecución de limpieza
    if cleaned_stations_path.exists():
        stations_csv_path = str(cleaned_stations_path)
        stations_report = {"file": cleaned_stations_path.name, "note": "skipped_cleaning_exists", "out_path": str(cleaned_stations_path)}
        print(f"[gen_pipeline] Found existing cleaned stations at {cleaned_stations_path}; skipping cleaning")
        reports.append(stations_report)
    else:
        try:
            cleaned_stations_path, stations_report = clean_stations(Path(path), processed_root, thresholds=cleaning_thresholds)
            stations_csv_path = str(cleaned_stations_path)
            print(f"[gen_pipeline] Cleaned stations written to {cleaned_stations_path}")
            reports.append(stations_report)
        except Exception as e:
            print(f"[gen_pipeline] Warning: failed cleaning stations: {e}; falling back to raw file")
            stations_report = {"file": Path(path).name, "original_rows": None}
            stations_csv_path = path
            reports.append(stations_report)

    # 1) read stations and build nodes GeoDataFrame (desde el CSV limpio)
    # Si ya existe grafo procesado completo en graph/, cargarlo en lugar de reconstruir
    graph_nodes_csv = graph_dir / "nodes.csv"
    graph_edges_csv = graph_dir / "edges.csv"
    graph_lines_csv = graph_dir / "lines.csv"

    if graph_nodes_csv.exists() and graph_edges_csv.exists() and graph_lines_csv.exists():
        print(f"[gen_pipeline] Found existing processed graph in {graph_dir}; loading instead of rebuilding")
        # load_processed_gdfs devuelve gdf_nodes, gdf_lines
        gdf_nodes, gdf_lines = load_processed_gdfs(str(graph_dir))
        # Reconstruir G a partir de nodes/edges CSVs
        print("[gen_pipeline] Reconstructing NetworkX graph from CSVs")
        G = nx.MultiDiGraph()
        node_meta = {}
        for _, row in gdf_nodes.iterrows():
            node_id = str(row["GTFS Stop ID"])
            attrs = {k: row[k] for k in row.index if k not in {"GTFS Stop ID", "geometry", "geometry_wkt"}}
            # Normalizar nombres de coordenadas para mantener compatibilidad
            # con la estructura esperada por el resto del código (x,y,lon,lat)
            if "_x" in row.index and pd.notna(row.get("_x")):
                try:
                    attrs["x"] = float(row.get("_x"))
                except Exception:
                    pass
            elif "x" in row.index and pd.notna(row.get("x")):
                try:
                    attrs["x"] = float(row.get("x"))
                except Exception:
                    pass
            if "_y" in row.index and pd.notna(row.get("_y")):
                try:
                    attrs["y"] = float(row.get("_y"))
                except Exception:
                    pass
            elif "y" in row.index and pd.notna(row.get("y")):
                try:
                    attrs["y"] = float(row.get("y"))
                except Exception:
                    pass
            if "GTFS Longitude" in row.index and pd.notna(row.get("GTFS Longitude")):
                try:
                    attrs["lon"] = float(row.get("GTFS Longitude"))
                except Exception:
                    pass
            if "GTFS Latitude" in row.index and pd.notna(row.get("GTFS Latitude")):
                try:
                    attrs["lat"] = float(row.get("GTFS Latitude"))
                except Exception:
                    pass
            # preservar geometría si está presente
            if "geometry" in row.index and row["geometry"] is not None:
                attrs["geometry"] = row["geometry"]
            G.add_node(node_id, **attrs)
            node_meta[node_id] = attrs

        # Cargar edges.csv manualmente
        try:
            edf = pd.read_csv(str(graph_edges_csv))
            for _, r in edf.iterrows():
                u = r["u"]
                v = r["v"]
                key = r.get("key", None)
                edata = {k: v for k, v in r.items() if k not in {"u", "v", "key", "geometry_wkt"}}
                if "geometry_wkt" in r and pd.notna(r["geometry_wkt"]) and r["geometry_wkt"] != "":
                    try:
                        edata["geometry"] = wkt.loads(r["geometry_wkt"])
                    except Exception:
                        pass
                if key is not None and not pd.isna(key):
                    try:
                        G.add_edge(u, v, key=key, **edata)
                    except Exception:
                        G.add_edge(u, v, **edata)
                else:
                    G.add_edge(u, v, **edata)
        except Exception as e:
            print(f"[gen_pipeline] Warning: failed reading edges.csv: {e}; graph may be incomplete")

    else:
        gdf_nodes = _load_nodes_from_csv(stations_csv_path)

        # 2) build line geometries and ordering
        gdf_lines, line_orders = _build_lines_from_nodes(gdf_nodes)

        # 3) build structural graph (nodes + spatial edges)
        G, node_meta = _build_graph_from_nodes_and_lines(gdf_nodes, line_orders)

    # --- Si hay stop_times disponibles, limpiar y calcular tiempo promedio por arista ---
    # stop_times: debe contener columnas como 'trip_uid','stop_id','arrival_time','departure_time'
    # localizar stop_times/trips si no fueron provistos explícitamente
    base_dir = os.path.dirname(path)
    if stop_times_path is None:
        candidates = [f for f in os.listdir(base_dir) if "stop_times" in f]
        stop_times_path = os.path.join(base_dir, candidates[0]) if candidates else None

    if trips_path is None:
        candidates = [f for f in os.listdir(base_dir) if "trip" in f and f.endswith(".csv")]
        trips_path = os.path.join(base_dir, candidates[0]) if candidates else None

    # 4) limpiar trips y stop_times (si existen) antes de procesar
    # Usar archivos limpios existentes si están presentes para evitar re-ejecuciones
    trips_map = {}

    # Primero trips
    # Determine trips cleaned file: prefer existing cleaned file in processed, else clean raw if available
    if cleaned_trips_path.exists():
        cleaned_trips_path = cleaned_trips_path
        try:
            trips_df = pd.read_csv(str(cleaned_trips_path), dtype=str)
            tcols = {c.lower(): c for c in trips_df.columns}
            t_trip_col = tcols.get("trip_uid") or tcols.get("trip_id")
            route_col = tcols.get("route_id")
            if t_trip_col and route_col:
                trips_map = dict(zip(trips_df[t_trip_col].astype(str), trips_df[route_col].astype(str)))
            trips_report = {"file": cleaned_trips_path.name, "note": "skipped_cleaning_exists", "out_path": str(cleaned_trips_path)}
            reports.append(trips_report)
            print(f"[gen_pipeline] Found existing cleaned trips at {cleaned_trips_path}; skipping cleaning")
        except Exception:
            trips_map = {}
    else:
        if trips_path and os.path.exists(trips_path):
            try:
                cleaned_trips_path, trips_report, trips_map = clean_trips(Path(trips_path), processed_root)
                reports.append(trips_report)
                print(f"[gen_pipeline] Cleaned trips written to {cleaned_trips_path}")
            except Exception as e:
                print(f"[gen_pipeline] Warning: failed cleaning trips: {e}; using raw file")

    # Luego stop_times: preferir archivo limpio existente si está presente
    if cleaned_stop_times_path.exists():
        stop_times_to_process = str(cleaned_stop_times_path)
        stop_report = {"file": cleaned_stop_times_path.name, "note": "skipped_cleaning_exists", "out_path": str(cleaned_stop_times_path)}
        reports.append(stop_report)
        print(f"[gen_pipeline] Found existing cleaned stop_times at {cleaned_stop_times_path}; skipping cleaning")
    else:
        if stop_times_path and os.path.exists(stop_times_path):
            try:
                station_ids = set(gdf_nodes["GTFS Stop ID"].astype(str).values)
                cleaned_stop_times_path, stop_report = clean_stop_times(Path(stop_times_path), processed_root, station_ids, thresholds=cleaning_thresholds, trips_map=trips_map if trips_map else None)
                reports.append(stop_report)
                print(f"[gen_pipeline] Cleaned stop_times written to {cleaned_stop_times_path}")
                stop_times_to_process = str(cleaned_stop_times_path)
            except Exception as e:
                print(f"[gen_pipeline] Warning: failed cleaning stop_times: {e}; using raw file")
                stop_times_to_process = stop_times_path
        else:
            stop_times_to_process = stop_times_path

    # 5) procesar stop_times (si existe) usando el CSV limpio cuando esté disponible
    _process_stop_times(stop_times_to_process, G, node_meta, gdf_nodes, trips_path=cleaned_trips_path or trips_path)

    metadata = {"num_nodes": G.number_of_nodes(), "num_edges": G.number_of_edges()}

    print(f"[gen_pipeline] Finished. Nodes: {metadata['num_nodes']}, Edges: {metadata['num_edges']}")

    # guardar procesado en carpeta 'processed' junto al raw
    # Guardar grafo procesado en la subcarpeta 'graph' dentro de processed
    graph_dir = dirs["graph"]
    try:
        save_processed_graph(G, gdf_nodes, gdf_lines, str(graph_dir))
        print(f"[gen_pipeline] Saved processed CSVs to {graph_dir}")
    except Exception as e:
        print(f"[gen_pipeline] Warning: failed saving processed files: {e}")
    # Graph CSVs are saved only in the `graph/` subdirectory to avoid duplication

    # Guardar manifest y reporte de calidad (aggregado)
    try:
        manifest = {
            "source_files": [r.get("file") for r in reports],
            "processing_date": pd.Timestamp.now().isoformat(),
            "crs": "EPSG:4326",
            "rows": {r.get("file"): {"original": r.get("original_rows"), "kept": r.get("kept_rows")} for r in reports},
            "date_ranges": {r.get("file"): r.get("date_range") for r in reports},
            "cleaning_thresholds": {r.get("file"): r.get("cleaning_thresholds") for r in reports},
        }
        write_manifest(manifest, dirs["manifests"])
        write_quality_report({r.get("file"): r for r in reports}, dirs["metrics"])
    except Exception as e:
        print(f"[gen_pipeline] Warning: failed writing manifest/quality report: {e}")

    # Calcular métricas de preprocesado y resumen por ruta
    try:
        metrics = compute_and_save_metrics(
            G,
            gdf_lines,
            scenario_id=scenario_id,
            target_km=target_km,
            curve_penalty=curve_penalty,
            tolerance=tolerance,
        )
        print(f"[gen_pipeline] Computed preprocessing metrics: {metrics}")
    except Exception as e:
        print(f"[gen_pipeline] Warning: failed computing metrics: {e}")
