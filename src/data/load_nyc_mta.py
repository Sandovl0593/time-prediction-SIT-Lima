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
    write_quality_report,
    compute_and_save_metrics,
)
from sklearn.decomposition import PCA


def _parse_bool_like(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return int(v) == 1
        except Exception:
            return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "si", "s"}


def _load_nodes_from_csv(path: str) -> gpd.GeoDataFrame:
    print(f"[load_nyc_mta] Reading stations CSV from {path}")
    df = pd.read_csv(path)
    # Eliminar completamente las columnas 'CBD' y 'Borough' por petición del usuario.
    for c in ["CBD", "Borough"]:
        if c in df.columns:
            print(f"[load_nyc_mta] Dropping column '{c}' as requested by user")
            df = df.drop(columns=[c])
    stop_id_col = "GTFS Stop ID"
    lat_col = "GTFS Latitude"
    lon_col = "GTFS Longitude"

    gdf_nodes = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
    )

    print("[load_nyc_mta] Projecting to metric CRS for distance calculations")
    gdf_proj = gdf_nodes.to_crs(epsg=3857)
    gdf_nodes["_x"] = gdf_proj.geometry.x
    gdf_nodes["_y"] = gdf_proj.geometry.y

    # 'CBD' and 'Borough' are intentionally discarded; no parsing required.

    return gdf_nodes


def _build_lines_from_nodes(gdf_nodes: gpd.GeoDataFrame):
    print("[load_nyc_mta] Building LineString geometries per 'Line' (assume header exists)")
    lines = []
    line_orders = []  # list of (line_name, ordered_gdf)
    for line_name, sub in gdf_nodes.groupby("Line"):
        sub = sub.dropna(subset=["_x", "_y"]) if len(sub) > 0 else sub
        if len(sub) < 2:
            continue
        coords = np.vstack([sub["_x"].values, sub["_y"].values]).T
        pca = PCA(n_components=1)
        proj = pca.fit_transform(coords).ravel()
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
    print("[load_nyc_mta] Building NetworkX MultiDiGraph with node attributes")
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
            "lines": row.get("Line"),
            "structure": row.get("Structure", "unknown"),
            "geometry": row.geometry,
        }
        G.add_node(node_id, **attrs)
        node_meta[node_id] = attrs

    print("[load_nyc_mta] Adding spatial edges between consecutive stations per line")
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


def _process_stop_times(stop_times_path: str, G: nx.MultiDiGraph, node_meta: dict, gdf_nodes: gpd.GeoDataFrame, trips_path: Optional[str] = None):
    if stop_times_path is None:
        print("[load_nyc_mta] No stop_times file provided; skipping observed travel-time processing")
        return

    if not os.path.exists(stop_times_path):
        print(f"[load_nyc_mta] stop_times file not found at {stop_times_path}; skipping")
        return

    print(f"[load_nyc_mta] Processing stop_times from {stop_times_path}")
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
        print(f"[load_nyc_mta] Reading trips file from {trips_path}")
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
    print(f"[load_nyc_mta] Found {len(st_order)} stop_time records with valid times for processing")

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
    print(f"[load_nyc_mta] Processed stop_times for {len(travel_acc)} observed edges")

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
    print(f"[load_nyc_mta] Added {len(travel_acc)} observed edges to graph")


def save_processed_graph(G: nx.MultiDiGraph, gdf_nodes: gpd.GeoDataFrame, gdf_lines: gpd.GeoDataFrame, out_dir: str):
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



def load_nyc_mta(
    path: str,
    stop_times_path: Optional[str] = None,
    trips_path: Optional[str] = None,
    scenario_id: str = "default",
    select_routes: Optional[List[str]] = None,
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
 
    print(f"[load_nyc_mta] Starting load for {path}")

    # Preparar directorios procesados
    processed_root = Path("src") / "data" / "processed"
    dirs = ensure_processed_dirs(processed_root)

    # 0) etapa de limpieza de estaciones (produce cleaned_stations.csv)
    try:
        cleaned_stations_path, stations_report = clean_stations(Path(path), processed_root, thresholds=cleaning_thresholds)
        stations_csv_path = str(cleaned_stations_path)
        print(f"[load_nyc_mta] Cleaned stations written to {cleaned_stations_path}")
    except Exception as e:
        print(f"[load_nyc_mta] Warning: failed cleaning stations: {e}; falling back to raw file")
        stations_report = {"file": Path(path).name, "original_rows": None}
        stations_csv_path = path

    # 1) read stations and build nodes GeoDataFrame (desde el CSV limpio)
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

    # 4) limpiar stop_times y trips (si existen) antes de procesar
    reports = [stations_report]

    cleaned_stop_times_path = None
    cleaned_trips_path = None
    trips_map = {}
    if stop_times_path and os.path.exists(stop_times_path):
        try:
            station_ids = set(gdf_nodes["GTFS Stop ID"].astype(str).values)
            cleaned_stop_times_path, stop_report = clean_stop_times(Path(stop_times_path), processed_root, station_ids, thresholds=cleaning_thresholds)
            reports.append(stop_report)
            print(f"[load_nyc_mta] Cleaned stop_times written to {cleaned_stop_times_path}")
            stop_times_to_process = str(cleaned_stop_times_path)
        except Exception as e:
            print(f"[load_nyc_mta] Warning: failed cleaning stop_times: {e}; using raw file")
            stop_times_to_process = stop_times_path
    else:
        stop_times_to_process = stop_times_path

    if trips_path and os.path.exists(trips_path):
        try:
            cleaned_trips_path, trips_report, trips_map = clean_trips(Path(trips_path), processed_root)
            reports.append(trips_report)
            print(f"[load_nyc_mta] Cleaned trips written to {cleaned_trips_path}")
        except Exception as e:
            print(f"[load_nyc_mta] Warning: failed cleaning trips: {e}; using raw file")

    # 5) procesar stop_times (si existe) usando el CSV limpio cuando esté disponible
    _process_stop_times(stop_times_to_process, G, node_meta, gdf_nodes, trips_path=cleaned_trips_path or trips_path)

    metadata = {"num_nodes": G.number_of_nodes(), "num_edges": G.number_of_edges()}

    print(f"[load_nyc_mta] Finished. Nodes: {metadata['num_nodes']}, Edges: {metadata['num_edges']}")

    # guardar procesado en carpeta 'processed' junto al raw
    # Guardar grafo procesado en la subcarpeta 'graph' dentro de processed
    graph_dir = dirs["graph"]
    try:
        save_processed_graph(G, gdf_nodes, gdf_lines, str(graph_dir))
        print(f"[load_nyc_mta] Saved processed CSVs to {graph_dir}")
    except Exception as e:
        print(f"[load_nyc_mta] Warning: failed saving processed files: {e}")
    # Para compatibilidad con scripts existentes que esperan nodes.csv en processed/
    try:
        save_processed_graph(G, gdf_nodes, gdf_lines, str(dirs["root"]))
        print(f"[load_nyc_mta] Also saved processed CSVs to {dirs['root']}")
    except Exception:
        pass

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
        print(f"[load_nyc_mta] Warning: failed writing manifest/quality report: {e}")

    # Calcular métricas de preprocesado y resumen por ruta
    try:
        metrics = compute_and_save_metrics(
            G,
            gdf_nodes,
            gdf_lines,
            reports,
            processed_root,
            scenario_id=scenario_id,
            select_routes=select_routes,
            target_km=target_km,
            curve_penalty=curve_penalty,
            tolerance=tolerance,
        )
        print(f"[load_nyc_mta] Computed preprocessing metrics: {metrics}")
    except Exception as e:
        print(f"[load_nyc_mta] Warning: failed computing metrics: {e}")


def visualize_nodes_edges(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_lines: gpd.GeoDataFrame,
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
    return ax
