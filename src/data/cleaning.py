"""Herramientas de limpieza y métricas para fuentes NYC (preprocesado).

Este módulo implementa validación de esquema, limpieza básica
(nulos, duplicados, IDs malformed, coordenadas inválidas, tiempos),
reportes de calidad y manifestos para reproducibilidad.

Las funciones escriben artefactos en una estructura bajo
`src/data/processed/` en subcarpetas `cleaned/`, `geometry/`,
`graph/`, `metrics/`, `manifests/`.

Las docstrings y comentarios están en español y explican por qué cada
paso importa para la geometría de rutas, el cálculo de distancias y la
extracción de tiempos observados.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point


def ensure_processed_dirs(processed_root: Path) -> Dict[str, Path]:
    processed_root = Path(processed_root)
    dirs = {
        "root": processed_root,
        "cleaned": processed_root / "cleaned",
        "geometry": processed_root / "geometry",
        "graph": processed_root / "graph",
        "metrics": processed_root / "metrics",
        "manifests": processed_root / "manifests",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # Buscador de columnas insensible a mayúsculas/espacios
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols:
            return cols[key]
    return None


def _parse_time_like(v: Any) -> float:
    """Convierte valores tipo 'HH:MM:SS' o segundos numéricos a float segundos.

    Por qué importa: los tiempos mal formados impiden calcular deltas de
    viaje observados; convertir formatos heterogéneos hace reproducible
    la extracción de travel_time entre par de estaciones.
    """
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    # formato hh:mm[:ss]
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 2:
                h, m = parts
                sec = h * 3600 + m * 60
            else:
                h, m, secp = parts[0], parts[1], parts[2]
                sec = h * 3600 + m * 60 + secp
            return float(sec)
        except Exception:
            return np.nan
    # intentar como número
    try:
        return float(s)
    except Exception:
        return np.nan


def clean_stations(raw_path: Path, processed_root: Path, thresholds: Optional[Dict[str, Any]] = None) -> Tuple[Path, Dict[str, Any]]:
    """Valida y limpia el CSV de estaciones.

    - Normaliza nombres de columnas y tipos.
    - Elimina duplicados exactos por (stop_id, lat, lon).
    - Elimina filas con coordenadas inválidas (fuera de rango) o no convertibles.
    - Imputa columnas no críticas (ej. 'CBD' -> False si falta).

    Retorna la ruta al CSV limpio y un reporte de calidad (diccionario).

    Notas: conservar geometría en EPSG:4326 y añadir columnas proyectadas
    en EPSG:3857 (`_x`, `_y`) para cálculos métricos posteriores.
    """
    raw_path = Path(raw_path)
    processed_root = Path(processed_root)
    dirs = ensure_processed_dirs(processed_root)

    df = pd.read_csv(raw_path, dtype=str)
    report: Dict[str, Any] = {"file": raw_path.name, "original_rows": len(df)}

    # Mapear columnas comunes a nombres esperados
    stop_col = _find_col(df, ["GTFS Stop ID", "stop_id", "stopid"])
    lat_col = _find_col(df, ["GTFS Latitude", "latitude", "lat"])
    lon_col = _find_col(df, ["GTFS Longitude", "longitude", "lon"])

    if stop_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"stations CSV missing required columns. Found columns: {list(df.columns)}")

    # Normalizar identificadores y coordenadas
    df[stop_col] = df[stop_col].astype(str).str.strip()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # Contar coordenadas inválidas y eliminarlas: lat fuera de [-90,90], lon fuera de [-180,180]
    # IMPORTANTE: coordenadas inválidas rompen la geometría de rutas (LineString)
    # y dan distancias métricas erróneas al proyectar a EPSG:3857. Por eso
    # las filas sin coordenadas útiles se descartan o requieren imputación
    # explícita si se dispone de otra fuente.
    before = len(df)
    invalid_coord_mask = df[lat_col].isna() | df[lon_col].isna() | (df[lat_col] < -90) | (df[lat_col] > 90) | (df[lon_col] < -180) | (df[lon_col] > 180)
    invalid_coords = int(invalid_coord_mask.sum())
    df_valid = df[~invalid_coord_mask].copy()

    # Duplicados exactos (stop_id, lat, lon)
    dup_subset = [stop_col, lat_col, lon_col]
    dup_before = len(df_valid)
    duplicates_mask = df_valid.duplicated(subset=dup_subset, keep="first")
    dup_count = int(duplicates_mask.sum())
    df_valid = df_valid.loc[~duplicates_mask].copy()

    # Descartar por completo 'CBD' y 'Borough' según petición del usuario.
    # No deben influir en ninguna etapa del preprocesado ni en las salidas.
    dropped_cols: List[str] = []
    drop_counts: Dict[str, int] = {}
    for c in ["CBD", "Borough"]:
        if c in df_valid.columns:
            drop_counts[c] = int(df_valid[c].notna().sum())
            df_valid = df_valid.drop(columns=[c])
            dropped_cols.append(c)
    # No se realizan imputaciones para estas columnas
    imputed = 0

    # Construir GeoDataFrame en WGS84
    gdf = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(df_valid[lon_col].astype(float), df_valid[lat_col].astype(float)), crs="EPSG:4326")

    # Proyectar para coordenadas métricas (EPSG:3857)
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf["_x"] = gdf_proj.geometry.x
    gdf["_y"] = gdf_proj.geometry.y

    out_path = dirs["cleaned"] / "cleaned_stations.csv"
    # Guardar geometría como WKT y mantener columnas originales
    gdf_out = gdf.copy()
    gdf_out["geometry_wkt"] = gdf_out.geometry.apply(lambda g: g.wkt if g is not None else "")
    gdf_out.to_csv(out_path, index=False)

    report.update({
        "kept_rows": len(gdf),
        "dropped_rows": before - len(gdf),
        "invalid_coords": invalid_coords,
        "duplicates_removed": dup_count,
        "imputed_values": imputed,
        "dropped_columns": dropped_cols,
        "dropped_counts": drop_counts,
        "crs": "EPSG:4326",
    })
    return out_path, report


def clean_stop_times(raw_path: Path, processed_root: Path, station_ids: Optional[set] = None, thresholds: Optional[Dict[str, Any]] = None) -> Tuple[Path, Dict[str, Any]]:
    """Valida y limpia el CSV de stop_times.

    - Normaliza columnas de trip_id/stop_id/arrival/departure.
    - Convierte tiempos heterogéneos (HH:MM:SS o segundos) a segundos numéricos.
    - Elimina duplicados exactos y reporta filas malformadas.

    `station_ids` se usa para calcular la tasa de matching entre stop_id y estaciones.
    """
    raw_path = Path(raw_path)
    processed_root = Path(processed_root)
    dirs = ensure_processed_dirs(processed_root)

    st = pd.read_csv(raw_path, dtype=str)
    report: Dict[str, Any] = {"file": raw_path.name, "original_rows": len(st)}

    # Mapear columnas comunes
    st_cols = {c.lower(): c for c in st.columns}
    trip_col = st_cols.get("trip_uid") or st_cols.get("trip_id")
    stop_col = st_cols.get("stop_id")
    arr_col = st_cols.get("arrival_time") or st_cols.get("arrival")
    dep_col = st_cols.get("departure_time") or st_cols.get("departure")

    if trip_col is None or stop_col is None:
        raise ValueError("stop_times file must contain a trip id column and a stop_id column")

    # Limpiar IDs
    st[trip_col] = st[trip_col].astype(str).str.strip()
    st[stop_col] = st[stop_col].astype(str).str.strip()

    # Convirtiendo tiempos a segundos
    # IMPORTANTE: los tiempos mal formados impiden calcular diferencias
    # entre llegadas/salidas (delta time) que usamos para estimar
    # travel_time observados por arista. Normalizar formatos (HH:MM:SS
    # o segundos) garantiza que el paso posterior (_process_stop_times)
    # pueda calcular deltas correctamente.
    time_cols = []
    malformed_times = 0
    for c in [arr_col, dep_col]:
        if c and c in st.columns:
            time_cols.append(c)
            st[c + "_secs"] = st[c].apply(_parse_time_like)
            malformed_times += int(st[c + "_secs"].isna().sum())

    # Detectar rango de fechas (si la columna existe)
    date_col = None
    for c in st.columns:
        if c.lower() in ("service_date", "date", "servicedate"):
            date_col = c
            break
    date_range = None
    if date_col:
        try:
            dts = pd.to_datetime(st[date_col], errors="coerce")
            if dts.notna().any():
                mn = dts.min()
                mx = dts.max()
                date_range = {"min": str(mn.date()), "max": str(mx.date())}
        except Exception:
            date_range = None

    before = len(st)
    # Eliminar duplicados por (trip, stop, arrival_sec, departure_sec)
    subset = [trip_col, stop_col] + [c + "_secs" for c in time_cols]
    dup_count = int(st.duplicated(subset=subset, keep="first").sum())
    st = st.drop_duplicates(subset=subset, keep="first").copy()

    # Matching rate con estaciones
    # Esta métrica indica cuánta parte de los registros de stop_times
    # corresponde a estaciones que conservamos tras la limpieza. Un
    # bajo matching_rate sugiere pérdida de cobertura de la red y
    # puede sesgar los travel_time observados.
    match_rate = None
    if station_ids is not None and len(station_ids) > 0:
        matched = st[stop_col].isin(station_ids).sum()
        total = len(st)
        match_rate = float(matched) / float(total) if total > 0 else 0.0

    out_path = dirs["cleaned"] / "cleaned_stop_times.csv"
    # Guardar versión limpia con columnas de segundos
    st.to_csv(out_path, index=False)

    report.update({
        "kept_rows": len(st),
        "dropped_rows": before - len(st),
        "duplicates_removed": dup_count,
        "malformed_time_values": malformed_times,
        "stop_id_match_rate": match_rate,
        "date_range": date_range,
        "cleaning_thresholds": thresholds or {},
    })
    return out_path, report


def clean_trips(raw_path: Optional[Path], processed_root: Path) -> Tuple[Optional[Path], Dict[str, Any], Dict[str, str]]:
    """Limpia archivo de trips/trip_times y devuelve mapping trip->route si existe.

    - Si existe mapeo de trip_id/trip_uid a route_id, devuelve el diccionario.
    - Si no, guarda el CSV limpio pero marca en el reporte que no es metadata de rutas.
    """
    processed_root = Path(processed_root)
    dirs = ensure_processed_dirs(processed_root)

    if raw_path is None:
        return None, {"file": None, "original_rows": 0}, {}

    df = pd.read_csv(raw_path, dtype=str)
    report = {"file": Path(raw_path).name, "original_rows": len(df)}
    tcols = {c.lower(): c for c in df.columns}
    t_trip_col = tcols.get("trip_uid") or tcols.get("trip_id")
    route_col = tcols.get("route_id")
    mapping = {}
    if t_trip_col and route_col:
        mapping = dict(zip(df[t_trip_col].astype(str), df[route_col].astype(str)))
        report["has_route_mapping"] = True
        report["kept_rows"] = len(df)
    else:
        report["has_route_mapping"] = False
        report["kept_rows"] = len(df)

    out_path = dirs["cleaned"] / "cleaned_trip_times.csv"
    df.to_csv(out_path, index=False)
    return out_path, report, mapping


def write_manifest(manifest: Dict[str, Any], manifests_dir: Path) -> Path:
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    path = manifests_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return path


def write_quality_report(report: Dict[str, Any], metrics_dir: Path) -> Path:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "quality_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return path


def compute_and_save_metrics(G, gdf_nodes: gpd.GeoDataFrame, gdf_lines: gpd.GeoDataFrame, reports: List[Dict[str, Any]], processed_root: Path, scenario_id: str = "default", select_routes: Optional[List[str]] = None, target_km: Optional[float] = None, curve_penalty: Optional[float] = None, tolerance: Optional[float] = None) -> Dict[str, Any]:
    """Computa métricas simples de preprocesado y guarda CSVs resumidos.

    Métricas implementadas: representatividad antes/después, missingness,
    duplicate rate, valid stop_id matching rate, edge coverage rate y
    resumen por ruta (agregado por día y por ruta si hay datos).
    """
    dirs = ensure_processed_dirs(processed_root)
    metrics = {}

    # Agregar estadísticos de grafo
    num_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    spatial_edges = sum(1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("spatial"))
    observed_edges = sum(1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("observed"))
    edge_coverage = float(observed_edges) / float(spatial_edges) if spatial_edges > 0 else None

    metrics.update({
        "num_nodes": num_nodes,
        "total_edges": total_edges,
        "spatial_edges": spatial_edges,
        "observed_edges": observed_edges,
        "edge_coverage_rate": edge_coverage,
    })

    # Representativeness y rates a partir de reports concatenados
    for r in reports:
        key = r.get("file") or "unknown"
        metrics[f"{key}_original_rows"] = r.get("original_rows")
        metrics[f"{key}_kept_rows"] = r.get("kept_rows")
        if r.get("original_rows"):
            metrics[f"{key}_representativeness"] = float(r.get("kept_rows", 0)) / float(r.get("original_rows"))

    # Guardar resumen de grafo
    graph_stats_path = dirs["metrics"] / "graph_stats.csv"
    pd.DataFrame([metrics]).to_csv(graph_stats_path, index=False)

    # Resumen de rutas: agregar por route_id si existen aristas con 'route_id'
    route_rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        route = d.get("route_id") or d.get("line")
        length_km = float(d.get("length_m", 0.0)) / 1000.0 if d.get("length_m") is not None else 0.0
        travel = float(d.get("travel_time_s", 0.0)) if d.get("travel_time_s") is not None else np.nan
        route_rows.append({"route": route, "length_km": length_km, "travel_time_s": travel})

    if route_rows:
        df_routes = pd.DataFrame(route_rows)
        summary = df_routes.groupby("route").agg({"length_km": "sum", "travel_time_s": ["mean", "count"]})
        # Aplanar multiindex
        summary.columns = ["_" . join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        fname = f"route_summary_{scenario_id}_t{target_km}_c{curve_penalty}_tol{tolerance}.csv"
        out_path = dirs["metrics"] / fname
        summary.to_csv(out_path, index=False)
        metrics["route_summary_path"] = str(out_path)

    # Si existe stop_times limpio, generar resúmenes por día y por weekday/weekend
    try:
        cleaned_st_path = dirs["cleaned"] / "cleaned_stop_times.csv"
        if cleaned_st_path.exists():
            st = pd.read_csv(cleaned_st_path, dtype=str)
            st_cols = {c.lower(): c for c in st.columns}
            trip_col = st_cols.get("trip_uid") or st_cols.get("trip_id")
            stop_col = st_cols.get("stop_id")

            # detectar o derivar fecha de servicio
            date_col = None
            for c in st.columns:
                if c.lower() in ("service_date", "date", "servicedate"):
                    date_col = c
                    break

            if date_col:
                st["_service_date"] = pd.to_datetime(st[date_col], errors="coerce").dt.date
            else:
                # si no hay columna explícita, intentar derivar desde una columna *_secs (epoch seconds)
                secs_col = None
                for cc in st.columns:
                    if "_secs" in cc.lower() or cc.lower().endswith("secs"):
                        secs_col = cc
                        break
                if secs_col:
                    try:
                        st["_service_date"] = pd.to_datetime(pd.to_numeric(st[secs_col], errors="coerce"), unit="s", errors="coerce").dt.date
                    except Exception:
                        st["_service_date"] = pd.NaT

            # si tenemos fechas (derivadas o explícitas), calcular resúmenes diarios y weekday/weekend
            if "_service_date" in st.columns and st["_service_date"].notna().any():
                daily = st.groupby("_service_date").agg(total_records=(trip_col, "count"), unique_trips=(trip_col, "nunique"))
                daily = daily.reset_index()
                daily_path = dirs["metrics"] / f"stop_times_daily_summary_{scenario_id}.csv"
                daily.to_csv(daily_path, index=False)
                metrics["stop_times_daily_summary"] = str(daily_path)

                # weekday/weekend summary
                try:
                    st["_weekday"] = pd.to_datetime(st["_service_date"], errors="coerce").dt.weekday
                    st["_is_weekend"] = st["_weekday"] >= 5
                    wk = st.groupby("_is_weekend").agg(total_records=(trip_col, "count"), unique_trips=(trip_col, "nunique"))
                    wk = wk.reset_index()
                    wk_path = dirs["metrics"] / f"stop_times_weekday_summary_{scenario_id}.csv"
                    wk.to_csv(wk_path, index=False)
                    metrics["stop_times_weekday_summary"] = str(wk_path)
                except Exception:
                    pass

                # per-route daily summary if trips mapping exists
                cleaned_trips_path = dirs["cleaned"] / "cleaned_trip_times.csv"
                if cleaned_trips_path.exists():
                    try:
                        trips_df = pd.read_csv(cleaned_trips_path, dtype=str)
                        tcols = {c.lower(): c for c in trips_df.columns}
                        t_trip_col = tcols.get("trip_uid") or tcols.get("trip_id")
                        route_col = tcols.get("route_id")
                        if t_trip_col and route_col:
                            trips_map = dict(zip(trips_df[t_trip_col].astype(str), trips_df[route_col].astype(str)))
                            st["_route_id"] = st[trip_col].map(trips_map)
                            per_route = st.groupby(["_service_date", "_route_id"]).agg(total_records=(trip_col, "count"), unique_trips=(trip_col, "nunique"))
                            per_route = per_route.reset_index()
                            per_route_path = dirs["metrics"] / f"per_route_daily_summary_{scenario_id}.csv"
                            per_route.to_csv(per_route_path, index=False)
                            metrics["per_route_daily_summary"] = str(per_route_path)
                    except Exception:
                        pass
    except Exception:
        pass

    return metrics
