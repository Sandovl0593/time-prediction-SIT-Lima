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
from shapely.geometry import Point, LineString
from datetime import datetime
import logging

from src.utils.others import get_logger

# Logger del módulo — se puede añadir FileHandler vía get_logger(..., log_file=...)
logger = get_logger("data_cleaning")


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


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Escribe CSV sin sobrescribir silenciosamente: si existe, añade sufijo con timestamp."""
    path = Path(path)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        new_name = f"{path.stem}_{ts}{path.suffix}"
        new_path = path.with_name(new_name)
        df.to_csv(new_path, index=False)
        logger.info(f"Existing file {path} preserved; wrote {new_path} instead")
        return new_path
    else:
        df.to_csv(path, index=False)
        return path


def validate_schema(df: pd.DataFrame, required: List[str], file_label: str) -> Tuple[bool, List[str]]:
    """Valida que el DataFrame tenga las columnas requeridas.

    Retorna (ok, missing_cols)
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"{file_label}: missing required columns: {missing}")
        return False, missing
    return True, []


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

    Si el CSV limpio ya existe en disco, se omite la limpieza y se regenera
    únicamente el reporte de calidad a partir del archivo existente.

    Retorna la ruta al CSV limpio y un reporte de calidad (diccionario).

    Notas: conservar geometría en EPSG:4326 y añadir columnas proyectadas
    en EPSG:3857 (`_x`, `_y`) para cálculos métricos posteriores.
    """
    raw_path = Path(raw_path)
    processed_root = Path(processed_root)
    dirs = ensure_processed_dirs(processed_root)

    # Si el CSV limpio ya existe, regenerar solo el reporte de calidad desde él
    out_path_check = dirs["cleaned"] / "cleaned_stations.csv"
    if out_path_check.exists():
        logger.info(f"[clean_stations] CSV limpio ya existe en {out_path_check}; regenerando reporte desde archivo existente")
        existing_df = pd.read_csv(out_path_check)
        report: Dict[str, Any] = {
            "file": raw_path.name,
            "note": "skipped_cleaning_exists",
            "original_rows": len(existing_df),
            "kept_rows": len(existing_df),
            "dropped_rows": 0,
            "out_path": str(out_path_check),
            "columns": list(existing_df.columns),
            "coverage_pct": 1.0,
            "crs": "EPSG:4326",
        }
        return out_path_check, report

    df = pd.read_csv(raw_path, dtype=str)
    report: Dict[str, Any] = {"file": raw_path.name, "original_rows": len(df)}

    # parámetros por defecto
    thresholds = thresholds or {}
    min_lat = thresholds.get("min_lat", -90)
    max_lat = thresholds.get("max_lat", 90)
    min_lon = thresholds.get("min_lon", -180)
    max_lon = thresholds.get("max_lon", 180)
    id_pattern = thresholds.get("id_pattern", r"^[A-Za-z0-9_.-]+$")

    # Mapear columnas comunes a nombres esperados
    stop_col = _find_col(df, ["GTFS Stop ID", "stop_id", "stopid"])
    lat_col = _find_col(df, ["GTFS Latitude", "latitude", "lat"])
    lon_col = _find_col(df, ["GTFS Longitude", "longitude", "lon"])

    if stop_col is None or lat_col is None or lon_col is None:
        raise ValueError(f"stations CSV missing required columns. Found columns: {list(df.columns)}")

    # Normalize and basic corrections
    df[stop_col] = df[stop_col].astype(str).str.strip()
    corrected = 0
    # numeric coercion
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    corrected += int(df[lat_col].notna().sum() + df[lon_col].notna().sum())

    before = len(df)

    # Detectar coordenadas inválidas
    invalid_coord_mask = (
        df[lat_col].isna() | df[lon_col].isna() | (df[lat_col] < min_lat) | (df[lat_col] > max_lat) | (df[lon_col] < min_lon) | (df[lon_col] > max_lon)
    )
    invalid_coords = int(invalid_coord_mask.sum())
    df_valid = df[~invalid_coord_mask].copy()

    # Detección de IDs malformados
    bad_id_mask = ~df_valid[stop_col].astype(str).str.match(id_pattern)
    bad_ids = int(bad_id_mask.sum())
    # intentar limpiar IDs malformados (trim y extraer prefijo numérico)
    if bad_ids > 0:
        logger.info(f"{bad_ids} station IDs appear malformed; attempting basic cleanup")
        def _fix_id(s: str) -> str:
            s2 = str(s).strip()
            m = re.match(r"^(\d+)[A-Za-z].*$", s2)
            if m:
                return m.group(1)
            return s2

        df_valid[stop_col] = df_valid[stop_col].apply(_fix_id)
        # Re-evaluate
        bad_id_mask = ~df_valid[stop_col].astype(str).str.match(id_pattern)
        bad_ids_after = int(bad_id_mask.sum())
        corrected += (bad_ids - bad_ids_after)
        bad_ids = bad_ids_after

    # Duplicados por (stop_id, lat, lon)
    dup_subset = [stop_col, lat_col, lon_col]
    duplicates_mask = df_valid.duplicated(subset=dup_subset, keep="first")
    dup_count = int(duplicates_mask.sum())
    df_valid = df_valid.loc[~duplicates_mask].copy()

    # Duplicados de stop_id con coordenadas distintas -> resolver conservando fila con más datos
    dup_id_counts = df_valid.duplicated(subset=[stop_col], keep=False).sum()
    if dup_id_counts > 0:
        # agrupar y conservar el primer con menos nulos
        def _keep_best(group):
            nonnull_counts = group.notna().sum(axis=1)
            return group.loc[nonnull_counts.idxmax()]
        groups = df_valid.groupby(stop_col)
        to_keep = []
        for name, g in groups:
            if len(g) == 1:
                to_keep.append(g.index[0])
            else:
                best = _keep_best(g)
                to_keep.append(best.name)
        df_valid = df_valid.loc[to_keep].copy()

    # Descartar por completo 'CBD' y 'Borough' según petición del usuario.
    dropped_cols: List[str] = []
    drop_counts: Dict[str, int] = {}
    for c in ["CBD", "Borough"]:
        if c in df_valid.columns:
            drop_counts[c] = int(df_valid[c].notna().sum())
            df_valid = df_valid.drop(columns=[c])
            dropped_cols.append(c)

    # Construir GeoDataFrame en WGS84
    gdf = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(df_valid[lon_col].astype(float), df_valid[lat_col].astype(float)), crs="EPSG:4326")

    # Proyectar para coordenadas métricas (EPSG:3857) y calcular x/y
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf["_x"] = gdf_proj.geometry.x
    gdf["_y"] = gdf_proj.geometry.y

    out_path = dirs["cleaned"] / "cleaned_stations.csv"
    # Guardar geometría como WKT y mantener columnas originales
    gdf_out = gdf.copy()
    gdf_out["geometry_wkt"] = gdf_out.geometry.apply(lambda g: g.wkt if g is not None else "")
    out_written = _safe_write_csv(gdf_out, out_path)

    kept = len(gdf)
    dropped = before - kept
    report.update({
        "kept_rows": kept,
        "dropped_rows": dropped,
        "invalid_coords": invalid_coords,
        "duplicates_removed": dup_count,
        "malformed_ids_remaining": bad_ids,
        "corrected_rows": corrected,
        "dropped_columns": dropped_cols,
        "dropped_counts": drop_counts,
        "columns": list(gdf_out.columns),
        "coverage_pct": float(kept) / float(before) if before > 0 else None,
        "crs": "EPSG:4326",
        "out_path": str(out_written),
    })
    return out_written, report


def clean_stop_times(raw_path: Path, processed_root: Path, station_ids: Optional[set] = None, thresholds: Optional[Dict[str, Any]] = None, trips_map: Optional[Dict[str, str]] = None) -> Tuple[Path, Dict[str, Any]]:
    """Valida y limpia el CSV de stop_times.

    - Normaliza columnas de trip_id/stop_id/arrival/departure.
    - Convierte tiempos heterogéneos (HH:MM:SS o segundos) a segundos numéricos.
    - Elimina duplicados exactos y reporta filas malformadas.

    Si el CSV limpio ya existe en disco, se omite la limpieza y se regenera
    únicamente el reporte de cobertura a partir del archivo existente.

    `station_ids` se usa para calcular la tasa de matching entre stop_id y estaciones.
    """
    raw_path = Path(raw_path)
    processed_root = Path(processed_root)
    dirs = ensure_processed_dirs(processed_root)

    # Si el CSV limpio ya existe, regenerar solo el reporte de cobertura desde él
    out_path_check = dirs["cleaned"] / "cleaned_stop_times.csv"
    if out_path_check.exists():
        logger.info(f"[clean_stop_times] CSV limpio ya existe en {out_path_check}; regenerando reporte desde archivo existente")
        existing_st = pd.read_csv(out_path_check)
        match_rate = None
        if station_ids is not None and len(station_ids) > 0:
            st_cols_existing = {c.lower(): c for c in existing_st.columns}
            stop_col_existing = st_cols_existing.get("stop_id")
            if stop_col_existing:
                matched = existing_st[stop_col_existing].astype(str).isin(station_ids).sum()
                match_rate = float(matched) / len(existing_st) if len(existing_st) > 0 else 0.0
        report: Dict[str, Any] = {
            "file": raw_path.name,
            "note": "skipped_cleaning_exists",
            "original_rows": len(existing_st),
            "kept_rows": len(existing_st),
            "dropped_rows": 0,
            "stop_id_match_rate": match_rate,
            "out_path": str(out_path_check),
        }
        return out_path_check, report

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
    mapping: Dict[str, str] = {}
    inconsistent = 0
    if t_trip_col and route_col:
        # detectar trip_id duplicados que apuntan a diferentes route_id
        grp = df.groupby(t_trip_col)[route_col].nunique()
        inconsistent = int((grp > 1).sum())
        if inconsistent > 0:
            logger.warning(f"{inconsistent} trip_ids map to multiple route_id values; keeping first occurrence")
        # conservar la primera aparición por trip_id
        df_clean = df.drop_duplicates(subset=[t_trip_col], keep="first").copy()
        mapping = dict(zip(df_clean[t_trip_col].astype(str), df_clean[route_col].astype(str)))
        report["has_route_mapping"] = True
        report["kept_rows"] = len(df_clean)
    else:
        df_clean = df.copy()
        report["has_route_mapping"] = False
        report["kept_rows"] = len(df_clean)

    out_path = dirs["cleaned"] / "cleaned_trip_times.csv"
    out_written = _safe_write_csv(df_clean, out_path)
    report.update({"inconsistent_trip_mappings": inconsistent, "out_path": str(out_written)})
    return out_written, report, mapping


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
