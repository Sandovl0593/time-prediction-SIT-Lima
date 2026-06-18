"""Herramientas para seleccionar y exportar los top-k tramos desde un `route.csv`.

Provee funciones para:
- resolver rutas a `route.csv` (acepta path directo, carpeta que contiene `route.csv`, o un cfg_tag bajo `src/outputs/`)
- leer el CSV a `pandas.DataFrame`
- seleccionar top-k tramos por una métrica (por defecto `score`)
- exportar el subconjunto a `src/routes/` preservando `geometry_wkt`

El archivo está pensado para integrarse con el visualizador en
`src/main_process/visualize.py` que espera la columna `geometry_wkt`.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[export_top_segments] %(message)s")


def resolve_route_csv(path_or_cfg_tag: str) -> Optional[str]:
    """Resolver un argumento a la ruta de un `route.csv`.

    Acepta:
    - path a un archivo CSV
    - path a un directorio que contiene `route.csv`
    - cfg_tag (p.ej. "target_20km_pen0p1_tol0p1") que se resuelve en
      `src/outputs/<cfg_tag>/route.csv`

    Devuelve la ruta como string o `None` si no se encuentra.
    """
    if not path_or_cfg_tag:
        return None

    p = Path(path_or_cfg_tag)

    # Si es un archivo csv existente
    if p.is_file() and p.suffix.lower() == ".csv":
        return str(p)

    # Si es un directorio, buscar route.csv dentro
    if p.is_dir():
        candidate = p / "route.csv"
        if candidate.is_file():
            return str(candidate)

    # Tratar como cfg_tag bajo src/outputs/<cfg_tag>/route.csv
    candidate = Path("src") / "outputs" / path_or_cfg_tag / "route.csv"
    if candidate.is_file():
        return str(candidate)

    # No resuelto
    return None


def read_route_csv(route_csv_path: str) -> pd.DataFrame:
    """Leer un `route.csv` y devolver un DataFrame.

    Lanza FileNotFoundError si no existe el CSV resuelto.
    """
    resolved = resolve_route_csv(route_csv_path) or route_csv_path
    p = Path(resolved)
    if not p.exists():
        raise FileNotFoundError(f"route.csv no encontrado en: {route_csv_path} (resuelto: {resolved})")

    df = pd.read_csv(str(p), low_memory=False)
    return df


def select_top_k(df: pd.DataFrame, k: int = 50, sort_by: str = "score", ascending: bool = False, filter_accepted: Optional[bool] = None) -> pd.DataFrame:
    """Seleccionar top-k filas del DataFrame ordenadas por `sort_by`.

    - `filter_accepted`: si no es None y existe la columna `accepted`, filtra por esa columna.
    - Devuelve una copia del DataFrame con como máximo `k` filas.
    """
    if filter_accepted is not None and "accepted" in df.columns:
        df = df[df["accepted"] == bool(filter_accepted)].copy()

    if sort_by not in df.columns:
        raise ValueError(f"La columna para ordenar '{sort_by}' no está presente en el DataFrame")

    # ordenar, colocando NaNs al final para evitar que contaminen el top
    df_sorted = df.sort_values(by=sort_by, ascending=ascending, na_position="last")
    topk = df_sorted.head(k).copy()
    return topk


def export_top_k_csv(df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Exportar `df` a CSV. Si `out_path` es None, crea `src/routes/topk_segments.csv`.

    Se asegura de que exista la columna `geometry_wkt` (creándola a partir de `geometry`
    si es posible) porque el visualizador la necesita.
    """
    if out_path is None:
        out_dir = Path("src") / "routes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "topk_segments.csv"
    else:
        out_path = Path(out_path)
        if out_path.parent:
            out_path.parent.mkdir(parents=True, exist_ok=True)

    # Garantizar geometry_wkt
    if "geometry_wkt" not in df.columns:
        if "geometry" in df.columns:
            try:
                df = df.copy()
                df["geometry_wkt"] = df["geometry"].apply(lambda g: getattr(g, "wkt", str(g)))
            except Exception:
                df = df.copy()
                df["geometry_wkt"] = ""
        else:
            df = df.copy()
            df["geometry_wkt"] = ""

    df.to_csv(str(out_path), index=False)
    logger.info("Exportado top-k CSV a %s", str(out_path))
    return str(out_path)


def export_top_k_from_route_csv(route_csv_path: str, k: int = 50, sort_by: str = "score",
                                ascending: bool = False,
                                filter_accepted: Optional[bool] = None) -> str:
    """Leer un `route.csv`, seleccionar top-k y exportarlo.

    - `route_csv_path`: path / dir / cfg_tag (resuelto con `resolve_route_csv`).
    - `out_dir`: si no se especifica, usa `src/routes/`.
    - `out_name`: nombre de archivo opcional; si no está, se usa `<cfg>_top{k}.csv`.

    Devuelve la ruta al CSV exportado.
    """
    resolved = resolve_route_csv(route_csv_path)
    if resolved is None:
        raise FileNotFoundError(f"No se pudo resolver route.csv desde '{route_csv_path}'")

    df = pd.read_csv(resolved, low_memory=False)
    topk = select_top_k(df, k=k, sort_by=sort_by, ascending=ascending, filter_accepted=filter_accepted)

    out_dir_path = Path("src") / "starter"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    parent = Path(resolved).parent.name or "starter"
    out_name = f"top{k}.csv"

    out_path = out_dir_path / parent / out_name
    exported = export_top_k_csv(topk, out_path)
    return exported

