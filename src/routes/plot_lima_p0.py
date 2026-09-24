"""Genera métricas, rutas multi-línea y pruebas de submuestreo de P0."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.routes.evaluate_p0 import DEFAULT_GAT_PATH, DEFAULT_GATV2_PATH
from src.routes.p0_analysis import (
    DEFAULT_SEGMENTS_PER_SIDE,
    DEFAULT_TRANSFER_MAX_KM,
    generate_p0_report,
)


def generate_p0_figures(
    outputs_dir: Path = Path("src/outputs"),
    gat_path: Path = DEFAULT_GAT_PATH,
    gatv2_path: Path = DEFAULT_GATV2_PATH,
    transfer_max_km: float = DEFAULT_TRANSFER_MAX_KM,
    segments_per_side: int = DEFAULT_SEGMENTS_PER_SIDE,
) -> dict[str, Path]:
    """Compatibility wrapper for the complete P0 reporting pipeline."""
    return generate_p0_report(
        outputs_dir=outputs_dir,
        gat_path=gat_path,
        gatv2_path=gatv2_path,
        transfer_max_km=transfer_max_km,
        segments_per_side=segments_per_side,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("src/outputs"))
    parser.add_argument("--gat", type=Path, default=DEFAULT_GAT_PATH)
    parser.add_argument("--gatv2", type=Path, default=DEFAULT_GATV2_PATH)
    parser.add_argument("--transfer-max-km", type=float, default=DEFAULT_TRANSFER_MAX_KM)
    parser.add_argument("--segments-per-side", type=int, default=DEFAULT_SEGMENTS_PER_SIDE)
    args = parser.parse_args()
    for name, path in generate_p0_figures(
        args.outputs_dir,
        args.gat,
        args.gatv2,
        args.transfer_max_km,
        args.segments_per_side,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
