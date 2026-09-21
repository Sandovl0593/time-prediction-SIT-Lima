"""Genera figuras y ejemplos verificables para el análisis P0 de Lima."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _latest_routes(outputs_dir: Path, model: str) -> Path:
    runs = sorted(
        (path for path in (outputs_dir / "training" / model).iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"No hay corridas para {model}.")
    path = runs[-1] / "route_predictions" / "all_routes.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return path


def _load_valid(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    return data.loc[data["covered"].fillna(False).astype(bool) & data["has_target"].fillna(False).astype(bool)].copy()


def _sample_routes(gat: pd.DataFrame, gatv2: pd.DataFrame) -> pd.DataFrame:
    """Pick representative shared routes from short, medium and long bins."""
    key_cols = ["start_stop_id", "end_stop_id", "hops"]
    gat_keyed = gat.set_index(key_cols, drop=False)
    gatv2_keyed = gatv2.set_index(key_cols, drop=False)
    rows: list[dict[str, object]] = []
    for bin_km in (5.0, 15.0, 30.0):
        candidates = gat_keyed[gat_keyed["tol_prox"] == bin_km]
        candidates = candidates[candidates.index.isin(gatv2_keyed.index)]
        if candidates.empty:
            continue
        # La ruta cuya longitud es la mediana evita seleccionar un extremo.
        median_length = candidates["length_real_km"].median()
        chosen = candidates.iloc[(candidates["length_real_km"] - median_length).abs().argsort().iloc[0]]
        route_key = tuple(chosen[column] for column in key_cols)
        for model, frame in (("GAT", gat_keyed), ("GATv2", gatv2_keyed)):
            route = frame.loc[route_key]
            if isinstance(route, pd.DataFrame):
                route = route.iloc[0]
            preds = json.loads(route["edge_predictions_s"])
            targets = json.loads(route["edge_targets_s"])
            rows.append(
                {
                    "model": model,
                    "bin_km": bin_km,
                    "line": route["line"],
                    "start_stop_id": route["start_stop_id"],
                    "end_stop_id": route["end_stop_id"],
                    "n_hops": int(route["n_hops"]),
                    "length_real_km": float(route["length_real_km"]),
                    "target_s": float(route["target"]),
                    "prediction_s": float(route["pred"]),
                    "sum_edge_predictions_s": float(route["sum_edge_predictions_s"]),
                    "sum_edge_targets_s": float(route["sum_edge_targets_s"]),
                    "absolute_error_s": abs(float(route["pred"]) - float(route["target"])),
                    "edge_predictions_s": json.dumps([round(value, 2) for value in preds]),
                    "edge_targets_s": json.dumps([round(value, 2) for value in targets]),
                }
            )
    return pd.DataFrame(rows)


def _plot_scatter(gat: pd.DataFrame, gatv2: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    maximum = max(gat[["target", "pred"]].to_numpy().max(), gatv2[["target", "pred"]].to_numpy().max())
    for axis, data, title, color in zip(axes, (gat, gatv2), ("GAT", "GATv2"), ("#1f77b4", "#d62728")):
        sample = data.sample(min(1800, len(data)), random_state=42)
        axis.scatter(sample["target"], sample["pred"], s=9, alpha=0.35, color=color, edgecolors="none")
        axis.plot([0, maximum], [0, maximum], "--", color="black", linewidth=1, label="predicción = objetivo")
        axis.set_title(title)
        axis.set_xlabel("Tiempo objetivo proyectado (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Tiempo predicho de ruta (s)")
    axes[1].legend(loc="upper left")
    fig.suptitle("P0 Lima: objetivo frente a suma de predicciones de aristas")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bins(metrics: pd.DataFrame, output: Path) -> None:
    bins = sorted(float(value) for value in metrics.loc[metrics["bin_km"] != "global", "bin_km"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    offsets = {"GAT": -0.18, "GATV2": 0.18}
    colors = {"GAT": "#1f77b4", "GATV2": "#d62728"}
    labels = {"GAT": "GAT", "GATV2": "GATv2"}
    for model in ("GAT", "GATV2"):
        subset = metrics[(metrics["model"] == model) & (metrics["bin_km"] != "global")].copy()
        subset["bin_km"] = subset["bin_km"].astype(float)
        subset = subset.set_index("bin_km").reindex(bins)
        x = np.arange(len(bins)) + offsets[model]
        axes[0].bar(x, subset["rmse_s"], width=0.36, color=colors[model], label=labels[model])
        axes[1].plot(bins, subset["r2"], marker="o", color=colors[model], label=labels[model])
    axes[0].set_xticks(np.arange(len(bins)), [str(int(value)) for value in bins])
    axes[0].set_xlabel("Bin de longitud (km)")
    axes[0].set_ylabel("RMSE de ruta (s)")
    axes[0].set_title("Error por bin")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Bin de longitud (km)")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("Capacidad explicativa por bin")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("P0 Lima: comparación de desempeño por longitud de ruta")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_decomposition(samples: pd.DataFrame, output: Path) -> None:
    bins = samples["bin_km"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(bins), 1, figsize=(11, 2.6 * len(bins)), squeeze=False)
    palette = plt.cm.tab20.colors
    for axis, bin_km in zip(axes[:, 0], bins):
        subset = samples[samples["bin_km"] == bin_km]
        for row_number, (_, row) in enumerate(subset.iterrows()):
            values = json.loads(row["edge_predictions_s"])
            left = 0.0
            for index, value in enumerate(values):
                axis.barh(row_number, value, left=left, color=palette[index % len(palette)], height=0.55)
                left += value
            axis.axvline(row["target_s"], ymin=(row_number + 0.15) / max(2, len(subset)), ymax=(row_number + 0.85) / max(2, len(subset)), color="black", linewidth=2)
        first = subset.iloc[0]
        axis.set_yticks(range(len(subset)), subset["model"])
        axis.set_xlabel("Segundos; segmentos apilados = suma de predicciones de aristas")
        axis.set_title(f"Ruta representativa del bin {int(bin_km)} km: {first['start_stop_id']} → {first['end_stop_id']} ({int(first['n_hops'])} aristas)")
        axis.grid(axis="x", alpha=0.25)
    fig.text(0.98, 0.01, "Línea negra: tiempo objetivo proyectado", ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_p0_figures(outputs_dir: Path) -> dict[str, Path]:
    outputs_dir = Path(outputs_dir)
    figure_dir = outputs_dir / "lima" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    gat = _load_valid(_latest_routes(outputs_dir, "gat"))
    gatv2 = _load_valid(_latest_routes(outputs_dir, "gatv2"))
    metrics = pd.read_csv(outputs_dir / "lima" / "p0_metrics.csv")
    samples = _sample_routes(gat, gatv2)
    sample_path = figure_dir / "p0_route_samples.csv"
    samples.to_csv(sample_path, index=False)
    paths = {
        "scatter": figure_dir / "p0_target_vs_prediction.png",
        "bins": figure_dir / "p0_performance_by_bin.png",
        "decomposition": figure_dir / "p0_route_decomposition.png",
        "samples": sample_path,
    }
    _plot_scatter(gat, gatv2, paths["scatter"])
    _plot_bins(metrics, paths["bins"])
    _plot_decomposition(samples, paths["decomposition"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("src/outputs"))
    args = parser.parse_args()
    for name, path in generate_p0_figures(args.outputs_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
