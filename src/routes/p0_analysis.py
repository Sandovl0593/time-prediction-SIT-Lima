"""Análisis exploratorio y pruebas de submuestreo para P0 Lima.

La única fuente son ``all_routes_gat.csv`` y ``all_routes_gatv2.csv``. Las
rutas multi-línea se construyen uniendo dos subtramos uni-línea mediante una
sola arista de transferencia corta; no se enumeran permutaciones de líneas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.routes.evaluate_p0 import evaluate_predictions, load_valid_predictions


ROUTE_KEYS = ["scenario_id", "route_type", "line", "start_stop_id", "end_stop_id", "hops"]
MODEL_COLORS = {"GAT": "#1f77b4", "GATv2": "#d62728"}
LINE_ORDER = ["BRT", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "TRANSFER"]
KM_BINS = (5, 10, 15, 20, 30, 40)
BIN_EDGES_KM = (0.0, 7.5, 12.5, 17.5, 25.0, 35.0, 45.0)
DEFAULT_TRANSFER_MAX_KM = 0.15
DEFAULT_SEGMENTS_PER_SIDE = 14
SUBSAMPLE_FRACTIONS = (0.25, 0.50, 0.75)
SUBSAMPLE_SEEDS = (20260921, 20260922, 20260923)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return _correlation(left_rank, right_rank)


def _assert_same_routes(gat: pd.DataFrame, gatv2: pd.DataFrame) -> None:
    """Verify that model predictions refer to the same ordered observations."""
    comparable = ROUTE_KEYS + ["target", "tol_prox", "length_real_km", "n_hops"]
    missing = [column for column in comparable if column not in gat or column not in gatv2]
    if missing:
        raise ValueError(f"No se puede alinear P0; faltan columnas: {missing}")
    if not gat[comparable].reset_index(drop=True).equals(gatv2[comparable].reset_index(drop=True)):
        raise ValueError("Los CSV de GAT y GATv2 no contienen exactamente la misma muestra P0.")
    for name, frame in (("GAT", gat), ("GATv2", gatv2)):
        if frame.duplicated(ROUTE_KEYS).any():
            raise ValueError(f"{name}: existen claves de ruta P0 duplicadas.")


def _diagnostics(frame: pd.DataFrame, model: str, group: str, value: object) -> dict[str, object]:
    target = frame["target"].to_numpy(dtype=float)
    prediction = frame["pred"].to_numpy(dtype=float)
    residual = prediction - target
    absolute = np.abs(residual)
    percentage = absolute / np.maximum(np.abs(target), 1e-12) * 100
    mse = float(np.mean(np.square(residual)))
    target_variation = float(np.square(target - target.mean()).sum())
    slope, intercept = np.polyfit(target, prediction, 1) if len(frame) > 1 else (float("nan"), float("nan"))
    return {
        "model": model,
        "group": group,
        "value": value,
        "n_routes": int(len(frame)),
        "mean_target_s": float(target.mean()),
        "target_sd_s": float(target.std(ddof=0)),
        "mean_prediction_s": float(prediction.mean()),
        "prediction_sd_s": float(prediction.std(ddof=0)),
        "mean_bias_s": float(residual.mean()),
        "median_error_s": float(np.median(residual)),
        "residual_sd_s": float(residual.std(ddof=0)),
        "error_q05_s": float(np.quantile(residual, 0.05)),
        "error_q25_s": float(np.quantile(residual, 0.25)),
        "error_q75_s": float(np.quantile(residual, 0.75)),
        "error_q95_s": float(np.quantile(residual, 0.95)),
        "rmse_s": float(np.sqrt(mse)),
        "mae_s": float(absolute.mean()),
        "mape_pct": float(percentage.mean()),
        "median_ape_pct": float(np.median(percentage)),
        "smape_pct": float(np.mean(2 * absolute / np.maximum(np.abs(target) + np.abs(prediction), 1e-12)) * 100),
        "r2": float(1 - np.square(residual).sum() / target_variation) if target_variation else float("nan"),
        "overprediction_pct": float(np.mean(residual > 0) * 100),
        "underprediction_pct": float(np.mean(residual < 0) * 100),
        "within_10pct_pct": float(np.mean(percentage <= 10) * 100),
        "within_20pct_pct": float(np.mean(percentage <= 20) * 100),
        "bias_share_mse_pct": float(np.square(residual.mean()) / mse * 100) if mse else 0.0,
        "calibration_slope": float(slope),
        "calibration_intercept_s": float(intercept),
        "pearson_target_prediction": _correlation(target, prediction),
        "spearman_target_prediction": _spearman(target, prediction),
        "spearman_abs_error_target": _spearman(absolute, target),
        "spearman_abs_error_length": _spearman(absolute, frame["length_real_km"].to_numpy(dtype=float)),
        "spearman_abs_error_hops": _spearman(absolute, frame["n_hops"].to_numpy(dtype=float)),
        "residual_skewness": float(pd.Series(residual).skew()),
        "residual_excess_kurtosis": float(pd.Series(residual).kurt()),
        "mean_error_per_hop_s": float(np.mean(residual / frame["n_hops"].to_numpy(dtype=float))),
        "max_absolute_error_s": float(absolute.max()),
    }


def _diagnostic_table(data_by_model: dict[str, pd.DataFrame], group_column: str | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in data_by_model.items():
        if group_column is None:
            rows.append(_diagnostics(frame, model, "global", "global"))
        else:
            for value, group in frame.groupby(group_column, sort=False):
                rows.append(_diagnostics(group, model, group_column, value))
    return pd.DataFrame(rows)


def _top_errors(data_by_model: dict[str, pd.DataFrame], count: int = 20) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ROUTE_KEYS + ["n_hops", "length_real_km", "tol_prox", "target", "pred"]
    for model, frame in data_by_model.items():
        result = frame[columns].copy()
        result.insert(0, "model", model)
        result["error_s"] = result["pred"] - result["target"]
        result["absolute_error_s"] = result["error_s"].abs()
        frames.append(result.nlargest(count, "absolute_error_s"))
    return pd.concat(frames, ignore_index=True)


def _parse_hops(value: object) -> list[str]:
    hops = json.loads(str(value))
    if not isinstance(hops, list) or len(hops) < 2:
        raise ValueError(f"Secuencia de hops inválida: {value!r}")
    return [str(stop) for stop in hops]


def _representative_segments(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Select length-quantile representatives instead of a full Cartesian set."""
    ordered = frame.sort_values(["length_real_km", "scenario_id"]).reset_index(drop=True)
    if len(ordered) <= maximum:
        return ordered
    positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, maximum)).astype(int))
    return ordered.iloc[positions].reset_index(drop=True)


def _length_bin(length_km: float) -> int | None:
    if not np.isfinite(length_km) or length_km <= BIN_EDGES_KM[0] or length_km > BIN_EDGES_KM[-1]:
        return None
    index = int(np.searchsorted(np.asarray(BIN_EDGES_KM[1:]), length_km, side="left"))
    return int(KM_BINS[index])


def _combined_source_rows(gat: pd.DataFrame, gatv2: pd.DataFrame) -> pd.DataFrame:
    combined = gat.reset_index(drop=True).copy().rename(columns={"pred": "pred_gat"})
    combined["pred_gatv2"] = gatv2.reset_index(drop=True)["pred"].to_numpy(dtype=float)
    return combined


def _bidirectional_route_view(source: pd.DataFrame) -> pd.DataFrame:
    """Create a temporary two-direction view from unique physical routes.

    The persisted Lima CSVs keep one canonical orientation per intercepted
    node sequence. Multi-line composition still needs to approach a transfer
    from either side, so reverse routes are materialized only in memory. Route
    targets and predictions are unchanged because both directions describe the
    same physical route in the Lima scenario.
    """
    reverse = source.copy()
    reverse["scenario_id"] = reverse["scenario_id"].astype(str) + "::reverse"
    reverse[["start_stop_id", "end_stop_id"]] = reverse[
        ["end_stop_id", "start_stop_id"]
    ].to_numpy()
    reverse["hops"] = reverse["hops"].map(
        lambda value: json.dumps(list(reversed(_parse_hops(value))), ensure_ascii=False)
    )
    for column in ("edge_predictions_s", "edge_targets_s"):
        if column in reverse:
            reverse[column] = reverse[column].map(
                lambda value: json.dumps(list(reversed(json.loads(str(value)))))
            )
    return pd.concat([source, reverse], ignore_index=True)


def _canonical_hops(value: object) -> tuple[str, ...]:
    hops = tuple(_parse_hops(value))
    reverse = tuple(reversed(hops))
    return min(hops, reverse)


def _build_uniline_routes(source: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario_id", "route_type", "line", "start_stop_id", "end_stop_id", "hops",
        "n_hops", "length_real_km", "tol_prox", "target", "pred_gat", "pred_gatv2",
    ]
    result = source.loc[source["route_type"].eq("in_line"), columns].copy()
    result.insert(0, "route_id", result["scenario_id"].astype(str))
    result["transfer_count"] = 0
    return result.reset_index(drop=True)


def _build_multiline_routes(
    source: pd.DataFrame,
    transfer_max_km: float,
    segments_per_side: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compose bounded two-line routes containing exactly one transfer edge."""
    inline = source[source["route_type"].eq("in_line")].copy()
    transfers = source[source["route_type"].eq("transfer")].copy()
    eligible = transfers[transfers["length_real_km"].le(transfer_max_km)].copy()
    records: list[dict[str, object]] = []
    for transfer in eligible.itertuples(index=False):
        transfer_start = str(transfer.start_stop_id)
        transfer_end = str(transfer.end_stop_id)
        prefixes = _representative_segments(
            inline[inline["end_stop_id"].astype(str).eq(transfer_start)], segments_per_side
        )
        suffixes = _representative_segments(
            inline[inline["start_stop_id"].astype(str).eq(transfer_end)], segments_per_side
        )
        for prefix in prefixes.itertuples(index=False):
            prefix_hops = _parse_hops(prefix.hops)
            if prefix_hops[-1] != transfer_start:
                continue
            for suffix in suffixes.itertuples(index=False):
                if str(prefix.line) == str(suffix.line):
                    continue
                suffix_hops = _parse_hops(suffix.hops)
                if suffix_hops[0] != transfer_end:
                    continue
                hops = prefix_hops + suffix_hops
                if len(hops) != len(set(hops)):
                    continue
                length_km = float(prefix.length_real_km + transfer.length_real_km + suffix.length_real_km)
                bin_km = _length_bin(length_km)
                if bin_km is None:
                    continue
                line_pair = "--".join(sorted((str(prefix.line), str(suffix.line))))
                records.append({
                    "route_type": "multi_line",
                    "line": line_pair,
                    "line_pair": line_pair,
                    "first_line": str(prefix.line),
                    "second_line": str(suffix.line),
                    "start_stop_id": str(prefix.start_stop_id),
                    "end_stop_id": str(suffix.end_stop_id),
                    "hops": json.dumps(hops, ensure_ascii=False),
                    "n_hops": int(prefix.n_hops + transfer.n_hops + suffix.n_hops),
                    "length_real_km": length_km,
                    "tol_prox": float(bin_km),
                    "target": float(prefix.target + transfer.target + suffix.target),
                    "pred_gat": float(prefix.pred_gat + transfer.pred_gat + suffix.pred_gat),
                    "pred_gatv2": float(prefix.pred_gatv2 + transfer.pred_gatv2 + suffix.pred_gatv2),
                    "transfer_start_stop_id": transfer_start,
                    "transfer_end_stop_id": transfer_end,
                    "transfer_length_km": float(transfer.length_real_km),
                    "transfer_count": 1,
                    "source_prefix_id": str(prefix.scenario_id),
                    "source_prefix_hops": str(prefix.hops),
                    "source_transfer_id": str(transfer.scenario_id),
                    "source_transfer_hops": str(transfer.hops),
                    "source_suffix_id": str(suffix.scenario_id),
                    "source_suffix_hops": str(suffix.hops),
                })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No se pudieron construir rutas multi-línea con el umbral indicado.")
    result["_canonical_hops"] = result["hops"].map(_canonical_hops)
    result = result.drop_duplicates(["_canonical_hops"]).drop(columns="_canonical_hops").sort_values(
        ["tol_prox", "line_pair", "length_real_km", "hops"]
    ).reset_index(drop=True)
    result.insert(0, "route_id", [f"P0_ML_{index:05d}" for index in range(1, len(result) + 1)])
    result.insert(1, "scenario_id", result["route_id"])
    if not result["transfer_count"].eq(1).all():
        raise AssertionError("Toda ruta multi-línea debe contener una sola transferencia.")
    if result["first_line"].eq(result["second_line"]).any():
        raise AssertionError("Las rutas multi-línea deben conectar líneas distintas.")
    return result, transfers


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    if len(target) < 2 or denominator == 0:
        return float("nan")
    return float(1 - np.square(prediction - target).sum() / denominator)


def _r2_subsampling(
    route_sets: dict[str, pd.DataFrame],
    fractions: tuple[float, ...] = SUBSAMPLE_FRACTIONS,
    seeds: tuple[int, ...] = SUBSAMPLE_SEEDS,
) -> pd.DataFrame:
    """Run three fresh, bin-stratified experiments with nested sample sizes."""
    rows: list[dict[str, object]] = []
    for route_set, frame in route_sets.items():
        for experiment, seed in enumerate(seeds, start=1):
            rng = np.random.default_rng(seed)
            for bin_km in KM_BINS:
                group = frame[frame["tol_prox"].eq(float(bin_km))].reset_index(drop=True)
                if len(group) < 4:
                    raise ValueError(f"{route_set}, bin {bin_km}: muestra insuficiente ({len(group)} rutas).")
                permutation = rng.permutation(len(group))
                for fraction in fractions:
                    sample_size = max(2, int(np.floor(len(group) * fraction)))
                    sample = group.iloc[permutation[:sample_size]]
                    target = sample["target"].to_numpy(dtype=float)
                    for model, column in (("GAT", "pred_gat"), ("GATv2", "pred_gatv2")):
                        rows.append({
                            "route_set": route_set,
                            "experiment": experiment,
                            "seed": seed,
                            "fraction": fraction,
                            "model": model,
                            "bin_km": bin_km,
                            "n_available": int(len(group)),
                            "n_sample": int(sample_size),
                            "r2": _r2(target, sample[column].to_numpy(dtype=float)),
                        })
    return pd.DataFrame(rows)


def _plot_target_prediction(data_by_model: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    maximum = max(frame[["target", "pred"]].to_numpy(dtype=float).max() for frame in data_by_model.values())
    for axis, (model, data) in zip(axes, data_by_model.items()):
        plot = axis.scatter(
            data["target"], data["pred"], c=data["length_real_km"],
            cmap="viridis", vmin=0, vmax=50, s=14, alpha=0.75,
        )
        axis.plot([0, maximum], [0, maximum], "--", color="black", linewidth=1)
        axis.set_title(model)
        axis.set_xlabel("Tiempo objetivo proyectado (s)")
        axis.grid(alpha=0.2)
        fig.colorbar(plot, ax=axis, label="Kilometraje de ruta (km)")
    axes[0].set_ylabel("Tiempo predicho de ruta (s)")
    fig.suptitle("P0 Lima: objetivo frente a predicción")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bins(metrics: pd.DataFrame, output: Path) -> None:
    per_bin = metrics[metrics["bin_km"] != "global"].copy()
    per_bin["bin_km"] = pd.to_numeric(per_bin["bin_km"])
    bins = sorted(per_bin["bin_km"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    offsets = {"GAT": -0.18, "GATv2": 0.18}
    for model in ("GAT", "GATv2"):
        subset = per_bin[per_bin["model"] == model].set_index("bin_km").reindex(bins)
        axes[0].bar(np.arange(len(bins)) + offsets[model], subset["rmse_s"], width=0.36,
                    color=MODEL_COLORS[model], label=model)
        axes[1].plot(bins, subset["r2"], marker="o", color=MODEL_COLORS[model], label=model)
    axes[0].set_xticks(np.arange(len(bins)), [str(int(value)) for value in bins])
    axes[0].set_xlabel("Bin de longitud (km)")
    axes[0].set_ylabel("RMSE de ruta (s)")
    axes[0].set_title("Error absoluto por bin")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Bin de longitud (km)")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_title("Capacidad explicativa dentro de cada bin")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("P0 Lima: desempeño por longitud de ruta")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(data_by_model: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for column, (model, data) in enumerate(data_by_model.items()):
        residual = data["pred"].to_numpy(dtype=float) - data["target"].to_numpy(dtype=float)
        plot = axes[column].scatter(
            data["target"], residual, c=data["length_real_km"],
            cmap="viridis", vmin=0, vmax=50, s=14, alpha=0.75,
        )
        axes[column].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[column].set_title(f"{model}: residual frente al objetivo")
        axes[column].set_xlabel("Tiempo objetivo (s)")
        axes[column].set_ylabel("Residual pred - target (s)")
        fig.colorbar(plot, ax=axes[column], label="Kilometraje de ruta (km)")
    fig.suptitle("P0 Lima: estructura y distribución del error")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bias_structure(by_line: pd.DataFrame, by_bin: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    x = np.arange(len(LINE_ORDER))
    width = 0.38
    for offset, model in ((-width / 2, "GAT"), (width / 2, "GATv2")):
        subset = by_line[by_line["model"] == model].set_index("value").reindex(LINE_ORDER)
        axes[0].bar(x + offset, subset["mae_s"], width=width, color=MODEL_COLORS[model], label=model)
    axes[0].set_xticks(x, LINE_ORDER, rotation=45, ha="right")
    axes[0].set_ylabel("MAE (s)")
    axes[0].set_title("Error por línea")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    bins = sorted(pd.to_numeric(by_bin["value"]).unique())
    for model in ("GAT", "GATv2"):
        subset = by_bin[by_bin["model"] == model].copy()
        subset["value"] = pd.to_numeric(subset["value"])
        subset = subset.set_index("value").reindex(bins)
        axes[1].plot(bins, subset["overprediction_pct"], marker="o",
                     color=MODEL_COLORS[model], label=model)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Bin de longitud (km)")
    axes[1].set_ylabel("Rutas sobreestimadas (%)")
    axes[1].set_title("Dirección del sesgo por longitud")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("P0 Lima: sesgo por corredor y longitud")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_multiline_composition(
    multiline: pd.DataFrame,
    all_transfers: pd.DataFrame,
    transfer_max_km: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    counts = multiline["tol_prox"].value_counts().reindex(KM_BINS, fill_value=0)
    axes[0].bar([str(value) for value in KM_BINS], counts.to_numpy(), color="#4c78a8")
    axes[0].set_xlabel("Bin de longitud (km)")
    axes[0].set_ylabel("Rutas multi-línea")
    axes[0].set_title("Cobertura por bin")
    axes[0].grid(axis="y", alpha=0.25)
    pair_counts = multiline["line_pair"].value_counts().head(12).sort_values()
    axes[1].barh(pair_counts.index, pair_counts.to_numpy(), color="#72b7b2")
    axes[1].set_xlabel("Rutas")
    axes[1].set_title("Pares de líneas más representados")
    axes[1].grid(axis="x", alpha=0.25)
    axes[2].hist(all_transfers["length_real_km"], bins=12, color="#f58518", alpha=0.85)
    axes[2].axvline(transfer_max_km, color="black", linestyle="--", linewidth=1.5,
                    label=f"umbral = {transfer_max_km:.2f} km")
    axes[2].set_xlabel("Longitud de arista de transferencia (km)")
    axes[2].set_ylabel("Aristas físicas únicas")
    axes[2].set_title("Filtro de cercanía")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P0 Lima: composición de las rutas multi-línea")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_r2_subsampling(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(KM_BINS)))
    for row, route_set in enumerate(("uni_line", "multi_line")):
        for column, model in enumerate(("GAT", "GATv2")):
            axis = axes[row, column]
            subset = results[(results["route_set"] == route_set) & (results["model"] == model)]
            for color, bin_km in zip(colors, KM_BINS):
                values = subset[subset["bin_km"] == bin_km]
                summary = values.groupby("fraction")["r2"].agg(["mean", "min", "max"]).reindex(SUBSAMPLE_FRACTIONS)
                x = np.asarray(SUBSAMPLE_FRACTIONS) * 100
                axis.plot(x, summary["mean"], marker="o", color=color, label=f"{bin_km} km")
                axis.fill_between(x, summary["min"], summary["max"], color=color, alpha=0.12)
            axis.axhline(0, color="black", linewidth=0.8)
            route_label = "Uni-línea" if route_set == "uni_line" else "Multi-línea"
            axis.set_title(f"{route_label} - {model}")
            axis.set_ylabel("$R^2$")
            axis.grid(alpha=0.25)
            if column == 1:
                axis.legend(title="Bin", ncol=2, fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Fracción aleatoria por bin (%)")
        axis.set_xticks([25, 50, 75])
    fig.suptitle("P0 Lima: sensibilidad de $R^2$ a tres submuestras aleatorias")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_p0_report(
    outputs_dir: Path,
    gat_path: Path,
    gatv2_path: Path,
    transfer_max_km: float = DEFAULT_TRANSFER_MAX_KM,
    segments_per_side: int = DEFAULT_SEGMENTS_PER_SIDE,
) -> dict[str, Path]:
    outputs_dir = Path(outputs_dir)
    result_dir = outputs_dir / "lima"
    figure_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths_by_model = {"GAT": Path(gat_path), "GATv2": Path(gatv2_path)}
    data_by_model = {model: load_valid_predictions(path, "pred") for model, path in paths_by_model.items()}
    _assert_same_routes(data_by_model["GAT"], data_by_model["GATv2"])
    metrics = pd.concat(
        [evaluate_predictions(path, model, "pred") for model, path in paths_by_model.items()],
        ignore_index=True,
    )
    global_diagnostics = _diagnostic_table(data_by_model, None)
    by_bin = _diagnostic_table(data_by_model, "tol_prox")
    by_line = _diagnostic_table(data_by_model, "line")
    by_route_type = _diagnostic_table(data_by_model, "route_type")
    top_errors = _top_errors(data_by_model)
    combined = _combined_source_rows(data_by_model["GAT"], data_by_model["GATv2"])
    uniline = _build_uniline_routes(combined)
    composition_source = _bidirectional_route_view(combined)
    multiline, _ = _build_multiline_routes(
        composition_source, transfer_max_km, segments_per_side
    )
    unique_transfers = combined[combined["route_type"].eq("transfer")].copy()
    subsampling = _r2_subsampling({"uni_line": uniline, "multi_line": multiline})
    subsampling_summary = (
        subsampling.groupby(["route_set", "fraction", "model", "bin_km"], as_index=False)["r2"]
        .agg(r2_mean="mean", r2_sd="std", r2_min="min", r2_max="max")
    )
    paths = {
        "metrics": result_dir / "p0_metrics.csv",
        "global_diagnostics": result_dir / "p0_diagnostics_global.csv",
        "by_bin": result_dir / "p0_diagnostics_by_bin.csv",
        "by_line": result_dir / "p0_diagnostics_by_line.csv",
        "by_route_type": result_dir / "p0_diagnostics_by_route_type.csv",
        "top_errors": result_dir / "p0_top_errors.csv",
        "uniline_routes": result_dir / "p0_uniline_routes.csv",
        "multiline_routes": result_dir / "p0_multiline_routes.csv",
        "r2_subsampling": result_dir / "p0_r2_subsampling.csv",
        "r2_subsampling_summary": result_dir / "p0_r2_subsampling_summary.csv",
        "audit": result_dir / "p0_data_audit.json",
        "scatter": figure_dir / "p0_target_vs_prediction.png",
        "bins": figure_dir / "p0_performance_by_bin.png",
        "residuals": figure_dir / "p0_residual_diagnostics.png",
        "bias_structure": figure_dir / "p0_bias_structure.png",
        "multiline_composition": figure_dir / "p0_multiline_composition.png",
        "r2_subsampling_figure": figure_dir / "p0_r2_subsampling.png",
    }
    for table, key in (
        (metrics, "metrics"), (global_diagnostics, "global_diagnostics"),
        (by_bin, "by_bin"), (by_line, "by_line"),
        (by_route_type, "by_route_type"), (top_errors, "top_errors"),
        (uniline, "uniline_routes"), (multiline, "multiline_routes"),
        (subsampling, "r2_subsampling"), (subsampling_summary, "r2_subsampling_summary"),
    ):
        table.to_csv(paths[key], index=False, float_format="%.6f")

    audit = {
        "route_identity": "canonical node sequence; route equals its reverse",
        "gat_path": str(paths_by_model["GAT"]),
        "gat_sha256": _sha256(paths_by_model["GAT"]),
        "gatv2_path": str(paths_by_model["GATv2"]),
        "gatv2_sha256": _sha256(paths_by_model["GATv2"]),
        "unique_routes": int(len(combined)),
        "in_line_routes": int(combined["route_type"].eq("in_line").sum()),
        "transfer_routes": int(combined["route_type"].eq("transfer").sum()),
        "derived_multiline_routes": int(len(multiline)),
        "transfer_max_km": float(transfer_max_km),
        "segments_per_side": int(segments_per_side),
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_target_prediction(data_by_model, paths["scatter"])
    _plot_bins(metrics, paths["bins"])
    _plot_residuals(data_by_model, paths["residuals"])
    _plot_bias_structure(by_line, by_bin, paths["bias_structure"])
    _plot_multiline_composition(
        multiline, unique_transfers, transfer_max_km, paths["multiline_composition"]
    )
    _plot_r2_subsampling(subsampling, paths["r2_subsampling_figure"])

    return paths
