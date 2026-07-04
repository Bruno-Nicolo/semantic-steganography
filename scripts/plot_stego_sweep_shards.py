from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("outputs/stego_sweep_shards/csv_exports")
DEFAULT_OUTPUT_DIR = Path("outputs/stego_sweep_shards/plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scientific plots for the sharded stego sweep.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-class-images", type=int, default=5)
    return parser.parse_args()


def load_results(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["source_csv"] = file.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["status"] == "success"].copy()

    numeric_columns = [
        "embedding_strength",
        "payload_bits",
        "payload_bits_capacity",
        "payload_retention_ratio",
        "payload_success_ratio",
        "payload_truncated",
        "roi_area_ratio",
        "BER",
        "PSNR_roi",
        "PSNR_full",
        "SSIM_roi",
        "exact_match",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["requested_error_rate"] = 1.0 - data["payload_success_ratio"]
    data["psnr_roi_is_finite"] = np.isfinite(data["PSNR_roi"])
    data["PSNR_roi_finite"] = data["PSNR_roi"].where(data["psnr_roi_is_finite"])
    return data


def mean_ci(values: pd.Series) -> pd.Series:
    values = values.dropna().astype(float)
    n = len(values)
    mean = values.mean() if n else np.nan
    ci95 = 1.96 * values.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return pd.Series({"mean": mean, "ci95": ci95, "n": n})


def aggregate_metric(data: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    # First average repeated technical configurations within each image/condition.
    image_cols = group_cols + ["image_id"]
    per_image = data.groupby(image_cols, dropna=False)[metric].mean().reset_index()
    summary = per_image.groupby(group_cols, dropna=False)[metric].apply(mean_ci).unstack().reset_index()
    return summary


def save_table(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(output_dir / f"{name}.csv", index=False)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=300)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_ber_vs_delta(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[data["embedding_strength_mode"] == "absolute"].copy()
    summary = aggregate_metric(subset, ["decoder_type", "attack_type", "embedding_strength"], "BER")
    save_table(summary, output_dir, "ber_vs_delta_absolute_summary")

    attacks = ["none", "gaussian_noise", "jpeg_compression", "gaussian_blur"]
    decoders = [decoder for decoder in ["non_blind", "blind"] if decoder in summary["decoder_type"].unique()]
    fig, axes = plt.subplots(len(decoders), len(attacks), figsize=(17, 7), sharex=True, sharey=True)
    if len(decoders) == 1:
        axes = np.array([axes])

    for row, decoder in enumerate(decoders):
        for col, attack in enumerate(attacks):
            ax = axes[row, col]
            part = summary[(summary["decoder_type"] == decoder) & (summary["attack_type"] == attack)].sort_values("embedding_strength")
            if not part.empty:
                ax.errorbar(part["embedding_strength"], part["mean"], yerr=part["ci95"], marker="o", capsize=3)
            ax.set_title(f"{decoder} | {attack}")
            ax.set_ylim(-0.02, 0.55)
            ax.grid(True, alpha=0.25)
            if row == len(decoders) - 1:
                ax.set_xlabel("Delta assoluto")
            if col == 0:
                ax.set_ylabel("BER medio per immagine")

    fig.suptitle("BER al variare di Delta assoluto", y=1.02)
    save_figure(fig, output_dir, "01_ber_vs_delta_absolute")


def plot_psnr_vs_delta(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[(data["embedding_strength_mode"] == "absolute") & (data["attack_type"] == "none")].copy()
    finite_summary = aggregate_metric(subset, ["embedding_strength"], "PSNR_roi_finite")
    inf_rate = aggregate_metric(subset.assign(psnr_inf=(~subset["psnr_roi_is_finite"]).astype(float)), ["embedding_strength"], "psnr_inf")
    save_table(finite_summary, output_dir, "psnr_roi_finite_vs_delta_absolute_summary")
    save_table(inf_rate, output_dir, "psnr_roi_inf_rate_vs_delta_absolute_summary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    finite_summary = finite_summary.sort_values("embedding_strength")
    axes[0].errorbar(finite_summary["embedding_strength"], finite_summary["mean"], yerr=finite_summary["ci95"], marker="o", capsize=3)
    axes[0].set_title("PSNR ROI finito")
    axes[0].set_xlabel("Delta assoluto")
    axes[0].set_ylabel("PSNR ROI [dB]")
    axes[0].grid(True, alpha=0.25)

    inf_rate = inf_rate.sort_values("embedding_strength")
    axes[1].plot(inf_rate["embedding_strength"], inf_rate["mean"], marker="o")
    axes[1].set_title("Quota PSNR ROI infinito")
    axes[1].set_xlabel("Delta assoluto")
    axes[1].set_ylabel("Frazione")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Impercettibilita al variare di Delta, senza attacco", y=1.02)
    save_figure(fig, output_dir, "02_psnr_vs_delta_absolute_no_attack")


def plot_payload_curves(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[(data["embedding_strength_mode"] == "absolute") & (data["attack_type"] == "none")].copy()
    deltas = [10.0, 20.0, 40.0, 80.0]

    for metric, ylabel, name in [
        ("BER", "BER medio per immagine", "03_ber_vs_payload_no_attack"),
        ("payload_retention_ratio", "Payload retention ratio", "04_retention_vs_payload"),
        ("requested_error_rate", "Errore sul messaggio richiesto", "05_requested_error_vs_payload"),
        ("payload_truncated", "Frazione payload troncati", "06_truncation_rate_vs_payload"),
    ]:
        summary = aggregate_metric(subset, ["embedding_strength", "payload_bits"], metric)
        save_table(summary, output_dir, f"{name}_summary")
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for delta in deltas:
            part = summary[summary["embedding_strength"] == delta].sort_values("payload_bits")
            if part.empty:
                continue
            ax.errorbar(part["payload_bits"], part["mean"], yerr=part["ci95"], marker="o", capsize=3, label=f"Delta={delta:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(subset["payload_bits"].dropna().unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Payload richiesto [bit]")
        ax.set_ylabel(ylabel)
        if metric != "BER":
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(title="Delta assoluto")
        save_figure(fig, output_dir, name)


def plot_yolo_class_analysis(data: pd.DataFrame, output_dir: Path, min_class_images: int) -> None:
    subset = data[
        (data["roi_strategy"] == "largest")
        & (data["attack_type"] == "none")
        & (data["embedding_strength_mode"] == "absolute")
        & (data["embedding_strength"] == 40.0)
        & (data["decoder_type"] == "non_blind")
        & data["roi_class_name"].notna()
    ].copy()

    class_counts = subset.groupby("roi_class_name")["image_id"].nunique()
    kept_classes = class_counts[class_counts >= min_class_images].index
    subset = subset[subset["roi_class_name"].isin(kept_classes)]
    if subset.empty:
        return

    image_level = subset.groupby(["roi_class_name", "image_id"], dropna=False).agg(
        BER=("BER", "mean"),
        PSNR_roi_finite=("PSNR_roi_finite", "mean"),
        payload_bits_capacity=("payload_bits_capacity", "mean"),
        roi_area_ratio=("roi_area_ratio", "mean"),
    ).reset_index()
    summary = image_level.groupby("roi_class_name").agg(
        images=("image_id", "nunique"),
        BER_mean=("BER", "mean"),
        BER_ci95=("BER", lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        PSNR_roi_mean=("PSNR_roi_finite", "mean"),
        capacity_mean=("payload_bits_capacity", "mean"),
        roi_area_ratio_mean=("roi_area_ratio", "mean"),
    ).sort_values("images", ascending=False).reset_index()
    save_table(summary, output_dir, "yolo_class_largest_delta40_summary")

    labels = [f"{row.roi_class_name}\n(n={int(row.images)})" for row in summary.itertuples()]
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(summary) * 1.2), 4.8))
    axes[0].bar(x, summary["BER_mean"], yerr=summary["BER_ci95"], capsize=3)
    axes[0].set_title("BER per classe YOLO")
    axes[0].set_ylabel("BER medio per immagine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, summary["capacity_mean"])
    axes[1].set_title("Capacita per classe YOLO")
    axes[1].set_ylabel("Capacita media [bit]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Analisi per classe YOLO, ROI largest, Delta=40, non-blind", y=1.04)
    save_figure(fig, output_dir, "07_yolo_class_ber_capacity")

    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.clip(image_level["payload_bits_capacity"].to_numpy(), 10, 250)
    ax.scatter(image_level["roi_area_ratio"], image_level["payload_bits_capacity"], s=sizes, alpha=0.55)
    ax.set_xlabel("Area ROI / area immagine")
    ax.set_ylabel("Capacita payload [bit]")
    ax.set_title("Relazione tra dimensione ROI e capacita")
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir, "08_roi_area_vs_capacity")


def plot_tradeoff(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[
        (data["embedding_strength_mode"] == "absolute")
        & (data["attack_type"] == "none")
        & data["psnr_roi_is_finite"]
    ].copy()
    image_level = subset.groupby(["embedding_strength", "image_id"], dropna=False).agg(
        BER=("BER", "mean"),
        PSNR_roi=("PSNR_roi_finite", "mean"),
    ).reset_index()
    save_table(image_level, output_dir, "ber_psnr_tradeoff_image_level")

    fig, ax = plt.subplots(figsize=(7, 5))
    for delta, part in image_level.groupby("embedding_strength"):
        ax.scatter(part["PSNR_roi"], part["BER"], alpha=0.45, label=f"Delta={delta:g}")
    ax.set_xlabel("PSNR ROI [dB]")
    ax.set_ylabel("BER medio per immagine")
    ax.set_title("Trade-off robustezza/impercettibilita, senza attacco")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Delta")
    save_figure(fig, output_dir, "09_ber_psnr_tradeoff")


def plot_status_breakdown(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        return
    full = pd.concat([pd.read_csv(file, usecols=["status"]) for file in files], ignore_index=True)
    counts = full["status"].fillna("unknown").value_counts().sort_values(ascending=True)
    summary = counts.reset_index()
    summary.columns = ["status", "rows"]
    save_table(summary, output_dir, "legacy_status_breakdown_summary")

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(counts) * 0.6)))
    ax.barh(counts.index.astype(str), counts.values, color="#4C78A8")
    ax.set_xlabel("Numero di righe")
    ax.set_title("Distribuzione degli stati sperimentali")
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, output_dir, "10_status_breakdown")


def plot_roi_svd_heatmaps(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[
        (data["embedding_strength_mode"] == "absolute")
        & (data["embedding_strength"] == 40.0)
    ].copy()
    if subset.empty:
        return

    metrics = [
        ("BER", "BER medio", "legacy_heatmap_ber"),
        ("payload_success_ratio", "Payload success ratio", "legacy_heatmap_payload_success"),
    ]
    for metric, label, name in metrics:
        summary = aggregate_metric(subset, ["decoder_type", "attack_type", "roi_strategy", "svd_band"], metric)
        save_table(summary, output_dir, f"{name}_summary")
        for decoder in sorted(summary["decoder_type"].dropna().unique()):
            part = summary[summary["decoder_type"] == decoder]
            attacks = [attack for attack in ["none", "gaussian_noise", "jpeg_compression", "gaussian_blur"] if attack in part["attack_type"].unique()]
            if not attacks:
                continue
            fig, axes = plt.subplots(1, len(attacks), figsize=(4.8 * len(attacks), 5.2), sharey=True)
            if len(attacks) == 1:
                axes = [axes]
            for ax, attack in zip(axes, attacks):
                pivot = (
                    part[part["attack_type"] == attack]
                    .pivot_table(index="roi_strategy", columns="svd_band", values="mean", aggfunc="mean")
                    .reindex(index=["full_image", "largest", "smallest"], columns=["low_energy", "mid_energy", "high_energy"])
                )
                values = pivot.to_numpy(dtype=float)
                image = ax.imshow(values, aspect="auto", cmap="magma_r" if metric == "BER" else "viridis")
                ax.set_title(attack)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                for row in range(values.shape[0]):
                    for col in range(values.shape[1]):
                        value = values[row, col]
                        text = "n/a" if np.isnan(value) else f"{value:.3f}"
                        ax.text(col, row, text, ha="center", va="center", color="white", fontsize=8)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{label} per ROI e banda SVD, Delta=40, decoder={decoder}", y=1.03)
            save_figure(fig, output_dir, f"11_{name}_{decoder}")


def plot_attack_degradation(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[(data["embedding_strength_mode"] == "absolute") & (data["embedding_strength"] == 40.0)].copy()
    keys = ["image_id", "roi_strategy", "svd_band", "decoder_type", "payload_bits"]
    baseline = subset[subset["attack_type"] == "none"][keys + ["BER", "payload_success_ratio"]].copy()
    if baseline.empty:
        return
    baseline = baseline.rename(columns={"BER": "BER_none", "payload_success_ratio": "payload_success_ratio_none"})
    attacked = subset[subset["attack_type"] != "none"].copy()
    merged = attacked.merge(baseline, on=keys, how="inner")
    if merged.empty:
        return
    merged["delta_BER"] = merged["BER"] - merged["BER_none"]
    merged["delta_payload_success_ratio"] = merged["payload_success_ratio"] - merged["payload_success_ratio_none"]

    summary = aggregate_metric(merged, ["attack_type", "decoder_type"], "delta_BER")
    save_table(summary, output_dir, "legacy_attack_degradation_delta_ber_summary")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    attacks = [attack for attack in ["gaussian_noise", "jpeg_compression", "gaussian_blur"] if attack in summary["attack_type"].unique()]
    x = np.arange(len(attacks))
    for offset, decoder in zip([-width / 2, width / 2], ["non_blind", "blind"]):
        part = summary[summary["decoder_type"] == decoder].set_index("attack_type").reindex(attacks)
        if part.empty:
            continue
        ax.bar(x + offset, part["mean"], width, yerr=part["ci95"], capsize=3, label=decoder)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_ylabel("Incremento BER rispetto a nessun attacco")
    ax.set_title("Degradazione media dovuta agli attacchi, Delta=40")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Decoder")
    save_figure(fig, output_dir, "12_attack_degradation_delta_ber")


def plot_category_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[
        (data["embedding_strength_mode"] == "absolute")
        & (data["embedding_strength"] == 40.0)
        & (data["attack_type"] == "none")
        & (data["decoder_type"] == "non_blind")
    ].copy()
    categories = ["roi_strategy", "svd_band"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, category in zip(axes, categories):
        summary = aggregate_metric(subset, [category], "BER").sort_values("mean")
        save_table(summary, output_dir, f"legacy_category_{category}_ber_summary")
        ax.bar(summary[category].astype(str), summary["mean"], yerr=summary["ci95"], capsize=3, color="#72B7B2")
        ax.set_title(category)
        ax.set_ylabel("BER medio per immagine")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Confronto controllato per categoria, Delta=40, nessun attacco, non-blind", y=1.03)
    save_figure(fig, output_dir, "13_category_comparison_controlled")


def write_dataset_summary(data: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "success_rows": len(data),
        "images": data["image_id"].nunique(),
        "roi_strategies": sorted(data["roi_strategy"].dropna().unique().tolist()),
        "svd_bands": sorted(data["svd_band"].dropna().unique().tolist()),
        "decoders": sorted(data["decoder_type"].dropna().unique().tolist()),
        "attack_types": sorted(data["attack_type"].dropna().unique().tolist()),
        "payload_bits": sorted(data["payload_bits"].dropna().unique().astype(int).tolist()),
        "absolute_deltas": sorted(data.loc[data["embedding_strength_mode"] == "absolute", "embedding_strength"].dropna().unique().tolist()),
        "proportional_deltas": sorted(data.loc[data["embedding_strength_mode"] == "proportional_singular", "embedding_strength"].dropna().unique().tolist()),
    }
    pd.Series(summary).to_json(output_dir / "dataset_summary.json", indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.input_dir)
    write_dataset_summary(data, args.output_dir)
    plot_ber_vs_delta(data, args.output_dir)
    plot_psnr_vs_delta(data, args.output_dir)
    plot_payload_curves(data, args.output_dir)
    plot_yolo_class_analysis(data, args.output_dir, args.min_class_images)
    plot_tradeoff(data, args.output_dir)
    plot_status_breakdown(args.input_dir, args.output_dir)
    plot_roi_svd_heatmaps(data, args.output_dir)
    plot_attack_degradation(data, args.output_dir)
    plot_category_comparison(data, args.output_dir)
    print(f"Saved plots and summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
