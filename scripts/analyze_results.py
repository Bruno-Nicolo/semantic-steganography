from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXCLUDED_CONFIG_COLUMNS = {
    "config_coco_root",
    "config_output_dir",
    "config_save_images",
    "config_save_roi_debug",
    "config_seed",
    "config_payload_seed",
    "config_yolo_model",
    "config_split",
    "config_attacks",
    "config_jpeg_qualities",
    "config_noise_sigmas",
    "config_blur_kernels",
    "config_roi_strategies",
    "config_svd_bands",
    "config_decoders",
}

SUCCESS_METRICS = [
    "BER",
    "PSNR_full",
    "PSNR_roi",
    "SSIM_full",
    "SSIM_roi",
    "character_accuracy",
    "payload_retention_ratio",
    "payload_success_ratio",
    "payload_bits_embedded",
    "payload_bits_capacity",
    "payload_bits_dropped",
    "bpp_roi",
    "bpp_image",
    "yolo_time_ms",
    "embedding_time_ms",
    "extraction_time_ms",
    "svd_time_ms",
    "attack_time_ms",
    "total_time_ms",
    "svd_reconstruction_error",
]


@dataclass(slots=True)
class AnalysisContext:
    frame: pd.DataFrame
    run_dirs: list[Path]
    varying_config_columns: list[str]
    embedding_keys: list[str]
    extraction_keys: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate comparisons.")
    parser.add_argument(
        "run_paths",
        nargs="*",
        help="Run directories or results.csv files. If omitted, all runs under outputs/ are discovered automatically.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory used for automatic run discovery.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("outputs") / "analysis",
        help="Destination directory for tables, conclusions, and plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Maximum number of configurations shown in ranking plots and markdown tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = discover_run_dirs(args.run_paths, args.outputs_root)
    if not run_dirs:
        raise SystemExit("No run directories with results.csv were found.")

    analysis_dir = args.analysis_dir
    plots_dir = analysis_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    context = load_context(run_dirs)

    consolidated_path = analysis_dir / "consolidated_results.csv"
    context.frame.to_csv(consolidated_path, index=False)

    embedding_summary = summarize_groups(context.frame, context.embedding_keys)
    extraction_frame = context.frame[context.frame["decoder_type"].ne("not_reached")].copy()
    extraction_summary = summarize_groups(extraction_frame, context.extraction_keys)
    attack_summary = summarize_groups(extraction_frame, context.extraction_keys + ["attack_type"])

    category_summary = build_category_summary(extraction_frame)

    embedding_summary.to_csv(analysis_dir / "embedding_summary.csv", index=False)
    extraction_summary.to_csv(analysis_dir / "extraction_summary.csv", index=False)
    attack_summary.to_csv(analysis_dir / "attack_summary.csv", index=False)
    category_summary.to_csv(analysis_dir / "category_summary.csv", index=False)

    write_key_metrics_json(analysis_dir, context, embedding_summary, extraction_summary, attack_summary)
    write_conclusions_markdown(analysis_dir, context, embedding_summary, extraction_summary, attack_summary, args.top_n)

    generate_plots(plots_dir, extraction_summary, attack_summary, category_summary, context.extraction_keys, args.top_n)

    print(f"Analyzed {len(run_dirs)} run(s).")
    print(f"Consolidated results: {consolidated_path}")
    print(f"Conclusions: {analysis_dir / 'conclusions.md'}")
    print(f"Plots directory: {plots_dir}")


def discover_run_dirs(run_paths: list[str], outputs_root: Path) -> list[Path]:
    if run_paths:
        candidates = [Path(item) for item in run_paths]
    else:
        if not outputs_root.exists():
            return []
        candidates = sorted(outputs_root.iterdir())

    run_dirs: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.name == "results.csv":
            run_dirs.append(resolved.parent)
            continue
        if not resolved.is_dir():
            continue
        if (resolved / "results.csv").exists():
            run_dirs.append(resolved)
            continue
        for nested in sorted(resolved.rglob("results.csv")):
            if "analysis" in nested.parts:
                continue
            run_dirs.append(nested.parent)

    unique_dirs: list[Path] = []
    seen: set[Path] = set()
    for run_dir in run_dirs:
        if run_dir not in seen:
            unique_dirs.append(run_dir)
            seen.add(run_dir)
    return unique_dirs


def load_context(run_dirs: list[Path]) -> AnalysisContext:
    frames: list[pd.DataFrame] = []
    config_columns: set[str] = set()

    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "results.csv")
        frame["source_run_dir"] = str(run_dir)
        frame["run_name"] = run_dir.name

        config_path = run_dir / "config.json"
        if config_path.exists():
            config_values = flatten_config(json.loads(config_path.read_text(encoding="utf-8")))
            for key, value in config_values.items():
                column = f"config_{key}"
                frame[column] = value
                config_columns.add(column)

        frames.append(frame)

    if not frames:
        raise SystemExit("No readable results.csv files were found.")

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame = standardize_frame(frame)

    varying_config_columns = select_varying_config_columns(frame, sorted(config_columns))
    embedding_keys = ["dataset", "roi_strategy", "svd_band", *varying_config_columns]
    extraction_keys = [*embedding_keys, "decoder_type"]
    return AnalysisContext(frame, run_dirs, varying_config_columns, embedding_keys, extraction_keys)


def flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else key
            flattened.update(flatten_config(item, nested_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: "|".join("" if item is None else str(item) for item in value)}
    return {prefix: value}


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in ["dataset", "roi_strategy", "svd_band", "decoder_type", "attack_type", "status"]:
        if column not in working.columns:
            working[column] = None

    working["attack_type"] = working["attack_type"].fillna("not_reached")
    working["decoder_type"] = working["decoder_type"].fillna("not_reached")
    working["roi_strategy"] = working["roi_strategy"].fillna("not_reached")
    working["svd_band"] = working["svd_band"].fillna("not_reached")
    working["status"] = working["status"].fillna("unknown")

    working["exact_match_bool"] = working["exact_match"].map(parse_bool).fillna(False)
    working["is_success"] = working["status"].eq("success")
    working["is_failure"] = ~working["is_success"]

    for metric in SUCCESS_METRICS + ["embedding_strength", "payload_bits"]:
        if metric in working.columns:
            working[metric] = pd.to_numeric(working[metric], errors="coerce")
    return working


def parse_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def select_varying_config_columns(frame: pd.DataFrame, config_columns: list[str]) -> list[str]:
    varying: list[str] = []
    for column in config_columns:
        if column in EXCLUDED_CONFIG_COLUMNS:
            continue
        distinct = frame[column].drop_duplicates().shape[0]
        if distinct > 1:
            varying.append(column)
    return varying


def summarize_groups(frame: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_keys + ["samples_total", "score"])

    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(group_keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        success = group[group["is_success"]]

        row.update(
            {
                "samples_total": int(len(group)),
                "success_count": int(group["is_success"].sum()),
                "failure_count": int(group["is_failure"].sum()),
                "success_rate": safe_mean(group["is_success"]),
                "failure_rate": safe_mean(group["is_failure"]),
                "exact_match_rate_all": safe_mean(group["exact_match_bool"]),
                "exact_match_rate_success": safe_mean(success["exact_match_bool"]),
            }
        )

        for metric in SUCCESS_METRICS:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = safe_mean(success[metric])
            row[f"{metric}_median"] = safe_median(success[metric])

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["score"] = compute_composite_score(summary)
    return summary.sort_values(["score", "exact_match_rate_all", "success_rate"], ascending=[False, False, False]).reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def safe_median(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.median())


def compute_composite_score(summary: pd.DataFrame) -> pd.Series:
    metrics = {
        "success_rate": (0.20, True),
        "exact_match_rate_all": (0.30, True),
        "payload_success_ratio_mean": (0.15, True),
        "BER_mean": (0.15, False),
        "PSNR_roi_mean": (0.10, True),
        "SSIM_roi_mean": (0.05, True),
        "total_time_ms_mean": (0.05, False),
    }

    score = pd.Series(0.0, index=summary.index, dtype=float)
    for column, (weight, higher_is_better) in metrics.items():
        if column not in summary.columns:
            continue
        normalized = normalize_metric(summary[column], higher_is_better=higher_is_better)
        score = score + weight * normalized
    return score.round(6)


def normalize_metric(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype=float)

    finite = values.dropna()
    minimum = finite.min()
    maximum = finite.max()
    if math.isclose(maximum, minimum):
        normalized = pd.Series(1.0, index=series.index, dtype=float)
        normalized[values.isna()] = 0.0
        return normalized

    if higher_is_better:
        normalized = (values - minimum) / (maximum - minimum)
    else:
        normalized = (maximum - values) / (maximum - minimum)
    return normalized.fillna(0.0)


def build_category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in ["roi_strategy", "svd_band", "decoder_type", "attack_type"]:
        if category not in frame.columns:
            continue
        for value, group in frame.groupby(category, dropna=False):
            success = group[group["is_success"]]
            rows.append(
                {
                    "category": category,
                    "value": value,
                    "samples_total": len(group),
                    "success_rate": safe_mean(group["is_success"]),
                    "exact_match_rate_all": safe_mean(group["exact_match_bool"]),
                    "BER_mean": safe_mean(success["BER"]) if "BER" in success.columns else float("nan"),
                    "PSNR_roi_mean": safe_mean(success["PSNR_roi"]) if "PSNR_roi" in success.columns else float("nan"),
                    "SSIM_roi_mean": safe_mean(success["SSIM_roi"]) if "SSIM_roi" in success.columns else float("nan"),
                    "total_time_ms_mean": safe_mean(success["total_time_ms"]) if "total_time_ms" in success.columns else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_key_metrics_json(
    analysis_dir: Path,
    context: AnalysisContext,
    embedding_summary: pd.DataFrame,
    extraction_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
) -> None:
    payload = {
        "runs_analyzed": [str(path) for path in context.run_dirs],
        "varying_config_columns": context.varying_config_columns,
        "rows_total": int(len(context.frame)),
        "success_rows": int(context.frame["is_success"].sum()),
        "failure_rows": int(context.frame["is_failure"].sum()),
        "best_embedding_configuration": dataframe_head_record(embedding_summary),
        "best_extraction_configuration": dataframe_head_record(extraction_summary),
        "best_per_attack": top_record_per_group(attack_summary, "attack_type"),
    }
    (analysis_dir / "analysis_overview.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataframe_head_record(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    return jsonify_row(frame.iloc[0].to_dict())


def top_record_per_group(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    if frame.empty or column not in frame.columns:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for value, group in frame.groupby(column, dropna=False):
        best = group.sort_values(["score", "exact_match_rate_all"], ascending=[False, False]).iloc[0]
        records[str(value)] = jsonify_row(best.to_dict())
    return records


def jsonify_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, Path):
            cleaned[key] = str(value)
        else:
            cleaned[key] = value
    return cleaned


def write_conclusions_markdown(
    analysis_dir: Path,
    context: AnalysisContext,
    embedding_summary: pd.DataFrame,
    extraction_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    top_n: int,
) -> None:
    lines: list[str] = []
    total_rows = len(context.frame)
    success_rows = int(context.frame["is_success"].sum())
    failure_rows = int(context.frame["is_failure"].sum())

    lines.append("# Analysis Conclusions")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(context.run_dirs)}")
    lines.append(f"- Total result rows: {total_rows}")
    lines.append(f"- Success rows: {success_rows}")
    lines.append(f"- Failure rows: {failure_rows}")
    lines.append(f"- Global success rate: {success_rows / max(total_rows, 1):.2%}")
    if context.varying_config_columns:
        lines.append(f"- Varying run-level parameters: {', '.join(context.varying_config_columns)}")
    lines.append("")

    if not embedding_summary.empty:
        lines.append("## Best Embedding Configuration")
        lines.append("")
        lines.extend(format_best_configuration(embedding_summary.iloc[0]))
        lines.append("")

    if not extraction_summary.empty:
        lines.append("## Best Extraction Configuration")
        lines.append("")
        lines.extend(format_best_configuration(extraction_summary.iloc[0]))
        lines.append("")

        lines.append("## Trade-Off Highlights")
        lines.append("")
        lines.extend(format_highlight_line("Most robust", best_by_metric(extraction_summary, "exact_match_rate_all", ascending=False)))
        lines.extend(format_highlight_line("Lowest BER", best_by_metric(extraction_summary, "BER_mean", ascending=True)))
        lines.extend(format_highlight_line("Best image quality", best_by_metric(extraction_summary, "PSNR_roi_mean", ascending=False)))
        lines.extend(format_highlight_line("Fastest extraction", best_by_metric(extraction_summary, "total_time_ms_mean", ascending=True)))
        lines.append("")

        lines.append(f"## Top {min(top_n, len(extraction_summary))} Extraction Configurations")
        lines.append("")
        top_frame = extraction_summary.head(top_n)
        lines.extend(
            markdown_table(
                top_frame,
                [
                    *context.extraction_keys,
                    "samples_total",
                    "success_rate",
                    "exact_match_rate_all",
                    "BER_mean",
                    "PSNR_roi_mean",
                    "SSIM_roi_mean",
                    "total_time_ms_mean",
                    "score",
                ],
            )
        )
        lines.append("")

    if not attack_summary.empty:
        lines.append("## Best Configuration Per Attack")
        lines.append("")
        for attack_type, group in attack_summary.groupby("attack_type", dropna=False):
            best = group.sort_values(["score", "exact_match_rate_all"], ascending=[False, False]).iloc[0]
            lines.append(f"### {attack_type}")
            lines.extend(format_best_configuration(best))
            lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `consolidated_results.csv`: all runs merged into one table")
    lines.append("- `embedding_summary.csv`: summary by ROI strategy and SVD band")
    lines.append("- `extraction_summary.csv`: summary by extraction configuration")
    lines.append("- `attack_summary.csv`: summary by extraction configuration and attack")
    lines.append("- `category_summary.csv`: macro comparison by ROI strategy, SVD band, decoder, attack")
    lines.append("- `analysis_overview.json`: machine-readable overview of the best configurations")
    lines.append("- `plots/`: comparison charts and heatmaps")

    (analysis_dir / "conclusions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_by_metric(frame: pd.DataFrame, metric: str, *, ascending: bool) -> pd.Series | None:
    if frame.empty or metric not in frame.columns:
        return None
    valid = frame[frame[metric].notna()]
    if valid.empty:
        return None
    return valid.sort_values(metric, ascending=ascending).iloc[0]


def format_best_configuration(row: pd.Series) -> list[str]:
    label = configuration_label(row)
    return [
        f"- Configuration: `{label}`",
        f"- Score: {format_number(row.get('score'))}",
        f"- Success rate: {format_percent(row.get('success_rate'))}",
        f"- Exact match rate: {format_percent(row.get('exact_match_rate_all'))}",
        f"- Mean BER: {format_number(row.get('BER_mean'))}",
        f"- Mean PSNR ROI: {format_number(row.get('PSNR_roi_mean'))}",
        f"- Mean SSIM ROI: {format_number(row.get('SSIM_roi_mean'))}",
        f"- Mean total time: {format_number(row.get('total_time_ms_mean'))} ms",
    ]


def format_highlight_line(title: str, row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    return [f"- {title}: `{configuration_label(row)}`"]


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return ["No data available."]

    header = "| " + " | ".join(available) + " |"
    separator = "| " + " | ".join(["---"] * len(available)) + " |"
    rows = [header, separator]
    for _, record in frame[available].iterrows():
        values = [format_cell(record[column]) for column in available]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}"


def format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def configuration_label(row: pd.Series | dict[str, Any]) -> str:
    if isinstance(row, dict):
        source = row
    else:
        source = row.to_dict()
    parts = []
    for key in ["dataset", "roi_strategy", "svd_band", "decoder_type", "attack_type"]:
        value = source.get(key)
        if value is not None and not pd.isna(value):
            parts.append(f"{key}={value}")
    for key in sorted(source):
        if not str(key).startswith("config_"):
            continue
        value = source.get(key)
        if value is None or pd.isna(value):
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def generate_plots(
    plots_dir: Path,
    extraction_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    extraction_keys: list[str],
    top_n: int,
) -> None:
    if extraction_summary.empty:
        return

    plot_top_ranking(plots_dir / "top_configurations_by_score.png", extraction_summary, extraction_keys, top_n)
    plot_quality_vs_robustness(plots_dir / "quality_vs_robustness.png", extraction_summary, extraction_keys)
    plot_speed_vs_robustness(plots_dir / "speed_vs_robustness.png", extraction_summary, extraction_keys)

    if not category_summary.empty:
        plot_category_bars(plots_dir / "category_comparison.png", category_summary)

    if not attack_summary.empty:
        for decoder_type, decoder_frame in attack_summary.groupby("decoder_type", dropna=False):
            plot_attack_heatmap(
                plots_dir / f"heatmap_exact_match_{sanitize_filename(str(decoder_type))}.png",
                decoder_frame,
                value_column="exact_match_rate_all",
                title=f"Exact Match Rate by ROI and SVD Band ({decoder_type})",
            )
            plot_attack_heatmap(
                plots_dir / f"heatmap_ber_{sanitize_filename(str(decoder_type))}.png",
                decoder_frame,
                value_column="BER_mean",
                title=f"BER by ROI and SVD Band ({decoder_type})",
                invert=False,
            )


def plot_top_ranking(path: Path, frame: pd.DataFrame, extraction_keys: list[str], top_n: int) -> None:
    top = frame.head(top_n).copy()
    if top.empty:
        return
    labels = [short_label(record, extraction_keys) for _, record in top.iterrows()]
    scores = top["score"].tolist()

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.5)))
    ax.barh(labels[::-1], scores[::-1], color="#4C78A8")
    ax.set_xlabel("Composite score")
    ax.set_title("Top extraction configurations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_vs_robustness(path: Path, frame: pd.DataFrame, extraction_keys: list[str]) -> None:
    valid = frame.dropna(subset=["exact_match_rate_all", "PSNR_roi_mean"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        valid["PSNR_roi_mean"],
        valid["exact_match_rate_all"],
        s=40 + valid["samples_total"].fillna(0) * 3,
        c=valid["score"],
        cmap="viridis",
        alpha=0.8,
    )
    ax.set_xlabel("Mean PSNR ROI")
    ax.set_ylabel("Exact match rate")
    ax.set_title("Quality vs robustness")

    for _, row in valid.nlargest(min(8, len(valid)), "score").iterrows():
        ax.annotate(short_label(row, extraction_keys), (row["PSNR_roi_mean"], row["exact_match_rate_all"]), fontsize=8)

    fig.colorbar(scatter, ax=ax, label="Composite score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_speed_vs_robustness(path: Path, frame: pd.DataFrame, extraction_keys: list[str]) -> None:
    valid = frame.dropna(subset=["total_time_ms_mean", "exact_match_rate_all"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        valid["total_time_ms_mean"],
        valid["exact_match_rate_all"],
        s=40 + valid["samples_total"].fillna(0) * 3,
        c=valid["BER_mean"].fillna(valid["BER_mean"].max()),
        cmap="plasma_r",
        alpha=0.8,
    )
    ax.set_xlabel("Mean total time (ms)")
    ax.set_ylabel("Exact match rate")
    ax.set_title("Speed vs robustness")

    for _, row in valid.nsmallest(min(8, len(valid)), "total_time_ms_mean").iterrows():
        ax.annotate(short_label(row, extraction_keys), (row["total_time_ms_mean"], row["exact_match_rate_all"]), fontsize=8)

    fig.colorbar(scatter, ax=ax, label="Mean BER")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_category_bars(path: Path, frame: pd.DataFrame) -> None:
    categories = ["roi_strategy", "svd_band", "decoder_type", "attack_type"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = list(axes.flatten())

    for ax, category in zip(axes_list, categories):
        subset = frame[frame["category"] == category].sort_values("exact_match_rate_all", ascending=False)
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.bar(subset["value"].astype(str), subset["exact_match_rate_all"], color="#72B7B2")
        ax.set_title(category)
        ax.set_ylabel("Exact match rate")
        ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_attack_heatmap(path: Path, frame: pd.DataFrame, *, value_column: str, title: str, invert: bool = False) -> None:
    if value_column not in frame.columns:
        return
    pivot = (
        frame.groupby(["attack_type", "roi_strategy", "svd_band"], dropna=False)[value_column]
        .mean()
        .reset_index()
        .pivot_table(index=["roi_strategy", "svd_band"], columns="attack_type", values=value_column)
        .sort_index()
    )
    if pivot.empty:
        return

    fig_width = max(8, len(pivot.columns) * 2)
    fig_height = max(6, len(pivot.index) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = "magma_r" if invert else "viridis"
    data = pivot.to_numpy(dtype=float)
    image = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(item) for item in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{roi} | {band}" for roi, band in pivot.index])
    ax.set_title(title)

    for row_index in range(data.shape[0]):
        for col_index in range(data.shape[1]):
            value = data[row_index, col_index]
            text = "n/a" if math.isnan(value) else f"{value:.3f}"
            ax.text(col_index, row_index, text, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, label=value_column)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def short_label(row: pd.Series, keys: list[str]) -> str:
    values = []
    for key in keys:
        if key not in row.index:
            continue
        value = row[key]
        if value is None or pd.isna(value):
            continue
        short_key = key.replace("config_", "")
        values.append(f"{short_key}={value}")
    return " | ".join(values)


def sanitize_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


if __name__ == "__main__":
    main()
