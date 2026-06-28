from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
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
    "numpy_svd_time_ms",
    "attack_time_ms",
    "total_time_ms",
    "svd_reconstruction_error",
]

CLEAN_BER_ELIGIBILITY_THRESHOLD = 0.25


@dataclass(slots=True)
class AnalysisContext:
    frame: pd.DataFrame
    run_dirs: list[Path]
    varying_config_columns: list[str]
    embedding_keys: list[str]
    extraction_keys: list[str]


@dataclass(slots=True)
class CoverageSummary:
    image_rows_total: int
    accepted_image_rows: int
    rejected_image_rows: int
    acceptance_rate: float
    rejection_by_reason: pd.DataFrame


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
    attack_delta_summary = build_attack_delta_summary(extraction_frame, context.extraction_keys)

    category_summary = build_category_summary(extraction_frame)
    roi_class_summary = build_roi_class_summary(extraction_frame, args.top_n)
    coverage_summary = build_coverage_summary(context.frame)

    embedding_summary.to_csv(analysis_dir / "embedding_summary.csv", index=False)
    extraction_summary.to_csv(analysis_dir / "extraction_summary.csv", index=False)
    attack_summary.to_csv(analysis_dir / "attack_summary.csv", index=False)
    attack_delta_summary.to_csv(analysis_dir / "attack_delta_summary.csv", index=False)
    category_summary.to_csv(analysis_dir / "category_summary.csv", index=False)
    roi_class_summary.to_csv(analysis_dir / "roi_class_summary.csv", index=False)
    coverage_summary.rejection_by_reason.to_csv(analysis_dir / "rejection_summary.csv", index=False)

    write_key_metrics_json(analysis_dir, context, embedding_summary, extraction_summary, attack_summary, attack_delta_summary, coverage_summary)
    write_conclusions_markdown(analysis_dir, context, embedding_summary, extraction_summary, attack_summary, attack_delta_summary, coverage_summary, args.top_n)

    generate_plots(plots_dir, context.frame, extraction_summary, attack_summary, attack_delta_summary, category_summary, roi_class_summary, coverage_summary, context.extraction_keys, args.top_n)

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
    working["image_filter_reason"] = working.get("image_filter_reason", None)
    working["payload_bits_requested"] = pd.to_numeric(working.get("payload_bits_requested"), errors="coerce")

    if "exact_match" not in working.columns:
        working["exact_match"] = False
    if "payload_truncated" not in working.columns:
        working["payload_truncated"] = False
    working["exact_match_bool"] = working["exact_match"].map(parse_bool).fillna(False).astype(bool)
    working["payload_truncated_bool"] = working["payload_truncated"].map(parse_bool).fillna(False).astype(bool)
    working["is_complete_payload"] = (~working["payload_truncated_bool"]).astype(bool)
    working["image_accepted_bool"] = working.get("image_accepted", False)
    if not isinstance(working["image_accepted_bool"], pd.Series):
        working["image_accepted_bool"] = pd.Series([working["image_accepted_bool"]] * len(working), index=working.index)
    working["image_accepted_bool"] = working["image_accepted_bool"].map(parse_bool).fillna(False).astype(bool)
    working["is_success"] = working["status"].eq("success")
    working["is_failure"] = ~working["is_success"]
    if "image_key" not in working.columns:
        image_ids = working["image_id"].astype(str) if "image_id" in working.columns else pd.Series([""] * len(working), index=working.index, dtype=str)
        source_run_dir = working.get("source_run_dir")
        if isinstance(source_run_dir, pd.Series):
            working["image_key"] = source_run_dir.fillna("") + "::" + image_ids
        else:
            working["image_key"] = image_ids

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
        complete_success = success[success["is_complete_payload"]]

        row.update(
            {
                "samples_total": int(len(group)),
                "success_count": int(group["is_success"].sum()),
                "failure_count": int(group["is_failure"].sum()),
                "success_rate": safe_mean(group["is_success"]),
                "failure_rate": safe_mean(group["is_failure"]),
                "exact_match_rate_all": safe_mean(group["exact_match_bool"]),
                "exact_match_rate_success": safe_mean(success["exact_match_bool"]),
                "samples_complete": int(len(complete_success)),
                "complete_payload_rate": safe_mean(success["is_complete_payload"]),
                "exact_match_rate_complete": safe_mean(complete_success["exact_match_bool"]),
                "exact_match_sem": safe_sem(group["exact_match_bool"]),
                "exact_match_complete_sem": safe_sem(complete_success["exact_match_bool"]),
            }
        )

        for metric in SUCCESS_METRICS:
            if metric not in group.columns:
                continue
            row[f"{metric}_mean"] = safe_mean(success[metric])
            row[f"{metric}_median"] = safe_median(success[metric])
            row[f"{metric}_std"] = safe_std(success[metric])

        if "payload_bits_requested" in group.columns:
            row["payload_bits_requested_mean"] = safe_mean(success["payload_bits_requested"])
            row["payload_bits_requested_median"] = safe_median(success["payload_bits_requested"])
            row["payload_bits_requested_std"] = safe_std(success["payload_bits_requested"])

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["score"] = compute_composite_score(summary)
    return summary.sort_values(["score", "exact_match_rate_all", "success_rate"], ascending=[False, False, False]).reset_index(drop=True)


def safe_mean(series: pd.Series) -> float:
    valid = finite_numeric(series)
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def safe_median(series: pd.Series) -> float:
    valid = finite_numeric(series)
    if valid.empty:
        return float("nan")
    return float(valid.median())


def safe_std(series: pd.Series) -> float:
    valid = finite_numeric(series)
    if valid.empty:
        return float("nan")
    return float(valid.std())


def safe_sem(series: pd.Series) -> float:
    valid = finite_numeric(series)
    if len(valid) <= 1:
        return 0.0 if len(valid) == 1 else float("nan")
    return float(valid.std(ddof=1) / math.sqrt(len(valid)))


def finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def compute_composite_score(summary: pd.DataFrame) -> pd.Series:
    metrics = {
        "success_rate": (0.20, True),
        "exact_match_rate_complete": (0.25, True),
        "complete_payload_rate": (0.05, True),
        "payload_success_ratio_mean": (0.15, True),
        "BER_mean": (0.15, False),
        "PSNR_roi_mean": (0.10, True),
        "SSIM_roi_mean": (0.05, True),
        "extraction_time_ms_mean": (0.05, False),
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
                    "exact_match_sem": safe_sem(group["exact_match_bool"]),
                    "complete_payload_rate": safe_mean(success["is_complete_payload"]),
                    "exact_match_rate_complete": safe_mean(success[success["is_complete_payload"]]["exact_match_bool"]),
                    "payload_success_ratio_mean": safe_mean(success["payload_success_ratio"]) if "payload_success_ratio" in success.columns else float("nan"),
                    "BER_mean": safe_mean(success["BER"]) if "BER" in success.columns else float("nan"),
                    "BER_std": safe_std(success["BER"]) if "BER" in success.columns else float("nan"),
                    "PSNR_roi_mean": safe_mean(success["PSNR_roi"]) if "PSNR_roi" in success.columns else float("nan"),
                    "SSIM_roi_mean": safe_mean(success["SSIM_roi"]) if "SSIM_roi" in success.columns else float("nan"),
                    "total_time_ms_mean": safe_mean(success["total_time_ms"]) if "total_time_ms" in success.columns else float("nan"),
                    "extraction_time_ms_mean": safe_mean(success["extraction_time_ms"]) if "extraction_time_ms" in success.columns else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_roi_class_summary(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty or "roi_class_name" not in frame.columns:
        return pd.DataFrame()
    working = frame[frame["roi_class_name"].notna() & frame["roi_class_name"].astype(str).ne("")].copy()
    if working.empty:
        return pd.DataFrame()

    class_order = working["roi_class_name"].astype(str).value_counts().head(top_n).index.tolist()
    working = working[working["roi_class_name"].astype(str).isin(class_order)]
    rows: list[dict[str, Any]] = []
    for class_name, group in working.groupby("roi_class_name", dropna=False):
        success = group[group["is_success"]]
        rows.append(
            {
                "roi_class_name": class_name,
                "samples_total": int(len(group)),
                "success_count": int(group["is_success"].sum()),
                "success_rate": safe_mean(group["is_success"]),
                "exact_match_rate_all": safe_mean(group["exact_match_bool"]),
                "exact_match_sem": safe_sem(group["exact_match_bool"]),
                "BER_mean": safe_mean(success["BER"]) if "BER" in success.columns else float("nan"),
                "BER_std": safe_std(success["BER"]) if "BER" in success.columns else float("nan"),
                "payload_success_ratio_mean": safe_mean(success["payload_success_ratio"]) if "payload_success_ratio" in success.columns else float("nan"),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["frequency_rank"] = summary["roi_class_name"].astype(str).map({name: index + 1 for index, name in enumerate(class_order)})
    return summary.sort_values("frequency_rank").reset_index(drop=True)


def build_attack_delta_summary(frame: pd.DataFrame, extraction_keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    join_keys = ["image_key", *extraction_keys]
    available_join_keys = [key for key in join_keys if key in frame.columns]
    if not available_join_keys:
        return pd.DataFrame()

    base_columns = available_join_keys + ["BER", "exact_match_bool", "payload_success_ratio"]
    baseline = frame[(frame["is_success"]) & (frame["attack_type"] == "none")][base_columns].copy()
    if baseline.empty:
        return pd.DataFrame()
    baseline = baseline.rename(
        columns={
            "BER": "BER_none",
            "exact_match_bool": "exact_match_none",
            "payload_success_ratio": "payload_success_ratio_none",
        }
    )

    attacked = frame[frame["is_success"]].copy()
    merged = attacked.merge(baseline, on=available_join_keys, how="left")
    merged = merged[merged["BER_none"].notna()].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["delta_BER"] = merged["BER"] - merged["BER_none"]
    merged["delta_exact_match"] = merged["exact_match_bool"].astype(float) - merged["exact_match_none"].astype(float)
    merged["delta_payload_success_ratio"] = merged["payload_success_ratio"] - merged["payload_success_ratio_none"]

    rows: list[dict[str, Any]] = []
    group_keys = [*extraction_keys, "attack_type"]
    for key_values, group in merged.groupby(group_keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = dict(zip(group_keys, key_values))
        row.update(
            {
                "samples_total": int(len(group)),
                "delta_BER_mean": safe_mean(group["delta_BER"]),
                "delta_BER_std": safe_std(group["delta_BER"]),
                "delta_exact_match_mean": safe_mean(group["delta_exact_match"]),
                "delta_exact_match_std": safe_std(group["delta_exact_match"]),
                "delta_payload_success_ratio_mean": safe_mean(group["delta_payload_success_ratio"]),
                "delta_payload_success_ratio_std": safe_std(group["delta_payload_success_ratio"]),
                "BER_none_mean": safe_mean(group["BER_none"]),
                "BER_attacked_mean": safe_mean(group["BER"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_coverage_summary(frame: pd.DataFrame) -> CoverageSummary:
    if frame.empty or "image_key" not in frame.columns:
        return CoverageSummary(0, 0, 0, float("nan"), pd.DataFrame(columns=["image_filter_reason", "image_count"]))

    image_rows = frame.groupby("image_key", dropna=False).agg(
        image_accepted_bool=("image_accepted_bool", "max"),
        image_filter_reason=("image_filter_reason", "first"),
    )
    accepted = int(image_rows["image_accepted_bool"].sum())
    total = int(len(image_rows))
    rejected = total - accepted
    rejection_by_reason = (
        image_rows[~image_rows["image_accepted_bool"]]
        .groupby("image_filter_reason", dropna=False)
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
    )
    return CoverageSummary(
        image_rows_total=total,
        accepted_image_rows=accepted,
        rejected_image_rows=rejected,
        acceptance_rate=(accepted / total) if total else float("nan"),
        rejection_by_reason=rejection_by_reason,
    )


def filter_attack_candidates_with_clean_baseline(frame: pd.DataFrame, threshold: float = CLEAN_BER_ELIGIBILITY_THRESHOLD) -> pd.DataFrame:
    if frame.empty or "BER_none_mean" not in frame.columns:
        return frame.iloc[0:0].copy() if hasattr(frame, "iloc") else frame
    return frame[(frame["attack_type"] != "none") & (frame["BER_none_mean"] <= threshold)].copy()


def write_key_metrics_json(
    analysis_dir: Path,
    context: AnalysisContext,
    embedding_summary: pd.DataFrame,
    extraction_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    attack_delta_summary: pd.DataFrame,
    coverage_summary: CoverageSummary,
) -> None:
    payload = {
        "runs_analyzed": [str(path) for path in context.run_dirs],
        "varying_config_columns": context.varying_config_columns,
        "rows_total": int(len(context.frame)),
        "success_rows": int(context.frame["is_success"].sum()),
        "failure_rows": int(context.frame["is_failure"].sum()),
        "image_coverage": {
            "candidate_images": coverage_summary.image_rows_total,
            "accepted_images": coverage_summary.accepted_image_rows,
            "rejected_images": coverage_summary.rejected_image_rows,
            "acceptance_rate": coverage_summary.acceptance_rate,
            "rejection_by_reason": coverage_summary.rejection_by_reason.to_dict(orient="records"),
        },
        "clean_ber_eligibility_threshold": CLEAN_BER_ELIGIBILITY_THRESHOLD,
        "best_embedding_configuration": dataframe_head_record(embedding_summary),
        "best_extraction_configuration": dataframe_head_record(extraction_summary),
        "best_per_attack": top_record_per_group(attack_summary, "attack_type"),
        "most_stable_attack_configuration": dataframe_head_record(filter_attack_candidates_with_clean_baseline(attack_delta_summary).sort_values(["delta_BER_mean", "delta_exact_match_mean"], ascending=[True, False])) if not filter_attack_candidates_with_clean_baseline(attack_delta_summary).empty else None,
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
        best = group.sort_values(["score", "exact_match_rate_complete", "exact_match_rate_all"], ascending=[False, False, False]).iloc[0]
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
    attack_delta_summary: pd.DataFrame,
    coverage_summary: CoverageSummary,
    top_n: int,
) -> None:
    lines: list[str] = []
    total_rows = len(context.frame)
    success_rows = int(context.frame["is_success"].sum())
    failure_rows = int(context.frame["is_failure"].sum())
    accepted_rows = int(context.frame["image_accepted_bool"].sum())
    rejected_rows = int((~context.frame["image_accepted_bool"]).sum())

    lines.append("# Analysis Conclusions")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Runs analyzed: {len(context.run_dirs)}")
    lines.append(f"- Total result rows: {total_rows}")
    lines.append(f"- Success rows: {success_rows}")
    lines.append(f"- Failure rows: {failure_rows}")
    lines.append(f"- Global success rate: {success_rows / max(total_rows, 1):.2%}")
    lines.append(f"- Accepted result rows: {accepted_rows}")
    lines.append(f"- Rejected result rows: {rejected_rows}")
    lines.append(f"- Candidate images: {coverage_summary.image_rows_total}")
    lines.append(f"- Accepted images: {coverage_summary.accepted_image_rows}")
    lines.append(f"- Rejected images: {coverage_summary.rejected_image_rows}")
    lines.append(f"- Image acceptance rate: {format_percent(coverage_summary.acceptance_rate)}")
    if context.varying_config_columns:
        lines.append(f"- Varying run-level parameters: {', '.join(context.varying_config_columns)}")
    lines.append("")

    if not coverage_summary.rejection_by_reason.empty:
        lines.append("## Rejection Breakdown")
        lines.append("")
        for _, row in coverage_summary.rejection_by_reason.iterrows():
            lines.append(f"- `{row['image_filter_reason']}`: {int(row['image_count'])} image(s)")
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
        lines.extend(format_highlight_line("Best complete-payload exact match", best_by_metric(extraction_summary, "exact_match_rate_complete", ascending=False)))
        lines.extend(format_highlight_line("Lowest BER", best_by_metric(extraction_summary, "BER_mean", ascending=True)))
        lines.extend(format_highlight_line("Best image quality", best_by_metric(extraction_summary, "PSNR_roi_mean", ascending=False)))
        lines.extend(format_highlight_line("Fastest extraction", best_by_metric(extraction_summary, "extraction_time_ms_mean", ascending=True)))
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
                    "exact_match_rate_complete",
                    "complete_payload_rate",
                    "BER_mean",
                    "PSNR_roi_mean",
                    "SSIM_roi_mean",
                    "extraction_time_ms_mean",
                    "score",
                ],
            )
        )
        lines.append("")

    if not attack_summary.empty:
        lines.append("## Best Configuration Per Attack")
        lines.append("")
        for attack_type, group in attack_summary.groupby("attack_type", dropna=False):
            best = group.sort_values(["score", "exact_match_rate_complete", "exact_match_rate_all"], ascending=[False, False, False]).iloc[0]
            lines.append(f"### {attack_type}")
            lines.extend(format_best_configuration(best))
            lines.append("")

    if not attack_delta_summary.empty:
        lines.append("## Attack Degradation Highlights")
        lines.append("")
        eligible = filter_attack_candidates_with_clean_baseline(attack_delta_summary)
        lowest_delta = best_by_metric(eligible, "delta_BER_mean", ascending=True)
        strongest_delta = best_by_metric(attack_delta_summary[attack_delta_summary["attack_type"] != "none"], "delta_BER_mean", ascending=False)
        if lowest_delta is None:
            lines.append(
                f"- Most stable under attack: no configuration met the clean BER eligibility threshold ({CLEAN_BER_ELIGIBILITY_THRESHOLD:.2f})"
            )
        else:
            lines.extend(format_highlight_line("Most stable under attack", lowest_delta))
        lines.extend(format_highlight_line("Largest BER degradation", strongest_delta))
        lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- `consolidated_results.csv`: all runs merged into one table")
    lines.append("- `embedding_summary.csv`: summary by ROI strategy and SVD band")
    lines.append("- `extraction_summary.csv`: summary by extraction configuration")
    lines.append("- `attack_summary.csv`: summary by extraction configuration and attack")
    lines.append("- `attack_delta_summary.csv`: degradation relative to the clean `none` baseline")
    lines.append("- `category_summary.csv`: macro comparison by ROI strategy, SVD band, decoder, attack")
    lines.append("- `roi_class_summary.csv`: BER and exact-match comparison for the most frequent ROI classes")
    lines.append("- `rejection_summary.csv`: accepted/rejected image coverage by reason")
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
    lines = [
        f"- Configuration: `{label}`",
        f"- Score: {format_number(row.get('score'))}",
        f"- Success rate: {format_percent(row.get('success_rate'))}",
        f"- Exact match rate: {format_percent(row.get('exact_match_rate_all'))}",
        f"- Exact match rate (complete payload): {format_percent(row.get('exact_match_rate_complete'))}",
        f"- Complete payload rate: {format_percent(row.get('complete_payload_rate'))}",
        f"- Mean BER: {format_number(row.get('BER_mean'))}",
        f"- Mean PSNR ROI: {format_number(row.get('PSNR_roi_mean'))}",
        f"- Mean SSIM ROI: {format_number(row.get('SSIM_roi_mean'))}",
        f"- Mean extraction time: {format_number(row.get('extraction_time_ms_mean'))} ms",
    ]
    if row.get("payload_bits_requested_mean") is not None and not pd.isna(row.get("payload_bits_requested_mean")):
        lines.append(
            "- Mean payload embedded/requested: "
            f"{format_number(row.get('payload_bits_embedded_mean'))} / {format_number(row.get('payload_bits_requested_mean'))} bits"
        )
    return lines


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
    frame: pd.DataFrame,
    extraction_summary: pd.DataFrame,
    attack_summary: pd.DataFrame,
    attack_delta_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    roi_class_summary: pd.DataFrame,
    coverage_summary: CoverageSummary,
    extraction_keys: list[str],
    top_n: int,
) -> None:
    for stale_plot in plots_dir.glob("*.png"):
        stale_plot.unlink()

    if not frame.empty:
        plot_status_breakdown(plots_dir / "status_breakdown.png", frame)

    if not extraction_summary.empty:
        plot_top_ranking(plots_dir / "top_configurations_by_score.png", extraction_summary, extraction_keys, top_n)
        plot_quality_vs_robustness(plots_dir / "quality_vs_robustness.png", extraction_summary, extraction_keys)
        plot_speed_vs_robustness(plots_dir / "speed_vs_robustness.png", extraction_summary, extraction_keys)
        plot_complete_payload_ranking(plots_dir / "top_complete_payload_configurations.png", extraction_summary, extraction_keys, top_n)

    if not category_summary.empty:
        plot_category_bars(plots_dir / "category_comparison.png", category_summary)

    if not roi_class_summary.empty:
        plot_roi_class_bars(plots_dir / "roi_class_comparison.png", roi_class_summary)

    if coverage_summary.image_rows_total > 0:
        plot_coverage_breakdown(plots_dir / "image_coverage.png", coverage_summary)
    if not coverage_summary.rejection_by_reason.empty:
        plot_rejection_reasons(plots_dir / "rejection_reasons.png", coverage_summary)

    if not attack_summary.empty:
        for decoder_type, decoder_frame in attack_summary.groupby("decoder_type", dropna=False):
            match_metric = select_match_metric(decoder_frame)
            if match_metric is not None:
                value_column, metric_label = match_metric
                plot_attack_heatmap(
                    plots_dir / f"heatmap_exact_match_{sanitize_filename(str(decoder_type))}.png",
                    decoder_frame,
                    value_column=value_column,
                    title=f"{metric_label} by ROI and SVD Band ({decoder_type})",
                )
            plot_attack_heatmap(
                plots_dir / f"heatmap_payload_success_{sanitize_filename(str(decoder_type))}.png",
                decoder_frame,
                value_column="payload_success_ratio_mean",
                title=f"Payload Success Ratio by ROI and SVD Band ({decoder_type})",
            )
            plot_attack_heatmap(
                plots_dir / f"heatmap_ber_{sanitize_filename(str(decoder_type))}.png",
                decoder_frame,
                value_column="BER_mean",
                title=f"BER by ROI and SVD Band ({decoder_type})",
                invert=False,
            )

    if not attack_delta_summary.empty:
        for decoder_type, decoder_frame in attack_delta_summary.groupby("decoder_type", dropna=False):
            plot_attack_heatmap(
                plots_dir / f"heatmap_delta_ber_{sanitize_filename(str(decoder_type))}.png",
                decoder_frame,
                value_column="delta_BER_mean",
                title=f"Delta BER vs clean baseline ({decoder_type})",
                invert=False,
            )
        plot_attack_degradation_bars(plots_dir / "attack_degradation_overview.png", attack_delta_summary)


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


def plot_complete_payload_ranking(path: Path, frame: pd.DataFrame, extraction_keys: list[str], top_n: int) -> None:
    metric_column, metric_label, sort_columns, ascending = select_robustness_metric(frame)
    top = frame.sort_values(sort_columns, ascending=ascending).head(top_n).copy()
    if top.empty:
        return
    labels = [short_label(record, extraction_keys) for _, record in top.iterrows()]
    values = top[metric_column].tolist()
    errors = None
    if metric_column == "exact_match_rate_complete" and "exact_match_complete_sem" in top.columns:
        errors = top["exact_match_complete_sem"].fillna(0.0).tolist()

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.5)))
    ax.barh(labels[::-1], values[::-1], xerr=errors[::-1] if errors is not None else None, color="#59A14F", capsize=4)
    ax.set_xlabel(metric_label)
    ax.set_title("Top complete-payload configurations")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quality_vs_robustness(path: Path, frame: pd.DataFrame, extraction_keys: list[str]) -> None:
    metric_column, metric_label, _, _ = select_robustness_metric(frame)
    valid = frame.dropna(subset=[metric_column, "PSNR_roi_mean"])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        valid["PSNR_roi_mean"],
        valid[metric_column],
        s=40 + valid["samples_total"].fillna(0) * 3,
        c=valid["score"],
        cmap="viridis",
        alpha=0.8,
    )
    if metric_column == "exact_match_rate_complete" and "exact_match_complete_sem" in valid.columns:
        ax.errorbar(valid["PSNR_roi_mean"], valid[metric_column], yerr=valid["exact_match_complete_sem"].fillna(0.0), fmt="none", ecolor="#666666", alpha=0.35)
    ax.set_xlabel("Mean PSNR ROI")
    ax.set_ylabel(metric_label)
    ax.set_title("Quality vs robustness")

    if valid[metric_column].nunique(dropna=True) > 1:
        for _, row in valid.nlargest(min(6, len(valid)), "score").iterrows():
            ax.annotate(short_label(row, extraction_keys), (row["PSNR_roi_mean"], row[metric_column]), fontsize=8)

    fig.colorbar(scatter, ax=ax, label="Composite score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_speed_vs_robustness(path: Path, frame: pd.DataFrame, extraction_keys: list[str]) -> None:
    metric_column, metric_label, _, _ = select_robustness_metric(frame)
    valid = frame.dropna(subset=["extraction_time_ms_mean", metric_column])
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        valid["extraction_time_ms_mean"],
        valid[metric_column],
        s=40 + valid["samples_total"].fillna(0) * 3,
        c=valid["BER_mean"].fillna(valid["BER_mean"].max()),
        cmap="plasma_r",
        alpha=0.8,
    )
    ax.set_xlabel("Mean extraction time (ms)")
    ax.set_ylabel(metric_label)
    ax.set_title("Speed vs robustness")

    if valid[metric_column].nunique(dropna=True) > 1:
        for _, row in valid.nsmallest(min(6, len(valid)), "extraction_time_ms_mean").iterrows():
            ax.annotate(short_label(row, extraction_keys), (row["extraction_time_ms_mean"], row[metric_column]), fontsize=8)

    fig.colorbar(scatter, ax=ax, label="Mean BER")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_category_bars(path: Path, frame: pd.DataFrame) -> None:
    categories = ["roi_strategy", "svd_band", "decoder_type", "attack_type"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_list = list(axes.flatten())
    metric_column, metric_label, _, _ = select_robustness_metric(frame)

    for ax, category in zip(axes_list, categories):
        subset = frame[frame["category"] == category].sort_values(metric_column, ascending=False)
        if subset.empty:
            ax.set_visible(False)
            continue
        errors = None
        if metric_column == "exact_match_rate_complete" and "exact_match_complete_sem" in subset.columns:
            errors = subset["exact_match_complete_sem"]
        elif metric_column == "exact_match_rate_all" and "exact_match_sem" in subset.columns:
            errors = subset["exact_match_sem"]
        ax.bar(subset["value"].astype(str), subset[metric_column], yerr=errors, color="#72B7B2", capsize=4)
        ax.set_title(category)
        ax.set_ylabel(metric_label)
        ax.tick_params(axis="x", rotation=25)
        for index, (_, row) in enumerate(subset.iterrows()):
            offset = 0.01 if row[metric_column] >= 0 else -0.03
            ax.text(index, row[metric_column] + offset, f"n={int(row['samples_total'])}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roi_class_bars(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    labels = frame["roi_class_name"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.7), 9), sharex=True)
    axes[0].bar(x, frame["BER_mean"], yerr=frame.get("BER_std"), color="#E15759", capsize=4)
    axes[0].set_ylabel("Mean BER")
    axes[0].set_title("Mean BER by frequent ROI class")

    axes[1].bar(x, frame["exact_match_rate_all"], yerr=frame.get("exact_match_sem"), color="#4C78A8", capsize=4)
    axes[1].set_ylabel("Exact match rate")
    axes[1].set_title("Exact match by frequent ROI class")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    for index, row in frame.iterrows():
        axes[0].text(index, row["BER_mean"] if not pd.isna(row["BER_mean"]) else 0, f"n={int(row['samples_total'])}", ha="center", va="bottom", fontsize=8)

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


def plot_attack_degradation_bars(path: Path, frame: pd.DataFrame) -> None:
    subset = frame[frame["attack_type"] != "none"].copy()
    if subset.empty:
        return
    ordered = subset.groupby("attack_type", dropna=False)["delta_BER_mean"].mean().sort_values().reset_index()
    errors = subset.groupby("attack_type", dropna=False)["delta_BER_mean"].std().reindex(ordered["attack_type"]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ordered["attack_type"].astype(str), ordered["delta_BER_mean"], yerr=errors, color="#E15759", capsize=4)
    ax.set_ylabel("Mean delta BER vs clean")
    ax.set_title("Attack degradation overview")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_coverage_breakdown(path: Path, coverage_summary: CoverageSummary) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(["accepted", "rejected"], [coverage_summary.accepted_image_rows, coverage_summary.rejected_image_rows], color=["#59A14F", "#E15759"])
    axes[0].set_title("Image coverage")
    axes[0].set_ylabel("Image count")

    rejection = coverage_summary.rejection_by_reason.copy()
    if rejection.empty:
        axes[1].set_visible(False)
    else:
        axes[1].barh(rejection["image_filter_reason"].astype(str)[::-1], rejection["image_count"][::-1], color="#F28E2B")
        axes[1].set_title("Rejected images by reason")
        axes[1].set_xlabel("Image count")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rejection_reasons(path: Path, coverage_summary: CoverageSummary) -> None:
    rejection = coverage_summary.rejection_by_reason.copy()
    if rejection.empty:
        return
    top = rejection.head(15)

    fig_height = max(6, len(top) * 0.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(top["image_filter_reason"].astype(str)[::-1], top["image_count"][::-1], color="#F28E2B")
    ax.set_title("Top rejection reasons")
    ax.set_xlabel("Image count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_status_breakdown(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty or "status" not in frame.columns:
        return
    counts = frame["status"].fillna("unknown").value_counts()
    if counts.empty:
        return

    fig_height = max(5, len(counts) * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(counts.index.astype(str)[::-1], counts.values[::-1], color="#4C78A8")
    ax.set_title("Result rows by status")
    ax.set_xlabel("Row count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_robustness_metric(frame: pd.DataFrame) -> tuple[str, str, list[str], list[bool]]:
    match_metric = select_match_metric(frame)
    if match_metric is not None:
        metric_column, metric_label = match_metric
        if metric_column == "exact_match_rate_complete":
            return (
                metric_column,
                metric_label,
                ["exact_match_rate_complete", "BER_mean", "complete_payload_rate"],
                [False, True, False],
            )
        return (
            metric_column,
            metric_label,
            ["exact_match_rate_all", "BER_mean", "success_rate"],
            [False, True, False],
        )
    return (
        "payload_success_ratio_mean",
        "Payload success ratio",
        ["payload_success_ratio_mean", "BER_mean", "complete_payload_rate"],
        [False, True, False],
    )


def select_match_metric(frame: pd.DataFrame) -> tuple[str, str] | None:
    if metric_has_signal(frame, "exact_match_rate_complete"):
        return ("exact_match_rate_complete", "Exact match rate (complete payload)")
    if metric_has_signal(frame, "exact_match_rate_all"):
        return ("exact_match_rate_all", "Exact match rate")
    return None


def metric_has_signal(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return False
    if valid.nunique(dropna=True) > 1:
        return True
    return bool(valid.gt(0).any())


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
