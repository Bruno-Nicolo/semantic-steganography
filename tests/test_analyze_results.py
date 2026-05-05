from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("analyze_results", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze_results = _load_module()


def test_build_attack_delta_summary_uses_none_as_baseline() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_key": "run::img1",
                "dataset": "coco/val2017",
                "roi_strategy": "largest",
                "svd_band": "high_energy",
                "decoder_type": "non_blind",
                "attack_type": "none",
                "BER": 0.10,
                "exact_match": True,
                "payload_success_ratio": 0.90,
                "status": "success",
                "payload_truncated": False,
                "image_accepted": True,
            },
            {
                "image_key": "run::img1",
                "dataset": "coco/val2017",
                "roi_strategy": "largest",
                "svd_band": "high_energy",
                "decoder_type": "non_blind",
                "attack_type": "gaussian_noise",
                "BER": 0.30,
                "exact_match": False,
                "payload_success_ratio": 0.70,
                "status": "success",
                "payload_truncated": False,
                "image_accepted": True,
            },
        ]
    )
    frame = analyze_results.standardize_frame(frame)

    summary = analyze_results.build_attack_delta_summary(
        frame,
        ["dataset", "roi_strategy", "svd_band", "decoder_type"],
    )

    attacked = summary[summary["attack_type"] == "gaussian_noise"].iloc[0]
    assert attacked["delta_BER_mean"] == pytest.approx(0.20)
    assert attacked["delta_exact_match_mean"] == -1.0
    assert attacked["BER_none_mean"] == pytest.approx(0.10)
    assert attacked["BER_attacked_mean"] == pytest.approx(0.30)


def test_build_coverage_summary_counts_rejections_by_reason() -> None:
    frame = pd.DataFrame(
        [
            {"image_key": "run::img1", "image_accepted": True, "status": "success", "image_filter_reason": None},
            {"image_key": "run::img2", "image_accepted": False, "status": "failed_payload_incompatible", "image_filter_reason": "ROI too small"},
            {"image_key": "run::img3", "image_accepted": False, "status": "failed_payload_incompatible", "image_filter_reason": "ROI too small"},
            {"image_key": "run::img4", "image_accepted": False, "status": "failed_payload_incompatible", "image_filter_reason": "No detection"},
        ]
    )
    frame = analyze_results.standardize_frame(frame)

    coverage = analyze_results.build_coverage_summary(frame)

    assert coverage.image_rows_total == 4
    assert coverage.accepted_image_rows == 1
    assert coverage.rejected_image_rows == 3
    assert coverage.acceptance_rate == 0.25
    assert coverage.rejection_by_reason.iloc[0]["image_filter_reason"] == "ROI too small"
    assert coverage.rejection_by_reason.iloc[0]["image_count"] == 2


def test_select_robustness_metric_falls_back_when_exact_match_is_flat() -> None:
    frame = pd.DataFrame(
        [
            {"exact_match_rate_complete": 0.0, "payload_success_ratio_mean": 0.72, "BER_mean": 0.31, "complete_payload_rate": 1.0},
            {"exact_match_rate_complete": 0.0, "payload_success_ratio_mean": 0.65, "BER_mean": 0.38, "complete_payload_rate": 1.0},
        ]
    )

    metric_column, metric_label, sort_columns, ascending = analyze_results.select_robustness_metric(frame)

    assert metric_column == "payload_success_ratio_mean"
    assert metric_label == "Payload success ratio"
    assert sort_columns[0] == "payload_success_ratio_mean"
    assert ascending == [False, True, False]


def test_select_robustness_metric_uses_overall_exact_match_when_complete_payload_is_unavailable() -> None:
    frame = pd.DataFrame(
        [
            {"exact_match_rate_complete": None, "exact_match_rate_all": 1.0, "BER_mean": 0.0, "success_rate": 1.0},
        ]
    )

    metric_column, metric_label, sort_columns, ascending = analyze_results.select_robustness_metric(frame)

    assert metric_column == "exact_match_rate_all"
    assert metric_label == "Exact match rate"
    assert sort_columns == ["exact_match_rate_all", "BER_mean", "success_rate"]
    assert ascending == [False, True, False]


def test_select_match_metric_prefers_overall_exact_match_when_complete_payload_is_missing() -> None:
    frame = pd.DataFrame(
        [
            {"exact_match_rate_complete": None, "exact_match_rate_all": 1.0},
        ]
    )

    metric_column, metric_label = analyze_results.select_match_metric(frame)

    assert metric_column == "exact_match_rate_all"
    assert metric_label == "Exact match rate"


def test_filter_attack_candidates_with_clean_baseline_excludes_bad_clean_rows() -> None:
    frame = pd.DataFrame(
        [
            {"attack_type": "gaussian_blur", "BER_none_mean": 0.20, "delta_BER_mean": 0.05},
            {"attack_type": "gaussian_noise", "BER_none_mean": 0.31, "delta_BER_mean": 0.01},
            {"attack_type": "none", "BER_none_mean": 0.10, "delta_BER_mean": 0.0},
        ]
    )

    filtered = analyze_results.filter_attack_candidates_with_clean_baseline(frame, threshold=0.25)

    assert len(filtered) == 1
    assert filtered.iloc[0]["attack_type"] == "gaussian_blur"


def test_safe_statistics_ignore_infinite_values() -> None:
    series = pd.Series([1.0, 2.0, np.inf, -np.inf, None])

    assert analyze_results.safe_mean(series) == pytest.approx(1.5)
    assert analyze_results.safe_median(series) == pytest.approx(1.5)
    assert analyze_results.safe_std(series) == pytest.approx(pd.Series([1.0, 2.0]).std())
    assert analyze_results.safe_sem(series) == pytest.approx(pd.Series([1.0, 2.0]).std(ddof=1) / (2 ** 0.5))
