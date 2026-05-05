from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from semantic_stego.config.schemas import Detection
from semantic_stego.experiments.runner import ExperimentRunner


def _runner(payload_policy: str = "truncate_message") -> ExperimentRunner:
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config = SimpleNamespace(
        roi_strategies=["largest", "smallest", "full_image"],
        svd_bands=["high_energy", "mid_energy"],
        min_roi_area=None,
        payload_text="CIAO",
        payload_policy=payload_policy,
    )
    runner.rng = np.random.default_rng(42)
    return runner


def _detections() -> list[Detection]:
    return [
        Detection(0, 0, 20, 20, 0.9, 0, "large"),
        Detection(0, 0, 3, 3, 0.8, 1, "small"),
    ]


def test_select_rois_keeps_partially_compatible_strategies() -> None:
    runner = _runner("truncate_message")
    runner._estimate_band_capacities = lambda image, roi: {
        "high_energy": 5 if roi.strategy in {"largest", "full_image"} else 0,
        "mid_energy": 0,
    }

    selected, error, extras = runner._select_rois_for_image(np.zeros((30, 30, 3), dtype=np.uint8), _detections(), np.zeros(32, dtype=np.uint8))

    assert error is None
    assert extras is None
    assert selected is not None
    assert set(selected) == {"largest", "full_image"}


def test_select_rois_rejects_image_when_no_strategy_has_usable_band() -> None:
    runner = _runner("truncate_message")
    runner._estimate_band_capacities = lambda image, roi: {"high_energy": 0, "mid_energy": 0}

    selected, error, extras = runner._select_rois_for_image(np.zeros((30, 30, 3), dtype=np.uint8), _detections(), np.zeros(32, dtype=np.uint8))

    assert selected is None
    assert error is not None
    assert "effective capacity 0" in error
    assert extras is not None


def test_strict_payload_policy_requires_full_capacity() -> None:
    runner = _runner("skip_image")
    runner._estimate_band_capacities = lambda image, roi: {
        "high_energy": 31 if roi.strategy == "largest" else 0,
        "mid_energy": 0,
    }

    selected, error, extras = runner._select_rois_for_image(np.zeros((30, 30, 3), dtype=np.uint8), _detections(), np.zeros(32, dtype=np.uint8))

    assert selected is None
    assert error is not None
    assert "required payload length 32" in error
    assert extras is not None
