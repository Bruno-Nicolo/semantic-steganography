from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from semantic_stego.config.schemas import AttackConfig, Detection
from semantic_stego.experiments.efficient_sweep_runner import EfficientSweepRunner, StrengthSetting
from semantic_stego.stego.embedder import SvdEmbedder
from semantic_stego.stego.extractor import SvdExtractor


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(0, 0, 64, 64, 0.9, 1, "object")], 12.0


class _FakeWriter:
    def __init__(self) -> None:
        self.rows = []

    def write_result(self, row):
        self.rows.append(row)


def test_efficient_runner_detects_once_per_image(monkeypatch) -> None:
    image = np.random.default_rng(42).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    detector = _FakeDetector()
    writer = _FakeWriter()
    runner = EfficientSweepRunner.__new__(EfficientSweepRunner)
    runner.config = SimpleNamespace(
        split="val2017",
        roi_strategies=["largest", "full_image"],
        svd_bands=["high_energy"],
        decoders=["non_blind"],
        min_roi_area=None,
        save_roi_debug=False,
        output_dir=Path("outputs/test"),
        payload_policy="truncate_message",
        repetition_factor=1,
        payload_seed=42,
        payload_bits=1,
        embedding_strength=10.0,
        embedding_strength_mode="absolute",
    )
    runner.payload_bits_values = [1]
    runner.strength_settings = [StrengthSetting(10.0, "absolute")]
    runner.experiment_block = "test_grid"
    runner.rng = np.random.default_rng(1)
    runner.payload_rng = np.random.default_rng(2)
    runner.detector = detector
    runner.embedder = SvdEmbedder(payload_policy="truncate_message", repetition_factor=1)
    runner.extractor = SvdExtractor()
    runner.writer = writer
    runner.run_id = "test"
    runner.accepted_images = 0

    monkeypatch.setattr("semantic_stego.experiments.efficient_sweep_runner.read_image_rgb", lambda _: image)

    runner._process_image("image_1", Path("image_1.jpg"), [AttackConfig("none", None, {})])

    assert detector.calls == 1
    assert runner.accepted_images == 1
    assert len(writer.rows) == 2
    assert {row["roi_strategy"] for row in writer.rows} == {"largest", "full_image"}
    assert all(row["status"] == "success" for row in writer.rows)
