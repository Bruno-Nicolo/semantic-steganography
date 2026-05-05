from __future__ import annotations

from semantic_stego.config.cli_args import build_parser
from semantic_stego.config.defaults import DEFAULT_REPETITION_FACTOR


def test_parser_uses_default_repetition_factor() -> None:
    args = build_parser().parse_args([])

    assert args.repetition_factor == DEFAULT_REPETITION_FACTOR


def test_parser_accepts_explicit_repetition_factor() -> None:
    args = build_parser().parse_args(["--repetition-factor", "7"])

    assert args.repetition_factor == 7
