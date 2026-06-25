from __future__ import annotations

from semantic_stego.config.cli_args import build_parser
from semantic_stego.config.defaults import DEFAULT_REPETITION_FACTOR


def test_parser_uses_default_repetition_factor() -> None:
    args = build_parser().parse_args([])

    assert args.repetition_factor == DEFAULT_REPETITION_FACTOR


def test_parser_accepts_explicit_repetition_factor() -> None:
    args = build_parser().parse_args(["--repetition-factor", "7"])

    assert args.repetition_factor == 7


def test_parser_accepts_proportional_embedding_strength_mode() -> None:
    args = build_parser().parse_args(["--embedding-strength-mode", "proportional_singular"])

    assert args.embedding_strength_mode == "proportional_singular"
