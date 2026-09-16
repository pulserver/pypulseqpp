"""The examples page lists every shipped sequence with its own summary line."""

import importlib
import re
from pathlib import Path

import pytest

import pypulseqpp.sequences as sequences

PAGE = Path(__file__).parents[1] / "docs/examples/index.md"

_ROW = re.compile(r"^\| `(\w+)` \| (.+?) \|$", re.M)


def _rows() -> dict[str, str]:
    return dict(_ROW.findall(PAGE.read_text()))


def _summary(name: str) -> str:
    """The first paragraph of a sequence module's entry point, on one line."""
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")
    return (module.main.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")


def test_every_shipped_sequence_has_a_row_and_no_row_is_stale():
    assert set(_rows()) == set(sequences.ZOO)


@pytest.mark.parametrize("name", sequences.ZOO)
def test_a_row_repeats_the_sequence_own_summary_line(name):
    assert _rows()[name] == _summary(name)
