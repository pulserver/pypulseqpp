"""Every code block in a how-to guide runs and produces the output it shows."""

import doctest
from pathlib import Path

import pytest

GUIDES = sorted((Path(__file__).parents[1] / "docs/guides").glob("*.md"))


def test_the_guides_directory_is_not_empty():
    assert GUIDES


@pytest.mark.parametrize("guide", GUIDES, ids=lambda path: path.stem)
def test_a_guide_runs_as_it_is_written(guide, monkeypatch, tmp_path):
    """A guide that cannot be followed literally is a guide that is wrong."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)
    failures, _ = doctest.testfile(
        str(guide),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert not failures
