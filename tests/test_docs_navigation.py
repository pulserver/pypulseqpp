"""The site's top-level sections appear in one fixed order, on the site and in the PDF."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"

#: The six top-level sections, in the order the sidebar lists them.
SECTIONS = [
    "user-guide/index",
    "developer-guide/index",
    "explanations/index",
    "examples/index",
    "api/index",
    "misc/index",
]


def _toctree(page: Path) -> list[str]:
    """The entries of the first toctree on ``page``, options and blanks left out."""
    match = re.search(r"^```\{toctree\}\n(.*?)^```", page.read_text(), re.M | re.S)
    assert match, page
    return [
        line.strip().lstrip("/")
        for line in match.group(1).splitlines()
        if line.strip() and not line.startswith(":")
    ]


@pytest.mark.parametrize("root", ["index.md", "manual.md"])
def test_the_six_top_level_sections_are_in_the_family_order(root):
    entries = [entry for entry in _toctree(DOCS / root) if entry in SECTIONS]
    assert entries == SECTIONS
