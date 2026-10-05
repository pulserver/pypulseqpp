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


def _gallery_sections() -> list[str]:
    """The gallery directories in `GALLERY_SECTIONS`, by name."""
    text = (DOCS / "conf.py").read_text()
    block = re.search(r"^GALLERY_SECTIONS = \[(.*?)^\]", text, re.M | re.S)
    assert block
    return re.findall(r'"\.\./gallery/([^"]+)"', block.group(1))


def _table(page: Path, heading: str) -> list[str]:
    """The documents linked from the first column of the table under ``heading``."""
    text = page.read_text()
    match = re.search(rf"^## {heading}\n(.*?)(?=^## |^```|\Z)", text, re.M | re.S)
    assert match, heading
    return re.findall(r"^\| \{doc\}`/?([^`<]+?)`", match.group(1), re.M)


def _gallery_directories(landing: str) -> set[str]:
    """The gallery directories whose pages a landing page's toctree lists."""
    return {
        entry.split("/")[2]
        for entry in _toctree(DOCS / f"{landing}.md")
        if entry.startswith("generated/gallery/")
    }


@pytest.mark.parametrize(
    ("heading", "pattern"),
    [("Course", r"0[1-7]-"), ("Tours", r"08-")],
)
def test_the_examples_landing_tables_cover_the_course_and_the_tours(heading, pattern):
    expected = {name for name in _gallery_sections() if re.match(pattern, name)}
    landings = _table(DOCS / "examples" / "index.md", heading)
    covered = set().union(*(_gallery_directories(page) for page in landings))
    assert expected and covered == expected


def test_every_gallery_section_is_reached_from_the_examples_landing_page():
    landings = _toctree(DOCS / "examples" / "index.md")
    covered = set()
    for landing in landings:
        covered |= _gallery_directories(landing)
        page = DOCS / f"{landing}.md"
        for child in _toctree(page) if "built-in-sequences" in landing else []:
            covered |= _gallery_directories(child)
    assert covered == set(_gallery_sections())


def test_a_section_landing_page_lists_every_script_in_its_gallery_directory():
    for landing in _toctree(DOCS / "examples" / "index.md"):
        for directory in _gallery_directories(landing):
            scripts = {
                f"generated/gallery/{directory}/{path.stem}"
                for path in (DOCS.parent / "gallery" / directory).glob("*.py")
            }
            assert scripts <= set(_toctree(DOCS / f"{landing}.md")), landing
