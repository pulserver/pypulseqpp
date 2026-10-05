"""Explanation pages with sections open with a TL;DR, and gallery pages open in Colab."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"
GALLERY = Path(__file__).parents[1] / "gallery"
EXPLANATIONS = sorted(
    page for page in (DOCS / "explanations").glob("*.md") if page.name != "index.md"
)
TLDR = "```{admonition} TL;DR"

#: Landing, API and example pages, none of which carries a TL;DR.
WITHOUT_TLDR = sorted(
    [
        DOCS / "explanations" / "index.md",
        *(DOCS / "examples").rglob("*.md"),
        *(DOCS / "api").glob("*.md"),
        *GALLERY.rglob("*.py"),
        *GALLERY.rglob("_gallery_header.md"),
    ]
)


def _sections(text: str) -> int:
    """The number of second-level headings outside fenced code."""
    text = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    return len(re.findall(r"^## ", text, flags=re.M))


def test_the_explanation_directory_is_not_empty():
    assert EXPLANATIONS


def test_the_explanations_are_one_flat_list():
    nested = [
        path.name
        for path in (DOCS / "explanations").iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    ]
    assert not nested


@pytest.mark.parametrize(
    "page", EXPLANATIONS, ids=lambda path: str(path.relative_to(DOCS / "explanations"))
)
def test_an_explanation_page_with_sections_opens_with_a_tldr(page):
    """The page's title, then a TL;DR block before anything else.

    A page with a single section may omit it; one that has it places it there.
    """
    text = page.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# "), page
    following = [line for line in lines[1:] if line.strip()]
    opens = following[:2] == [TLDR, ":class: tldr"]
    if _sections(text) > 1:
        assert opens, page
    else:
        assert opens or TLDR not in text, page


@pytest.mark.parametrize(
    "page", WITHOUT_TLDR, ids=lambda path: str(path.relative_to(path.parents[1]))
)
def test_landing_api_and_example_pages_carry_no_tldr(page):
    assert "TL;DR" not in page.read_text(encoding="utf-8"), page


def _colab():
    """``docs/colab.py``, which is not a package module."""
    spec = importlib.util.spec_from_file_location("colab", DOCS / "colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE_PAGE = """.. _sphx_glr_generated_gallery_01-pulseq-basics_01_fid.py:


====================
Free induction decay
====================

The smallest complete Pulseq sequence is a pulse-acquire experiment.
"""


def test_an_example_page_carries_the_colab_badge_under_its_title():
    colab = _colab()
    text = colab.with_badge(
        _EXAMPLE_PAGE, "generated/gallery/01-pulseq-basics/01_fid", "latest"
    )
    title_end = (
        text.index("====================\n\n", text.index("Free induction")) + 22
    )
    badge = text.index("colab-badge.svg")
    assert title_end < badge < text.index("The smallest complete")
    assert "blob/gh-pages/latest/_colab/01-pulseq-basics/01_fid.ipynb" in text
    section = colab.with_badge(
        _EXAMPLE_PAGE, "generated/gallery/01-pulseq-basics/index", "latest"
    )
    assert section == _EXAMPLE_PAGE
    elsewhere = colab.with_badge(
        _EXAMPLE_PAGE, "explanations/pulseq-representation", "latest"
    )
    assert elsewhere == _EXAMPLE_PAGE


def test_the_colab_notebook_is_the_gallery_notebook_after_a_setup_cell(tmp_path):
    """The downloadable notebook is left alone; the Colab copy installs first."""
    colab = _colab()
    notebook = {
        "cells": [{"cell_type": "code", "source": ["import pypulseqpp"]}],
        "nbformat": 4,
    }
    source = (
        tmp_path / "docs" / "generated" / "gallery" / "13-fast-spin-echo" / "fse.ipynb"
    )
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(notebook))

    assert colab.write(tmp_path / "docs", tmp_path / "site", "v1.2.3") == 1
    copy = json.loads(
        (tmp_path / "site" / "_colab" / "13-fast-spin-echo" / "fse.ipynb").read_text()
    )
    assert json.loads(source.read_text()) == notebook
    install = "".join(copy["cells"][1]["source"])
    assert install.startswith("%pip install")
    assert "'pypulseqpp[plot]==1.2.3'" in install and "blochsim" in install
    assert copy["cells"][2:] == notebook["cells"]
    assert (
        "blochsim"
        not in colab.setup_cells("01-pulseq-basics", "latest")[1]["source"][0]
    )
