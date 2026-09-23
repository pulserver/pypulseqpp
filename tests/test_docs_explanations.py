"""Explanation pages open with a TL;DR, and gallery pages open in Colab."""

import importlib.util
import json
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"
EXPLANATIONS = sorted(
    page
    for page in (DOCS / "explanations").rglob("*.md")
    if page.name != "index.md" and not page.parent.name.startswith("_")
)


def test_the_explanation_directory_is_not_empty():
    assert EXPLANATIONS


@pytest.mark.parametrize(
    "page", EXPLANATIONS, ids=lambda path: str(path.relative_to(DOCS / "explanations"))
)
def test_every_explanation_page_opens_with_a_tldr(page):
    """The page's title, then a TL;DR block before anything else."""
    lines = page.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), page
    following = [line for line in lines[1:] if line.strip()]
    assert following[:2] == ["```{admonition} TL;DR", ":class: tldr"], page


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
    elsewhere = colab.with_badge(_EXAMPLE_PAGE, "explanations/pulseq/index", "latest")
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
    assert "'pypulseqpp[plot]==1.2.3'" in install and "torchsim" in install
    assert copy["cells"][2:] == notebook["cells"]
    assert (
        "torchsim"
        not in colab.setup_cells("01-pulseq-basics", "latest")[1]["source"][0]
    )
