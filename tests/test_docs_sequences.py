"""Every shipped sequence is classified once and documented from its own docstring."""

import ast
import importlib
import sys
from pathlib import Path

import pytest

import pypulseqpp.sequences as sequences

sys.path.insert(0, str(Path(__file__).parents[1] / "docs"))

import sequence_reference

CATALOGUE = Path(__file__).parents[1] / "docs/sequences.md"
GALLERY = Path(__file__).parents[1] / "gallery"


def test_the_catalogue_classifies_every_shipped_sequence_exactly_once():
    classified = [doc.module for doc in sequence_reference.SEQUENCES]
    assert sorted(classified) == sorted(sequences.ZOO)
    assert len(classified) == len(set(classified))


def test_every_family_has_a_heading_and_a_table_in_the_catalogue():
    page = CATALOGUE.read_text()
    families = {doc.family for doc in sequence_reference.SEQUENCES}
    assert families == set(sequence_reference.FAMILIES)
    for family in sequence_reference.FAMILIES:
        stem = family.lower().replace(" ", "-")
        assert f"generated/sequences/tables/{stem}.rst" in page
        assert f"## {family}\n" in page


@pytest.mark.parametrize(
    "doc", sequence_reference.SEQUENCES, ids=lambda doc: doc.module
)
def test_a_catalogue_row_repeats_the_application_own_summary_line(doc):
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{doc.module}")
    summary = (module.main.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
    assert sequence_reference.summary(doc) == summary
    assert summary in sequence_reference.catalogue_table(doc.family)


@pytest.mark.parametrize(
    "doc", sequence_reference.SEQUENCES, ids=lambda doc: doc.module
)
def test_every_sequence_has_a_gallery_page_that_designs_it(doc):
    """The reference page links its example by name, so the two have to match."""
    scripts = sorted(GALLERY.rglob(f"{doc.module}.py"))
    assert len(scripts) == 1, scripts


def test_built_in_sequence_galleries_defer_constraint_checks_to_safety_example():
    """Sequence tours do not duplicate the canonical constraint-check workflow."""
    scripts = sorted(GALLERY.glob("1[0-6]-*/*_sequence.py"))
    assert len(scripts) == len(sequence_reference.SEQUENCES)
    for script in scripts:
        source = script.read_text()
        assert "Safety checks" not in source, script
        assert "safety_table(" not in source, script


def test_fast_spin_echo_modes_have_separate_scientific_examples():
    """Fixed, adaptive, and shuffled FSE acquisitions have focused pages."""
    directory = GALLERY / "13-fast-spin-echo"
    assert {path.name for path in directory.glob("*.py")} == {
        "fse3D_sequence.py",
        "fse3D_adaptive.py",
        "fse3D_shuffling.py",
    }


@pytest.mark.parametrize(
    "doc", sequence_reference.SEQUENCES, ids=lambda doc: doc.module
)
def test_every_sequence_gallery_uses_the_automatic_paper_plot_selection(doc):
    """Built-in tours exercise the plotting default rather than hiding its defects."""
    script = next(GALLERY.rglob(f"{doc.module}.py"))
    tree = ast.parse(script.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paper_plot"
    ]

    assert len(calls) == 1
    assert not calls[0].args
    assert not calls[0].keywords
