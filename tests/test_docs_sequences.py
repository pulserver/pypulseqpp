"""Every shipped sequence is classified once and documented from its own docstring."""

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
