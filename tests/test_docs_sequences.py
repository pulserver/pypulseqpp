"""Every shipped sequence is classified once and documented from its own docstring."""

import importlib
import sys
from pathlib import Path

import pytest

import pypulseqpp.sequences as sequences

sys.path.insert(0, str(Path(__file__).parents[1] / "docs"))

import sequence_pages

CATALOGUE = Path(__file__).parents[1] / "docs/sequences.md"


def test_the_catalogue_classifies_every_shipped_sequence_exactly_once():
    classified = [doc.module for doc in sequence_pages.SEQUENCES]
    assert sorted(classified) == sorted(sequences.ZOO)
    assert len(classified) == len(set(classified))


def test_every_family_has_a_heading_and_a_table_in_the_catalogue():
    page = CATALOGUE.read_text()
    families = {doc.family for doc in sequence_pages.SEQUENCES}
    assert families == set(sequence_pages.FAMILIES)
    for family in sequence_pages.FAMILIES:
        stem = family.lower().replace(" ", "-")
        assert f"generated/sequences/tables/{stem}.rst" in page
        assert f"## {family}\n" in page


@pytest.mark.parametrize("doc", sequence_pages.SEQUENCES, ids=lambda doc: doc.module)
def test_a_catalogue_row_repeats_the_application_own_summary_line(doc):
    module = importlib.import_module(f"pypulseqpp.sequences.sequence.{doc.module}")
    summary = (module.main.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
    assert sequence_pages.summary(doc) == summary
    assert summary in sequence_pages.catalogue_table(doc.family)


@pytest.mark.parametrize("doc", sequence_pages.SEQUENCES, ids=lambda doc: doc.module)
def test_the_documentation_configuration_designs_a_sequence_that_passes_its_timing(doc):
    """A figure is drawn from a designed sequence, so the configuration has to build."""
    seq = sequence_pages.design(doc)
    assert seq.check_timing()[0]
