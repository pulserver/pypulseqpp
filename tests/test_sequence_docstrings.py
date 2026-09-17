"""Every shipped sequence documents its return value in a section of its own."""

import inspect

import pytest
from sphinx.ext.napoleon import Config
from sphinx.ext.napoleon.docstring import NumpyDocstring

import pypulseqpp as pp
import pypulseqpp.sequences as sequences
from pypulseqpp.sequences._app import _split_sections

#: The sections every entry point carries, in the order NumPy prescribes.
LEADING = ("Parameters", "Returns")

#: The only return entry: the designed sequence, and nothing else.
RETURNS = "pypulseqpp.Sequence\n    The designed sequence."


@pytest.fixture(params=sequences.ZOO)
def entry_point(request):
    """One shipped sequence's module-level entry point."""
    return getattr(sequences, request.param)


def test_an_entry_point_returns_a_sequence(entry_point):
    assert inspect.signature(entry_point).return_annotation is pp.Sequence


def test_the_returns_section_holds_the_return_value_and_nothing_else(entry_point):
    sections = dict(_split_sections(entry_point.__doc__ or "")[2])
    assert sections["Returns"] == RETURNS


def test_the_sections_are_independent_and_ordered(entry_point):
    order = [name for name, _ in _split_sections(entry_point.__doc__ or "")[2]]
    assert order[: len(LEADING)] == list(LEADING)
    assert len(order) == len(set(order))
    assert order[-1] == "Examples"
    assert set(order) <= {"Parameters", "Returns", "Raises", "Notes", "Examples"}


def test_the_description_precedes_the_parameters(entry_point):
    """A class's extended description belongs above Parameters, not below Returns."""
    summary, description, _ = _split_sections(entry_point.__doc__ or "")
    assert summary and not summary.endswith("-")
    assert "\n\n" not in summary
    assert "Returns" not in description


def test_the_docstring_parses_as_numpy(entry_point):
    """Napoleon turns each section into its own field list rather than one blob."""
    rendered = str(NumpyDocstring(entry_point.__doc__ or "", Config()))
    assert ":returns: **pypulseqpp.Sequence" in rendered or ":returns:" in rendered
    if "Raises" in dict(_split_sections(entry_point.__doc__ or "")[2]):
        assert ":raises" in rendered
    assert ".. rubric:: Examples" in rendered or "Examples\n" in rendered
