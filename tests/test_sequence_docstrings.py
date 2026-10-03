"""Every shipped sequence documents its return value in a section of its own."""

import inspect

import pytest
from sphinx.ext.napoleon import Config
from sphinx.ext.napoleon.docstring import NumpyDocstring

import pypulseqpp as pp
import pypulseqpp.sequences as sequences

#: The sections every entry point carries, in the order NumPy prescribes.
LEADING = ("Parameters", "Returns")

#: The only return entry of a function that designs one sequence.
RETURNS = "pypulseqpp.Sequence\n    The designed sequence."

#: The type of the only return entry of a function that returns a chain of
#: sequences, prescans first and the main sequence last.
CHAIN_RETURNS = "list of pypulseqpp.Sequence"


def split_sections(doc):
    """Split a NumPy docstring into its summary, its description and its sections.

    The sections are ``(heading, body)`` pairs in the order they appear. Text
    before the first heading that is not the summary is the extended description.
    """
    lines = doc.splitlines()
    headings = [
        (i, line.strip())
        for i, line in enumerate(lines[:-1])
        if line.strip() and line.strip() == line and set(lines[i + 1].strip()) == {"-"}
    ]
    bounds = [i for i, _ in headings] + [len(lines)]
    sections = [
        (name, "\n".join(lines[start + 2 : stop]).strip("\n"))
        for (start, name), stop in zip(headings, bounds[1:], strict=True)
    ]
    head = "\n".join(lines[: bounds[0]]).strip()
    summary, _, description = head.partition("\n\n")
    return summary, description.strip(), sections


@pytest.fixture(params=sequences.ZOO)
def entry_point(request):
    """One shipped sequence's module-level entry point."""
    return getattr(sequences, request.param)


def returns_a_chain(entry_point):
    """Whether the entry point returns a list of sequences, not one sequence."""
    signature = inspect.signature(entry_point.main, eval_str=True)
    return signature.return_annotation == list[pp.Sequence]


def test_an_entry_point_returns_a_sequence_or_a_chain_of_them(entry_point):
    signature = inspect.signature(entry_point.main, eval_str=True)

    assert signature.return_annotation in (pp.Sequence, list[pp.Sequence])


def test_the_returns_section_holds_the_return_value_and_nothing_else(entry_point):
    sections = dict(split_sections(inspect.getdoc(entry_point) or "")[2])
    if returns_a_chain(entry_point):
        kind, *description = sections["Returns"].splitlines()
        assert kind == CHAIN_RETURNS
        assert description
        assert all(line.startswith("    ") for line in description)
    else:
        assert sections["Returns"] == RETURNS


def test_the_sections_are_independent_and_ordered(entry_point):
    order = [name for name, _ in split_sections(inspect.getdoc(entry_point) or "")[2]]
    assert order[: len(LEADING)] == list(LEADING)
    assert len(order) == len(set(order))
    assert order[-1] == "Examples"
    assert set(order) <= {"Parameters", "Returns", "Raises", "Notes", "Examples"}


def test_the_description_precedes_the_parameters(entry_point):
    """The extended description belongs above Parameters, not below Returns."""
    summary, description, _ = split_sections(inspect.getdoc(entry_point) or "")
    assert summary and not summary.endswith("-")
    assert "\n\n" not in summary
    assert "Returns" not in description


def test_the_docstring_parses_as_numpy(entry_point):
    """Napoleon turns each section into its own field list rather than one blob."""
    doc = inspect.getdoc(entry_point) or ""
    rendered = str(NumpyDocstring(doc, Config()))
    returns = rendered.partition(":returns:")[2].partition("\n:")[0]
    assert "The designed sequence" in returns
    assert ">>>" not in returns
    kind = CHAIN_RETURNS if returns_a_chain(entry_point) else "pypulseqpp.Sequence"
    assert f":rtype: {kind}\n" in rendered
    if "Raises" in dict(split_sections(doc)[2]):
        assert ":raises" in rendered
    assert ".. rubric:: Examples" in rendered
