"""The Parameters section as the command line and SequenceApp.parameters read it."""

from pypulseqpp._prescription import accepts_none, documented, scalar

DOC = """Summary.

Parameters
----------
n_x, n_y, \\
n_z : int, default=64
    Matrix size. Along
    each axis.
te : float | None, default=None
    Echo time (s).
n_x : int
    Described again, and ignored.
flag : bool

Raises
------
ValueError
    Never.
"""


def test_the_parameters_section_ends_at_the_next_heading():
    assert list(documented(DOC)) == ["n_x", "n_y", "n_z", "te"]


def test_names_on_a_wrapped_header_share_its_type_and_description():
    assert documented(DOC)["n_z"] == (
        "int, default=64",
        "Matrix size. Along each axis.",
    )
    assert documented(DOC)["n_x"] == documented(DOC)["n_z"]


def test_a_name_without_a_description_is_left_out():
    assert "flag" not in documented(DOC)


def test_a_docstring_without_parameters_documents_nothing():
    assert documented("Summary.\n\nReturns\n-------\nint\n    One.\n") == {}
    assert documented(None) == {}


def test_a_scalar_is_read_from_an_annotation_or_a_union():
    assert scalar(float | None) is float
    assert scalar(tuple) is None
    assert accepts_none(int | None)
    assert not accepts_none(int)
