"""The reader, against files this package did not write.

`tests/seq/` holds the reference corpus: one sequence at seven revisions,
the MATLAB toolbox's own output beside the Python toolbox's, and a spread of
real acquisitions -- EPI, TSE, HASTE, MPRAGE, UTE, radial, labels and soft
delays. A reader tested only against files its own writer produced is tested
against its own assumptions, so these are the tests of record for reading.

What is compared is the file, normalised the way the reference suite
normalises it: the signature goes, since it covers bytes that legitimately
move, and so does trailing whitespace. Two lines are excluded and named
below, because they say who wrote the file rather than what the sequence is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pypulseqpp import _ext

CORPUS = Path(__file__).parent / "seq"

#: Files a revision behind, which are read by converting them: the columns
#: moved and three fields are missing, so these are not merely parsed.
OLDER_THAN_1_5 = sorted(path.name for path in CORPUS.glob("simple_mprage14*.seq"))

#: Older still, and refused: below 1.4 a gradient carries no time shape and a
#: block's duration is an index into a section the format no longer has.
OLDER_THAN_1_4 = sorted(path.name for path in CORPUS.glob("simple_mprage1[23]*.seq"))

#: Every file the reader is expected to take as it stands.
CURRENT = sorted(
    path.name
    for path in CORPUS.glob("*.seq")
    if path.name not in OLDER_THAN_1_5 and path.name not in OLDER_THAN_1_4
)


def rows(text: bytes, section: str):
    """The data lines of one section of a written file."""
    out, inside = [], False
    for line in text.decode().splitlines():
        if line.startswith("["):
            inside = line == section
        elif inside and line and not line.startswith("#"):
            out.append(line)
    return out


def normalise(text: bytes) -> str:
    """A file's content, without what a rewrite is allowed to move.

    Four things go. Comments, because they are the writing toolbox's own
    prose -- this corpus holds files from two of them and MATLAB's differ from
    the Python ones word for word, so keeping comments would compare who wrote
    a file rather than what is in it. The signature, because it covers the
    bytes above it and so moves whenever any of them do. And `TotalDuration`,
    which every writer recomputes from the blocks, so a file that never
    carried one gains it. And the revision, which says which revision of the
    format the writer implements rather than anything about the sequence, so
    a 1.5.0 file read here goes back out as 1.5.1.

    What is left is every line that says what the sequence plays.
    """
    out = text.decode().replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n\[SIGNATURE\][\s\S]*$", "", out)
    out = re.sub(r"^#.*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"^revision \d+\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"^TotalDuration .*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + "\n"


#: Files a rewrite here cannot reproduce byte for byte, and why.
#:
#: A soft delay's offset is written `%g` by the toolbox that defines the
#: format and `%.0f` by the other one, so a whole number of microseconds
#: comes out as `1e+06` there and `1000000` here. These files were written
#: the second way. What is held for them instead is that writing is
#: idempotent, which is the test below.
WRITTEN_WITH_OTHER_CONVENTIONS = {"seq6.seq"}


@pytest.fixture(params=CURRENT, ids=lambda name: name[:-4])
def corpus_file(request):
    """Each current-revision file of the corpus in turn."""
    return CORPUS / request.param


def test_the_corpus_is_here(corpus_file):
    """A guard: an empty glob would make every test below vacuous."""
    assert len(CURRENT) >= 30
    assert corpus_file.stat().st_size > 0


def test_a_file_is_written_back_as_it_was_read(corpus_file):
    """The reader and the writer are each other's inverse on real files."""
    if corpus_file.name in WRITTEN_WITH_OTHER_CONVENTIONS:
        pytest.skip("written with the other toolbox's conventions; see below")
    contents = corpus_file.read_bytes()

    written = _ext.write_text(_ext.read(contents), True)

    assert normalise(written) == normalise(contents)


def test_writing_a_file_twice_writes_the_same_bytes(corpus_file):
    """Reading back what we wrote changes nothing, whoever wrote the source.

    This is what still holds for a file written with another toolbox's
    formatting: our conventions may differ from the source's, but applying
    them twice has to be the same as applying them once.
    """
    once = _ext.write_text(_ext.read(corpus_file.read_bytes()), True)

    twice = _ext.write_text(_ext.read(once), True)

    assert twice == once


def test_a_file_survives_the_binary_form(corpus_file):
    """Everything but the shapes, which the binary form holds in float32."""
    text = _ext.write_text(_ext.read(corpus_file.read_bytes()), True)

    through_binary = _ext.write_text(
        _ext.read(_ext.write_binary(_ext.read(corpus_file.read_bytes()))), True
    )

    def comparable(rendered):
        return re.sub(r"\n\[SHAPES\][\s\S]*$", "", normalise(rendered))

    assert comparable(through_binary) == comparable(text)


def test_reading_a_file_splits_it_into_definitions_and_instances(corpus_file):
    """A sequence off disk is forked, whoever wrote the file."""
    sequence = _ext.read(corpus_file.read_bytes())

    definitions = sequence.instance_definitions()
    assert len(definitions) == sequence.num_blocks()
    if sequence.num_blocks():
        assert 1 <= definitions.min() <= definitions.max()
        assert definitions.max() <= sequence.num_block_definitions()


def test_a_signature_the_file_carries_verifies(corpus_file):
    """Every file that was signed was signed over the bytes it still has."""
    contents = corpus_file.read_bytes()
    if b"[SIGNATURE]" not in contents:
        pytest.skip("this file carries no signature")

    _ext.read(contents, True)


@pytest.mark.parametrize("name", OLDER_THAN_1_4)
def test_a_file_older_than_1_4_is_named_rather_than_misread(name):
    """Saying so beats reading columns that are not where they look."""
    with pytest.raises(RuntimeError, match="1.4.0 is the oldest"):
        _ext.read((CORPUS / name).read_bytes())


# --------------------------------------------------------------------------
# Reading a 1.4 file.
#
# The corpus holds one sequence written at 1.4.0, 1.4.1, 1.4.2 and 1.5.0, so
# what the 1.5.0 file says is what the others have to be read as. Three
# fields are missing from the older ones and each is recovered rather than
# defaulted: an RF pulse's centre from its own envelope, and an arbitrary
# gradient's first and last sample from walking the block table.
# --------------------------------------------------------------------------

AS_1_5 = CORPUS / "simple_mprage150.seq"


def rendered(path):
    return normalise(_ext.write_text(_ext.read(path.read_bytes()), True))


@pytest.mark.parametrize("name", OLDER_THAN_1_5)
def test_a_1_4_file_holds_the_libraries_the_1_5_file_holds(name):
    older = _ext.read((CORPUS / name).read_bytes())
    current = _ext.read(AS_1_5.read_bytes())

    for count in ("num_blocks", "num_rf", "num_gradients", "num_adc", "num_shapes"):
        assert getattr(older, count)() == getattr(current, count)(), count


@pytest.mark.parametrize("name", OLDER_THAN_1_5)
def test_a_1_4_file_reads_as_the_1_5_file_but_for_what_it_cannot_carry(name):
    """Two things it cannot carry, and nothing else differs.

    A 1.4 file has no `use` column, so what each pulse is *for* is unknown
    and stays undefined rather than being guessed from its flip angle. And
    one gradient's `last` is derived by the extrapolation the factory would
    have made, which is close to but not the value the design knew: the
    reference toolbox derives the same number from the same file, so this is
    the format's limit rather than a difference between readers.
    """
    differing = [
        (before, after)
        for before, after in zip(
            rendered(AS_1_5).splitlines(),
            rendered(CORPUS / name).splitlines(),
            strict=False,
        )
        if before != after
    ]

    unknown_use = [
        pair
        for pair in differing
        if pair[0][:-1] == pair[1][:-1] and pair[1][-1] == "u"
    ]
    derived_edge = [pair for pair in differing if pair not in unknown_use]

    assert len(unknown_use) == _ext.read(AS_1_5.read_bytes()).num_rf()
    assert len(derived_edge) == 1


@pytest.mark.parametrize("name", OLDER_THAN_1_5)
def test_a_1_4_file_says_1_5_once_it_has_been_read(name):
    """It has been converted, so it is a 1.5 sequence and says so."""
    converted = _ext.write_text(_ext.read((CORPUS / name).read_bytes()), True)

    assert rows(converted, "[VERSION]") == ["major 1", "minor 5", "revision 1"]


@pytest.mark.parametrize("name", OLDER_THAN_1_5)
def test_a_converted_file_writes_the_same_bytes_twice(name):
    once = _ext.write_text(_ext.read((CORPUS / name).read_bytes()), True)

    assert _ext.write_text(_ext.read(once), True) == once


def test_the_corpus_spans_the_revisions_it_is_here_for():
    """A guard on the corpus itself, so a lost file is noticed."""
    declared = set()
    for path in CORPUS.glob("*.seq"):
        text = path.read_text(errors="replace")
        version = re.search(
            r"\[VERSION\]\s*\nmajor (\d+)\s*\nminor (\d+)\s*\nrevision (\d+)", text
        )
        assert version, path.name
        declared.add(".".join(version.groups()))

    assert {"1.2.0", "1.3.0", "1.3.1", "1.4.0", "1.4.1", "1.4.2"} <= declared
    assert {"1.5.0", "1.5.1"} <= declared
