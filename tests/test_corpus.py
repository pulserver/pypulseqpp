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

#: Files declaring a revision older than 1.5.0. Reading them is a separate
#: job -- the columns moved, and what is missing has to be derived -- so they
#: are collected here rather than skipped one by one.
OLDER_THAN_1_5 = sorted(
    path.name for path in CORPUS.glob("simple_mprage1[234]*.seq")
)

#: Every file the reader is expected to take as it stands.
CURRENT = sorted(
    path.name for path in CORPUS.glob("*.seq") if path.name not in OLDER_THAN_1_5
)


def normalise(text: bytes) -> str:
    """A file's content, without what a rewrite is allowed to move.

    Three things go. Comments, because they are the writing toolbox's own
    prose -- this corpus holds files from two of them and MATLAB's differ from
    the Python ones word for word, so keeping comments would compare who wrote
    a file rather than what is in it. The signature, because it covers the
    bytes above it and so moves whenever any of them do. And `TotalDuration`,
    which every writer recomputes from the blocks, so a file that never
    carried one gains it.

    What is left is every line that says what the sequence plays.
    """
    out = text.decode().replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n\[SIGNATURE\][\s\S]*$", "", out)
    out = re.sub(r"^#.*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"^TotalDuration .*\n", "", out, flags=re.MULTILINE)
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.rstrip() + "\n"


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
    contents = corpus_file.read_bytes()

    written = _ext.write_text(_ext.read(contents), True)

    assert normalise(written) == normalise(contents)


def test_a_file_survives_the_binary_form(corpus_file):
    """Everything but the shapes, which the binary form holds in float32.

    The version goes too: a binary file declares at least 1.5.1, the revision
    the format arrived in, so a 1.5.0 file comes back saying so.
    """
    text = _ext.write_text(_ext.read(corpus_file.read_bytes()), True)

    through_binary = _ext.write_text(
        _ext.read(_ext.write_binary(_ext.read(corpus_file.read_bytes()))), True
    )

    def comparable(rendered):
        out = re.sub(r"\n\[SHAPES\][\s\S]*$", "", normalise(rendered))
        return re.sub(r"^revision \d+\n", "", out, flags=re.MULTILINE)

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


@pytest.mark.parametrize("name", OLDER_THAN_1_5)
def test_a_file_older_than_1_5_is_named_rather_than_misread(name):
    """Until the columns are converted, saying so beats reading them wrong."""
    with pytest.raises(RuntimeError, match="1.5.0 is the oldest"):
        _ext.read((CORPUS / name).read_bytes())


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
