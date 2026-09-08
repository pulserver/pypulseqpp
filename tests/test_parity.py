"""The file pypulseqpp writes is the file the reference toolbox writes.

This is the invariant the package rests on. A design script that ran against
`pypulseq-matlab-like` -- the transcription of MATLAB Pulseq this package
treats as the authority for the format -- has to produce the same sequence
here, and "the same" means the bytes, not a tolerance: an interpreter reads
the file, and a file that differs is a different scan.

Nothing is excused: what is compared is the bytes.

Both deduplication modes are checked. Collapsing identical library rows
renumbers every reference to them, so a sequence that agrees before
deduplication and disagrees after has a renumbering bug rather than a writing
bug, and the two cases separate those.
"""

import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert
import reference

from pypulseqpp import _ext

SIGNATURE_MARKER = "\n[SIGNATURE]\n"

#: Sequences whose block *numbering* differs, and only that.
#:
#: A block id is a label: the file writes it, a reader indexes by it, and
#: nothing else refers to it. The reference toolbox keys its blocks in a
#: dictionary, so dropping one leaves a gap the numbering keeps --
#: `gre_with_noise_scan` adds a delay of the system's RF dead time, which
#: defaults to zero, and that degenerate block goes while the ids after it
#: stay where they were. This package numbers blocks by position, so it
#: closes the gap. Every row is otherwise identical, in the same order.
NUMBERED_AROUND_A_DROPPED_BLOCK = {"gre_with_noise_scan"}


def written_by_reference(seq, path, *, deduplicate):
    seq.write(str(path), remove_duplicates=deduplicate)
    return path.read_text()


def written_by_core(seq, *, deduplicate):
    core = convert.to_core(seq)
    if deduplicate:
        core.remove_duplicates()
    return _ext.write_text(core, True).decode()


@pytest.mark.parametrize("deduplicate", [False, True], ids=["as-built", "deduplicated"])
def test_the_written_file_is_byte_identical_to_the_reference(
    build_reference, reference_name, deduplicate, tmp_path
):
    if reference_name in NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip(
            "the reference leaves a gap in the block numbering; see the test below"
        )

    expected = written_by_reference(
        build_reference(), tmp_path / "reference.seq", deduplicate=deduplicate
    )
    assert written_by_core(build_reference(), deduplicate=deduplicate) == expected


@pytest.mark.parametrize("name", sorted(NUMBERED_AROUND_A_DROPPED_BLOCK))
def test_only_the_block_numbering_differs_where_a_block_was_dropped(name, tmp_path):
    """Same rows, same order, renumbered from one.

    The block id is the first column of `[BLOCKS]` and nothing refers to it,
    so a gap in it carries no information about the sequence. What has to be
    the same is everything after that column.
    """
    theirs = written_by_reference(
        reference.ZOO[name](), tmp_path / "reference.seq", deduplicate=False
    )
    ours = written_by_core(reference.ZOO[name](), deduplicate=False)

    def rows_without_their_numbers(text):
        inside = False
        out = []
        for line in text.splitlines():
            if line.startswith("["):
                inside = line == "[BLOCKS]"
            elif inside and line and not line.startswith("#"):
                out.append(line.split(maxsplit=1)[1])
        return out

    assert rows_without_their_numbers(ours) == rows_without_their_numbers(theirs)
    assert ours.count("\n") == theirs.count("\n")
