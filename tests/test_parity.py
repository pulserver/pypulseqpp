"""The file pypulseqpp writes is the file PyPulseq writes.

This is the invariant the package rests on. A design script that ran against
the reference toolbox has to produce the same sequence here, and "the same"
means the bytes, not a tolerance: an interpreter reads the file, and a file
that differs is a different scan.

Both deduplication modes are checked. Collapsing identical library rows
renumbers every reference to them, so a sequence that agrees before
deduplication and disagrees after has a renumbering bug rather than a writing
bug, and the two cases separate those.
"""

import hashlib

import convert
import pytest

from pypulseqpp import _ext

SIGNATURE_MARKER = "\n[SIGNATURE]\n"


def written_by_upstream(seq, path, *, deduplicate):
    seq.write(str(path), remove_duplicates=deduplicate)
    return path.read_text()


def written_by_core(seq, *, deduplicate):
    core = convert.to_core(seq)
    if deduplicate:
        core.remove_duplicates()
    return _ext.write_text(core, True).decode()


@pytest.mark.parametrize("deduplicate", [False, True], ids=["as-built", "deduplicated"])
def test_the_written_file_is_byte_identical_to_upstream(
    build_reference, deduplicate, tmp_path
):
    expected = written_by_upstream(
        build_reference(), tmp_path / "upstream.seq", deduplicate=deduplicate
    )
    assert written_by_core(build_reference(), deduplicate=deduplicate) == expected


def test_the_signature_is_the_digest_of_everything_above_it(build_reference):
    written = written_by_core(build_reference(), deduplicate=True)

    body, _, trailer = written.partition(SIGNATURE_MARKER)
    assert trailer, "the writer produced no [SIGNATURE] section"

    recorded = next(
        line.split()[1] for line in trailer.splitlines() if line.startswith("Hash ")
    )
    # MD5 is what the file format specifies for `[SIGNATURE]`, so it is the
    # algorithm under test rather than a security choice.
    assert recorded == hashlib.md5(body.encode(), usedforsecurity=False).hexdigest()


def test_deduplication_never_changes_the_number_of_blocks(build_reference):
    core = convert.to_core(build_reference())
    before = core.num_blocks()
    core.remove_duplicates()
    assert core.num_blocks() == before


def test_deduplication_never_changes_the_duration(build_reference):
    core = convert.to_core(build_reference())
    before = core.duration()
    core.remove_duplicates()
    assert core.duration() == pytest.approx(before)
