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


#: The two ways a sequence reaches the core. Both must produce the same file:
#: one registers a row at a time, the other hands each library over as one
#: array, which is the path a protocol-scale scan takes.
LOADERS = {"row-by-row": convert.to_core, "bulk": convert.to_core_bulk}


def written_by_core(seq, *, deduplicate, load=convert.to_core):
    core = load(seq)
    if deduplicate:
        core.remove_duplicates()
    return _ext.write_text(core, True).decode()


@pytest.fixture(params=sorted(LOADERS), ids=lambda name: name)
def load(request):
    return LOADERS[request.param]


@pytest.mark.parametrize("deduplicate", [False, True], ids=["as-built", "deduplicated"])
def test_the_written_file_is_byte_identical_to_upstream(
    build_reference, load, deduplicate, tmp_path
):
    expected = written_by_upstream(
        build_reference(), tmp_path / "upstream.seq", deduplicate=deduplicate
    )
    assert (
        written_by_core(build_reference(), deduplicate=deduplicate, load=load)
        == expected
    )


def test_both_ways_of_loading_a_sequence_agree(build_reference):
    row_by_row = written_by_core(
        build_reference(), deduplicate=False, load=convert.to_core
    )
    in_bulk = written_by_core(
        build_reference(), deduplicate=False, load=convert.to_core_bulk
    )
    assert in_bulk == row_by_row


def test_a_bulk_load_fills_every_library(build_reference):
    core = convert.to_core_bulk(build_reference())
    reference_core = convert.to_core(build_reference())

    counts = (
        "num_blocks",
        "num_rf",
        "num_gradients",
        "num_adc",
        "num_shapes",
        "num_label_set",
        "num_label_inc",
        "num_extensions",
        "num_soft_delays",
    )
    assert [getattr(core, name)() for name in counts] == [
        getattr(reference_core, name)() for name in counts
    ]


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
