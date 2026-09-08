"""The file pypulseqpp writes is the file the reference toolbox writes.

This is the invariant the package rests on. A design script that ran against
`pypulseq-matlab-like` -- the transcription of MATLAB Pulseq this package
treats as the authority for the format -- has to produce the same sequence
here, and "the same" means the bytes, not a tolerance: an interpreter reads
the file, and a file that differs is a different scan.

Two lines are excused, and only two. Both are things a writer decides about
itself rather than about the sequence, and the reference suite excuses them
the same way, by allowing a file to match either its source or its own
canonical rewrite.

Both deduplication modes are checked. Collapsing identical library rows
renumbers every reference to them, so a sequence that agrees before
deduplication and disagrees after has a renumbering bug rather than a writing
bug, and the two cases separate those.
"""

import hashlib
import re

import convert
import pytest
import reference

from pypulseqpp import _ext

SIGNATURE_MARKER = "\n[SIGNATURE]\n"

#: Reference sequences carrying a sample small enough for the reference toolbox's
#: deduplication to blunt it. It floors a magnitude at 1e-12 before
#: taking its logarithm, so a sample below that keeps four significant digits
#: where nine were asked for. pypulseqpp does not floor, and the value it
#: writes is the one the pulse plays, so these two disagree after
#: deduplication on purpose. See test_deduplication_keeps_the_precision below.
BLUNTED_BY_UPSTREAM = {"arbitrary_rf"}

#: The revision a file declares. The reference toolbox stamps its own version
#: on everything it writes; this package writes the oldest revision that can
#: read the file back, so a sequence using no 1.5.1 event still says 1.5.0.
#: Keeping what the file declared is what lets a 1.5.0 file read here and
#: write back unchanged, which `test_corpus.py` holds over the whole corpus.
#:
#: `TotalDuration` is the second. The reference toolbox records it in
#: `test_report`, this package in `write`, so a file written here carries a
#: duration the sequence really has and one written there may not carry one.
SELF_DESCRIPTION = re.compile(r"^(revision \d+|TotalDuration .*)\n", re.MULTILINE)


def what_the_sequence_says(text):
    """The file without the two lines a writer says about itself."""
    return SELF_DESCRIPTION.sub("", text)


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
    if deduplicate and reference_name in BLUNTED_BY_UPSTREAM:
        pytest.skip("the reference toolbox rounds a tiny sample away; see the precision test")

    expected = written_by_reference(
        build_reference(), tmp_path / "reference.seq", deduplicate=deduplicate
    )
    ours = written_by_core(build_reference(), deduplicate=deduplicate)

    assert what_the_sequence_says(ours) == what_the_sequence_says(expected)


@pytest.mark.parametrize("name", sorted(BLUNTED_BY_UPSTREAM))
def test_deduplication_keeps_the_precision_the_reference_rounds_away(name, tmp_path):
    """Where the two disagree, ours is the sample the pulse actually plays.

    Upstream's rounding softens a logarithm with a 1e-12 floor, so every
    magnitude below that shares one exponent and nine significant digits keep
    four. A pulse really does carry such samples: a sinc that crosses zero
    leaves one at 4e-17. This holds that the disagreement is only ever that,
    and that the value written here is the one deduplication was given.
    """
    build = reference.ZOO[name]
    upstream = written_by_reference(
        build(), tmp_path / "upstream.seq", deduplicate=True
    ).splitlines()
    ours = written_by_core(build(), deduplicate=True).splitlines()
    as_built = written_by_core(build(), deduplicate=False).splitlines()

    assert len(ours) == len(upstream)
    differing = [
        i for i, (a, b) in enumerate(zip(upstream, ours, strict=True)) if a != b
    ]
    assert differing, "nothing differs, so this sequence no longer belongs here"

    for line in differing:
        if upstream[line].startswith("Hash "):
            continue
        theirs, mine = float(upstream[line]), float(ours[line])
        # Same number, fewer digits: a rounding, not a different value.
        assert mine == pytest.approx(theirs, rel=1e-3)
        # And ours is what the sequence held before deduplication touched it.
        assert ours[line] == as_built[line]


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
