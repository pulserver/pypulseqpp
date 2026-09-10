"""Stored Pulseq revision after construction and legacy-file conversion."""

import pypulseqpp as pp


def test_a_sequence_built_here_is_what_this_package_writes():
    sequence = pp.Sequence()

    assert (sequence.version_major, sequence.version_minor) == (1, 5)
    assert sequence.version_revision >= 1


def test_an_older_file_is_held_as_the_revision_it_was_converted_to(tmp_path):
    """A 1.4.1 file is read *into* 1.5: what it lacks is derived on the way in.

    So what the sequence says afterwards is what it is, not what the file
    said. There is no column in 1.4 for an RF pulse's centre or a gradient's
    first and last sample, and those are worked out as the file is read.
    """
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3))
    path = tmp_path / "old.seq"
    sequence.write_v141(str(path))
    assert "minor 4" in path.read_text()

    reread = pp.Sequence(pp.Opts())
    reread.read(str(path))

    assert (reread.version_major, reread.version_minor) == (1, 5)


def test_the_version_is_three_numbers_a_scanner_can_compare():
    """Pulseq compares them as one: major * 1e6 + minor * 1e3 + revision."""
    sequence = pp.Sequence()

    combined = (
        sequence.version_major * 1000000
        + sequence.version_minor * 1000
        + sequence.version_revision
    )

    assert combined >= 1005001


def test_a_sequence_can_be_told_what_it_is():
    sequence = pp.Sequence()

    sequence._native.set_version(1, 4, 1)

    assert (
        sequence.version_major,
        sequence.version_minor,
        sequence.version_revision,
    ) == (
        1,
        4,
        1,
    )
