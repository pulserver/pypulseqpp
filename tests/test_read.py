"""Reading a `.seq` file back, and getting the sequence that wrote it.

The reader is the writer's inverse, so the test of record is a round trip
rather than a set of expected values: a file written here, read back and
written again is the file it started as, byte for byte. A column scaled by
the wrong power of ten fails that comparison, where a test written against
numbers someone typed in would only fail if they had typed the right ones.

The stronger half is the same trip through the reference toolbox. Upstream
writes the file, we read it, and what we write back is compared with what
upstream wrote -- so the reader is held against the format as another
implementation produces it, not only against our own writer.
"""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import reference
from convert import to_core

from pypulseqpp import _ext


def written(name):
    """The reference sequence ``name``, as our writer produces it."""
    return _ext.write_text(to_core(reference.ZOO[name]()), True)


def test_a_file_read_back_writes_the_same_bytes(reference_name, build_reference):
    first = _ext.write_text(to_core(build_reference()), True)

    second = _ext.write_text(_ext.read(first), True)

    assert second == first


def test_a_file_the_reference_wrote_reads_into_the_same_sequence(
    reference_name, build_reference, tmp_path
):
    """Read what another toolbox wrote, and write back the bytes it wrote."""
    import test_parity

    if reference_name in test_parity.NUMBERED_AROUND_A_DROPPED_BLOCK:
        pytest.skip("the reference leaves a gap in the block numbering")

    path = tmp_path / f"{reference_name}.seq"
    build_reference().write(str(path))
    theirs = path.read_bytes()

    assert _ext.write_text(_ext.read(theirs), True) == theirs


def test_reading_from_disk_matches_reading_from_memory(build_reference, tmp_path):
    contents = _ext.write_text(to_core(build_reference()), True)
    path = tmp_path / "sequence.seq"
    path.write_bytes(contents)

    assert _ext.write_text(_ext.read_file(str(path)), True) == contents


def test_reading_leaves_the_sequence_split_as_building_it_would(
    reference_name, build_reference
):
    """A sequence off disk is forked like one that was built, not left flat.

    The structure survives exactly: definitions are made of shape ids and
    times, and the file carries both without loss.
    """
    built = to_core(build_reference())

    loaded = _ext.read(_ext.write_text(built, True))

    assert loaded.num_block_definitions() == built.num_block_definitions()
    np.testing.assert_array_equal(
        loaded.instance_definitions(), built.instance_definitions()
    )
    np.testing.assert_array_equal(
        loaded.instance_adc_definitions(), built.instance_adc_definitions()
    )


def test_the_per_playout_numbers_come_back_to_the_precision_the_file_holds(
    reference_name, build_reference
):
    """An amplitude is written to six significant digits, so that is the floor."""
    built = to_core(build_reference())

    loaded = _ext.read(_ext.write_text(built, True))

    np.testing.assert_allclose(
        loaded.instance_parameters(), built.instance_parameters(), rtol=1e-5
    )


def test_a_collapsed_sequence_comes_back_with_its_numbers_unchanged(
    reference_name, build_reference
):
    """Deduplication rounds to what the file can carry, so nothing is lost after it."""
    built = to_core(build_reference())
    built.remove_duplicates()

    loaded = _ext.read(_ext.write_text(built, True))

    np.testing.assert_array_equal(
        loaded.instance_parameters(), built.instance_parameters()
    )


def test_the_libraries_come_back_the_size_they_went_in(build_reference):
    built = to_core(build_reference())

    loaded = _ext.read(_ext.write_text(built, True))

    for count in (
        "num_blocks",
        "num_rf",
        "num_gradients",
        "num_adc",
        "num_shapes",
        "num_extensions",
        "num_triggers",
        "num_rotations",
        "num_label_set",
        "num_label_inc",
        "num_soft_delays",
    ):
        assert getattr(loaded, count)() == getattr(built, count)(), count


def test_a_signed_file_verifies():
    _ext.read(written("spin_echo"), True)


def test_a_file_whose_bytes_moved_does_not_verify():
    contents = written("spin_echo")
    tampered = contents.replace(b"[BLOCKS]\n1 ", b"[BLOCKS]\n1  ", 1)
    assert tampered != contents

    with pytest.raises(RuntimeError, match="signature does not match"):
        _ext.read(tampered, True)


def test_a_file_carrying_no_signature_is_read_either_way():
    unsigned = _ext.write_text(to_core(reference.ZOO["spin_echo"]()), False)

    assert b"[SIGNATURE]" not in unsigned
    assert _ext.write_text(_ext.read(unsigned, True), False) == unsigned


def test_verification_is_off_unless_it_is_asked_for():
    """A file may be read to look at, whatever its signature says."""
    tampered = written("spin_echo").replace(b"[BLOCKS]\n1 ", b"[BLOCKS]\n1  ", 1)

    _ext.read(tampered)


def test_a_file_older_than_the_format_says_so_rather_than_reading_wrong():
    """1.2.0 is where the format is defined from; below it there is nothing."""
    contents = written("spin_echo").replace(b"minor 5", b"minor 1", 1)

    with pytest.raises(RuntimeError, match=r"1\.2\.0 is the oldest"):
        _ext.read(contents)


def test_a_malformed_row_names_the_line_it_is_on():
    contents = written("spin_echo").replace(b"[ADC]\n", b"[ADC]\nzz 1 2 3\n", 1)

    with pytest.raises(RuntimeError, match=r"line \d+: expected an ADC id"):
        _ext.read(contents)


def test_a_block_naming_an_event_that_was_never_registered_is_refused():
    """A gap in the numbering would silently shift every id after it."""
    contents = written("gauss_pulses").replace(b"shape_id 1\n", b"shape_id 2\n", 1)

    with pytest.raises(RuntimeError, match="registered as"):
        _ext.read(contents)


def test_the_rasters_come_from_the_file_rather_than_from_a_default():
    built = to_core(reference.ZOO["spin_echo"]())
    contents = _ext.write_text(built, True)

    loaded = _ext.read(contents)

    definitions = loaded.definitions()
    for key in (
        "GradientRasterTime",
        "RadiofrequencyRasterTime",
        "AdcRasterTime",
        "BlockDurationRaster",
    ):
        assert key in definitions
    # Block durations are stored as raster ticks, so a raster read wrongly
    # would show up as every duration being wrong by the same factor.
    np.testing.assert_allclose(loaded.block_durations(), built.block_durations())


def test_a_text_definition_stays_text_and_numbers_stay_numbers():
    sequence = _ext.Sequence()
    sequence.set_definition("Name", "gre")
    sequence.set_definition("FOV", [0.256, 0.256, 0.005])
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)

    loaded = _ext.read(_ext.write_text(sequence, True))

    assert loaded.definitions()["Name"] == "gre"
    np.testing.assert_allclose(loaded.definitions()["FOV"], [0.256, 0.256, 0.005])
