"""The binary form of a sequence file.

The same sequence as the text form and the same units; only the container
changes. So the test of record is that the two agree: a sequence written both
ways and read back both ways is one sequence, and what the binary form
cannot carry is named rather than discovered.

There is one thing it cannot carry. Shape samples are single precision where
the text form writes nine significant digits, so a waveform comes back within
a float32 of itself and everything else comes back exactly.
"""

import numpy as np
import pytest
from convert import to_core

from pypulseqpp import _ext


def rows(text, section):
    """The data lines of one section, comments and blanks dropped."""
    out, inside = [], False
    for line in text.decode().splitlines():
        if line.startswith("["):
            inside = line == section
        elif inside and line and not line.startswith("#"):
            out.append(line)
    return out


def sections(text):
    return {line for line in text.decode().splitlines() if line.startswith("[")}


def test_the_binary_form_is_recognised_by_what_is_in_it(build_reference):
    """A reader is handed bytes, not a file name, so the bytes have to say."""
    core = to_core(build_reference())

    assert _ext.is_binary(_ext.write_binary(core))
    assert not _ext.is_binary(_ext.write_text(core, True))


def test_a_binary_file_carries_the_same_sequence_as_the_text_one(
    reference_name, build_reference
):
    """Everything but the shapes, which are single precision -- see below.

    The version is left out too: a binary file always declares at least
    1.5.1, which is the revision the format arrived in.
    """
    from_text = _ext.write_text(to_core(build_reference()), True)
    from_binary = _ext.write_text(
        _ext.read(_ext.write_binary(to_core(build_reference()))), True
    )

    assert sections(from_binary) == sections(from_text)
    for section in sections(from_text) - {"[SHAPES]", "[SIGNATURE]", "[VERSION]"}:
        assert rows(from_binary, section) == rows(from_text, section), section


def test_a_binary_file_is_never_older_than_the_format_that_writes_it():
    """A 1.5.0 file could not have been written in this form."""
    sequence = _ext.Sequence()
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)
    assert _ext.required_revision(sequence) == 0

    loaded = _ext.read(_ext.write_binary(sequence))

    assert rows(_ext.write_text(loaded, True), "[VERSION]") == [
        "major 1",
        "minor 5",
        "revision 1",
    ]


def test_a_shape_comes_back_within_a_single_precision_sample(
    reference_name, build_reference
):
    from_text = _ext.write_text(to_core(build_reference()), True)
    from_binary = _ext.write_text(
        _ext.read(_ext.write_binary(to_core(build_reference()))), True
    )

    for written, read_back in zip(
        rows(from_text, "[SHAPES]"), rows(from_binary, "[SHAPES]"), strict=False
    ):
        if written.startswith(("shape_id", "num_samples")):
            assert written == read_back
        else:
            np.testing.assert_allclose(
                float(read_back), float(written), rtol=1.2e-7, atol=0
            )


def test_reading_a_binary_file_leaves_the_sequence_split(build_reference):
    """Reading registers its events, so it forks as it parses."""
    built = to_core(build_reference())

    loaded = _ext.read(_ext.write_binary(to_core(build_reference())))

    assert loaded.num_block_definitions() == built.num_block_definitions()
    np.testing.assert_array_equal(
        loaded.instance_definitions(), built.instance_definitions()
    )


def test_a_definition_longer_than_a_byte_could_count_round_trips():
    """A slice position per slice is more values than 255, and must fit.

    The name's length and the value count are both written as int32, which is
    what makes that true; a byte would have capped a 3D acquisition at 255
    slices.
    """
    positions = list(np.linspace(-0.1, 0.1, 512))
    sequence = _ext.Sequence()
    sequence.set_definition("SlicePositions", positions)
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)

    loaded = _ext.read(_ext.write_binary(sequence))

    np.testing.assert_allclose(loaded.definitions()["SlicePositions"], positions)


def test_an_integer_definition_stays_an_integer():
    """The binary form tags a value's type, where the text form cannot.

    Held by writing the file twice: the second write can only produce the
    same bytes if the `i` tag survived the read, since a float would go back
    out as eight bytes rather than four.
    """
    sequence = _ext.Sequence()
    sequence.set_integer_definition("ReceiverGainHigh", [1])
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)
    written = _ext.write_binary(sequence)

    assert _ext.write_binary(_ext.read(written)) == written
    np.testing.assert_allclose(
        _ext.read(written).definitions()["ReceiverGainHigh"], [1]
    )


def test_a_text_definition_stays_text():
    sequence = _ext.Sequence()
    sequence.set_definition("Name", "gre")
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)

    loaded = _ext.read(_ext.write_binary(sequence))

    assert loaded.definitions()["Name"] == "gre"


def test_a_file_that_is_neither_form_says_so():
    with pytest.raises(RuntimeError):
        _ext.read(b"\x01pulseq\x02" + b"\x00" * 24)


def test_a_binary_file_written_the_other_way_round_is_named_rather_than_read():
    """A major version of 1 read backwards is an enormous number."""
    sequence = _ext.Sequence()
    sequence.add_block(0, 0, 0, 0, 0, 0, 1e-3)
    contents = bytearray(_ext.write_binary(sequence))
    contents[8:16] = bytes(reversed(contents[8:16]))

    with pytest.raises(RuntimeError, match="the other way round"):
        _ext.read(bytes(contents))


# --------------------------------------------------------------------------
# The event kinds 1.5.1 and 1.5.2 added. The reference toolbox this package is
# compared against implements none of them, so these are held by a round trip
# through both forms rather than against another writer's bytes.
# --------------------------------------------------------------------------


def test_an_extended_sequence_survives_the_text_form(extended_name, build_extended):
    written = _ext.write_text(build_extended(), True)

    assert _ext.write_text(_ext.read(written), True) == written


def test_an_extended_sequence_survives_the_binary_form(extended_name, build_extended):
    text = _ext.write_text(build_extended(), True)
    from_binary = _ext.write_text(_ext.read(_ext.write_binary(build_extended())), True)

    for section in sections(text) - {"[SHAPES]", "[SIGNATURE]", "[VERSION]"}:
        assert rows(from_binary, section) == rows(text, section), section


def test_the_extension_sections_are_actually_written(extended_name, build_extended):
    """The point of these sequences: each names a section the zoo never reaches."""
    expected = {
        "rotations": "extension ROTATIONS",
        "rf_shims": "extension RF_SHIMS",
        "custom_labels": "extension LABELSET",
        "every_extension": "extension ROTATIONS",
    }[extended_name]

    assert expected in _ext.write_text(build_extended(), True).decode()


def test_a_rotation_or_a_shim_makes_the_file_revision_one():
    """Both arrived in 1.5.1, so a file using either says 1.5.1."""
    import extended

    assert _ext.required_revision(extended.rotations()) == 1
    assert _ext.required_revision(extended.rf_shims()) == 1


def test_a_label_pulseq_does_not_define_makes_the_file_revision_two():
    import extended

    assert _ext.required_revision(extended.custom_labels()) == 2


def test_a_custom_label_comes_back_by_name(build_extended):
    """A label id means nothing outside the sequence that handed it out."""
    import extended

    loaded = _ext.read(_ext.write_text(extended.custom_labels(), True))

    assert extended.CUSTOM_LABEL in _ext.write_text(loaded, True).decode()
