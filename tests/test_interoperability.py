"""Bidirectional binary interoperability with the optional reference toolbox."""

import math

import pytest

from pypulseqpp import _ext

mr = pytest.importorskip(
    "pypulseq_matlab_like", reason="the reference toolbox for the binary form"
)


@pytest.fixture
def their_sequence():
    """A small sequence built by the other toolbox."""
    seq = mr.Sequence()
    seq.add_block(mr.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"))
    seq.add_block(mr.make_trapezoid("x", area=1000))
    seq.add_block(
        mr.make_trapezoid("x", area=1000, duration=10e-3),
        mr.make_adc(num_samples=100, duration=10e-3),
    )
    seq.set_definition("Name", "interop")
    seq.set_definition("FOV", [0.256, 0.256, 0.005])
    return seq


def test_we_read_a_binary_file_they_wrote(their_sequence, tmp_path):
    """Their binary and their text have to reach the same sequence here."""
    their_sequence.write_binary(str(tmp_path / "theirs.bin"))
    their_sequence.write(str(tmp_path / "theirs.seq"))

    from_binary = _ext.read((tmp_path / "theirs.bin").read_bytes())
    from_text = _ext.read((tmp_path / "theirs.seq").read_bytes())

    assert _ext.write_text(from_binary, True) == _ext.write_text(from_text, True)


def test_they_read_a_binary_file_we_wrote(their_sequence, tmp_path):
    """The direction that matters: a scanner is the one reading."""
    their_sequence.write(str(tmp_path / "theirs.seq"))
    ours = _ext.read((tmp_path / "theirs.seq").read_bytes())
    (tmp_path / "ours.bin").write_bytes(_ext.write_binary(ours))

    theirs = mr.Sequence()
    theirs.read_binary(str(tmp_path / "ours.bin"))

    assert len(theirs.block_events) == ours.num_blocks()
    assert len(theirs.rf_library.data) == ours.num_rf()
    assert len(theirs.grad_library.data) == ours.num_gradients()
    assert len(theirs.adc_library.data) == ours.num_adc()
    assert len(theirs.shape_library.data) == ours.num_shapes()
    assert theirs.definitions["Name"] == "interop"


def test_they_read_every_reference_sequence_we_write(build_reference, tmp_path):
    """Nothing the zoo reaches puts a section in the file they cannot take."""
    import convert

    ours = convert.to_core(build_reference())
    (tmp_path / "ours.bin").write_bytes(_ext.write_binary(ours))

    theirs = mr.Sequence()
    theirs.read_binary(str(tmp_path / "ours.bin"))

    assert len(theirs.block_events) == ours.num_blocks()


def test_they_read_a_file_using_a_label_they_do_not_know(tmp_path):
    """The names go in `[DEFINITIONS]`, so no section they cannot parse.

    A reader that knows nothing of custom labels still reads the sequence;
    it just cannot say what the label is called, which is what it would have
    done before the name was carried at all.
    """
    import extended

    (tmp_path / "custom.bin").write_bytes(_ext.write_binary(extended.custom_labels()))

    theirs = mr.Sequence()
    theirs.read_binary(str(tmp_path / "custom.bin"))

    assert len(theirs.block_events) == extended.custom_labels().num_blocks()
    assert extended.CUSTOM_LABEL in str(theirs.definitions["CustomLabels"])
