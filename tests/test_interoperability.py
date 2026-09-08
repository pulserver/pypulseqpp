"""The binary form, against the toolbox that defines it.

`pypulseq-matlab-like` is a transcription of MATLAB Pulseq, and it is where
the binary layout in this package comes from. These tests hold the two
readers and writers against each other in both directions.

It is not on PyPI, so it is not a dependency and these skip without it:

    pip install git+https://github.com/m-a-x-i-m-z/pypulseq-matlab-like
"""

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
