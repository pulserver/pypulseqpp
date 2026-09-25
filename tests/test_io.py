"""Reading a file builds the system it was written under; writing is the inverse.

A ``.seq`` file records its rasters and nothing about the limits the sequence
was designed against, so a file read onto the shared default system is checked
against limits that have nothing to do with it. ``pypulseqpp.io.read`` builds
the system from the file instead, and these tests pin what it puts in it.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

#: A prescription small enough to write and read in a test.
SMALL = {"n_x": 32, "n_y": 8, "n_slices": 1, "tr": None, "n_dummy": 0}


@pytest.fixture
def gre():
    return sequences.gre2D_sequence(**SMALL)


@pytest.fixture
def strong():
    """A sequence whose gradients go beyond the default limits."""
    system = pp.Opts(
        max_grad=78.0,
        grad_unit="mT/m",
        max_slew=190.0,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid("x", area=8000, duration=3e-3, system=system))
    seq.add_block(
        pp.make_block_pulse(
            np.pi / 2, duration=1e-3, delay=100e-6, system=system, use="excitation"
        )
    )
    return seq


@pytest.mark.parametrize("binary", [False, True])
def test_a_sequence_written_either_way_reads_back_with_the_same_blocks(
    tmp_path, gre, binary
):
    path = tmp_path / ("s.bin" if binary else "s.seq")
    pp.io.write(gre, path, binary=binary)

    assert pp.io.read(path).num_blocks == gre.num_blocks


@pytest.mark.parametrize("binary", [False, True])
def test_a_sequence_read_from_its_bytes_is_the_sequence_read_from_its_file(
    tmp_path, gre, binary
):
    path = tmp_path / ("s.bin" if binary else "s.seq")
    pp.io.write(gre, path, binary=binary)

    from_file = pp.io.read(path)
    from_bytes = pp.io.read(io.BytesIO(path.read_bytes()), verify=True)

    pp.io.write(from_file, tmp_path / "file.seq")
    pp.io.write(from_bytes, tmp_path / "bytes.seq")
    assert (tmp_path / "bytes.seq").read_bytes() == (tmp_path / "file.seq").read_bytes()
    assert from_bytes.system.max_grad == from_file.system.max_grad


def test_bytes_edited_after_signing_are_refused_when_verified(tmp_path, gre):
    path = tmp_path / "s.seq"
    pp.io.write(gre, path)
    signed = path.read_bytes()
    edited = signed.replace(b"[BLOCKS]\n", b"[BLOCKS]\n ", 1)
    assert edited != signed

    with pytest.raises(RuntimeError, match="signature"):
        pp.io.read(io.BytesIO(edited), verify=True)


def test_a_file_opened_as_text_is_refused_rather_than_decoded(tmp_path, gre):
    path = tmp_path / "s.seq"
    pp.io.write(gre, path)

    with path.open() as text, pytest.raises(TypeError, match="binary mode"):
        pp.Sequence().read(text)


def test_the_binary_form_is_smaller_and_returns_no_signature(tmp_path, gre):
    text, binary = tmp_path / "s.seq", tmp_path / "s.bin"

    signature = pp.io.write(gre, text)

    assert pp.io.write(gre, binary, binary=True) is None
    assert len(signature) == 32
    assert binary.stat().st_size < text.stat().st_size
    assert "[SIGNATURE]" in text.read_text()


def test_a_written_sequence_reads_back_as_itself(tmp_path, gre):
    path, again = tmp_path / "gre.seq", tmp_path / "again.seq"
    pp.io.write(gre, path)

    pp.io.write(pp.io.read(path), again)

    assert again.read_text() == path.read_text()


@pytest.mark.parametrize("binary", [False, True])
def test_the_rasters_of_the_system_built_are_the_ones_the_file_carries(
    tmp_path, binary
):
    system = pp.Opts(
        rf_raster_time=1e-6, grad_raster_time=1e-5, adc_raster_time=1e-7, B0=3.0
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=system))
    path = tmp_path / ("s.bin" if binary else "s.seq")
    pp.io.write(seq, path, binary=binary)

    built = pp.io.read(path).system

    assert built.rf_raster_time == system.rf_raster_time
    assert built.grad_raster_time == system.grad_raster_time
    assert built.adc_raster_time == system.adc_raster_time


def test_the_limits_built_admit_the_waveforms_the_file_holds(tmp_path, strong):
    path = tmp_path / "strong.seq"
    pp.io.write(strong, path)

    read_back = pp.io.read(path)

    assert pp.safety.check_max_grad(read_back)[0]
    assert pp.safety.check_max_slew(read_back)[0]
    # The default limits are what a bare read would have measured it against.
    onto_default = pp.Sequence()
    onto_default.read(str(path))
    assert not pp.safety.check_max_grad(onto_default, pp.Opts())[0]


def test_a_gentle_sequence_keeps_the_default_limits_rather_than_shrinking_to_fit(
    tmp_path, gre
):
    path = tmp_path / "gre.seq"
    pp.io.write(gre, path)

    built = pp.io.read(path).system

    assert built.max_grad == pp.Opts().max_grad
    assert built.max_slew == pp.Opts().max_slew


def test_a_limit_passed_in_wins_over_the_one_the_file_would_give(tmp_path, strong):
    path = tmp_path / "strong.seq"
    pp.io.write(strong, path)

    built = pp.io.read(path, max_grad=1e6, max_slew=1e10).system

    assert (built.max_grad, built.max_slew) == (1e6, 1e10)


def test_a_limit_the_file_states_is_the_one_the_system_carries(tmp_path, gre):
    path = tmp_path / "stated.seq"
    gre.set_definition(key="MaxGrad", value=2.5e6)
    gre.set_definition(key="MaxSlew", value=1.5e10)
    gre.set_definition(key="B0", value=7.0)
    pp.io.write(gre, path)

    built = pp.io.read(path).system

    assert (built.max_grad, built.max_slew, built.B0) == (2.5e6, 1.5e10, 7.0)
