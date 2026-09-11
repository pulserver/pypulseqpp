"""Balanced SSFP 3D: phase cycling, its half-flip preparation and zero moments per TR."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, sequences

SMALL = {"n_x": 64, "n_y": 16, "n_z": 4, "n_acs": 0, "n_acs_z": 0, "n_dummy": 2}


def app(**kwargs):
    return sequences.bssfp3D_sequence.Bssfp3DApp(pp.Opts(), **{**SMALL, **kwargs})


def pulses(seq):
    """Per RF pulse, in order: its phase, its peak amplitude, and its ADC's phase."""
    rows = []
    for index in range(1, len(seq.block_events) + 1):
        block = seq.get_block(index)
        if block.rf is not None:
            peak = np.abs(np.asarray(block.rf.signal)).max()
            rows.append([block.rf.phase_offset, peak, None])
        elif block.adc is not None:
            rows[-1][2] = block.adc.phase_offset
    return rows


def test_the_phase_alternates_after_an_opposite_half_flip():
    rows = pulses(app().design())
    phases = [float(np.mod(phase, 2 * np.pi)) for phase, _, _ in rows]

    assert phases[0] == pytest.approx(0.0)
    assert phases[1:] == pytest.approx(
        [np.pi * ((k + 1) % 2) for k in range(len(rows) - 1)]
    )
    assert rows[0][1] == pytest.approx(rows[1][1] / 2, rel=1e-3)
    assert all(adc == pytest.approx(phase) for phase, _, adc in rows if adc is not None)


def test_the_half_flip_leads_the_first_pulse_by_half_a_repetition():
    sequence = app()
    centres = np.asarray(sequence.design().rf_times()[0])

    assert centres[1] - centres[0] == pytest.approx(
        sequence.repetition_time / 2, abs=pp.Opts().block_duration_raster
    )
    assert np.diff(centres[1:]) == pytest.approx(sequence.repetition_time)


def test_every_repetition_returns_its_gradient_moments_to_zero():
    waves, excitations = app(acceleration=2).design().waveforms_and_times()[:2]
    edges = np.asarray(excitations)[0]
    for channel in waves:
        t, g = np.asarray(channel)
        if t.size == 0:
            continue
        grid = np.union1d(t, edges)
        amplitude = np.interp(grid, t, g, left=0.0, right=0.0)
        area = np.concatenate(
            [[0.0], np.cumsum(np.diff(grid) * (amplitude[1:] + amplitude[:-1]) / 2)]
        )
        assert np.diff(np.interp(edges, grid, area)) == pytest.approx(0.0, abs=1e-3)


def test_each_acquisition_carries_the_pair_it_encodes():
    sequence = app(acceleration=2, n_acs=4, n_acs_z=2)
    found = sequence.design().evaluate_labels(evolution="adc")
    lin, par, once = (
        np.atleast_1d(found.get(name, 0)) for name in ("LIN", "PAR", "ONCE")
    )

    assert list(zip(lin, par, strict=True)) == [tuple(view) for view in sequence.pairs]
    assert set(once) == {0}


def test_a_repetition_shorter_than_the_readout_is_refused():
    with pytest.raises(ValueError):
        app(tr=1e-4)


def test_the_command_line_describes_the_flip(capsys):
    with pytest.raises(SystemExit):
        cli.run(sequences.bssfp3D_sequence.main, ["--help"])

    assert "Flip angle of every" in capsys.readouterr().out
