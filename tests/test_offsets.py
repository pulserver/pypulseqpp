"""Frequency and phase offsets with their ppm offsets resolved at a field."""

from types import SimpleNamespace

import numpy as np
import pytest

import pypulseqpp as pp

GAMMA = 42576000.0


def test_a_ppm_offset_is_resolved_at_the_larmor_frequency_of_the_system():
    """ppm times MHz is Hz, and rad/MHz times MHz is rad: one factor serves both."""
    pulse = pp.make_block_pulse(
        1.0,
        duration=4e-3,
        freq_offset=100.0,
        phase_offset=0.25,
        freq_ppm=-3.45,
        phase_ppm=0.5,
    )
    larmor_mhz = 1e-6 * GAMMA * 3.0

    frequency, phase = pp.calc_absolute_offsets(pulse, system=pp.Opts(B0=3.0))

    assert frequency == pytest.approx(100.0 - 3.45 * larmor_mhz, rel=1e-15)
    assert phase == pytest.approx(0.25 + 0.5 * larmor_mhz, rel=1e-15)


def test_an_adc_resolves_its_ppm_offsets_as_a_pulse_does():
    window = pp.make_adc(
        num_samples=8, duration=1e-3, freq_offset=-20.0, freq_ppm=1.0, phase_ppm=-0.1
    )
    larmor_mhz = 1e-6 * GAMMA * 7.0

    assert pp.calc_absolute_offsets(window, system=pp.Opts(B0=7.0)) == pytest.approx(
        (-20.0 + larmor_mhz, -0.1 * larmor_mhz), rel=1e-15
    )


def test_an_event_without_ppm_offsets_keeps_the_offsets_it_carries():
    upstream = SimpleNamespace(freq_offset=150.0, phase_offset=-0.5)

    assert pp.calc_absolute_offsets(upstream, system=pp.Opts(B0=3.0)) == (150.0, -0.5)


def test_without_a_system_the_default_system_resolves_the_ppm_offsets():
    pulse = pp.make_block_pulse(1.0, duration=4e-3, freq_ppm=2.0)
    default = pp.Opts.default

    assert pp.calc_absolute_offsets(pulse) == pp.calc_absolute_offsets(
        pulse, system=default
    )
    assert pp.calc_absolute_offsets(pulse)[0] == pytest.approx(
        2.0 * 1e-6 * default.gamma * default.B0
    )


def test_every_library_row_is_resolved_as_its_event_is():
    seq = pp.Sequence(pp.Opts())
    for ppm in (-3.45, 0.0, 1.2):
        seq.add_block(
            pp.make_block_pulse(
                1.0, duration=2e-3, freq_offset=40.0, freq_ppm=ppm, phase_ppm=ppm / 7
            )
        )
        seq.add_block(
            pp.make_adc(num_samples=16, duration=2e-3, phase_offset=0.3, freq_ppm=-ppm)
        )
    system = pp.Opts(B0=2.89)

    rf, adc = seq.libraries().absolute_offsets(system)

    blocks = [seq.get_block(index) for index in seq.block_events]
    np.testing.assert_array_equal(
        rf,
        [pp.calc_absolute_offsets(block.rf, system=system) for block in blocks[::2]],
    )
    np.testing.assert_array_equal(
        adc,
        [pp.calc_absolute_offsets(block.adc, system=system) for block in blocks[1::2]],
    )


def test_a_1_4_1_file_resolves_ppm_offsets_at_the_field_of_the_sequence(tmp_path):
    """As the reference writer does: at its system's gamma and B0, not at 1.5 T."""
    system = pp.Opts(B0=3.0)
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(1.0, duration=4e-3, freq_ppm=-3.45))
    by_default, at_three, at_one_and_a_half = (
        tmp_path / name for name in ("default.seq", "3T.seq", "1.5T.seq")
    )

    seq.write_v141(by_default)
    seq.write_v141(at_three, gamma=system.gamma, field=3.0)
    seq.write_v141(at_one_and_a_half, gamma=system.gamma, field=1.5)

    assert by_default.read_bytes() == at_three.read_bytes()
    assert by_default.read_bytes() != at_one_and_a_half.read_bytes()
    rf_row = by_default.read_text().split("[RF]\n")[1].splitlines()[0].split()
    # The text writes the offset to a thousandth of a hertz.
    assert float(rf_row[6]) == pytest.approx(
        pp.calc_absolute_offsets(seq.get_block(1).rf, system=system)[0], abs=5e-4
    )
