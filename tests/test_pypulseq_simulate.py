"""RF simulation checked against flip, refocusing and bandwidth expectations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pypulseqpp as pp


def simulate(rf, *args, **kwargs):
    """MATLAB's six outputs, under the names MATLAB gives them."""
    mz_z, mz_xy, frequency, ref_eff, mx_xy, my_xy = pp.sim_rf(rf, *args, **kwargs)
    return SimpleNamespace(
        mz_z=mz_z,
        mz_xy=mz_xy,
        frequency=frequency,
        ref_eff=ref_eff,
        mx_xy=mx_xy,
        my_xy=my_xy,
    )


@pytest.fixture
def system():
    return pp.Opts(rf_raster_time=1e-6)


def _pulse(system, flip_deg, use, *, time_bw_product=4, duration=2e-3):
    return pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_deg),
        duration=duration,
        time_bw_product=time_bw_product,
        system=system,
        use=use,
        return_gz=False,
    )


def _on_resonance(response):
    return int(np.argmin(np.abs(response.frequency)))


def test_a_ninety_tips_fully_and_a_one_eighty_inverts(system):
    excitation = simulate(_pulse(system, 90, "excitation"))
    at = _on_resonance(excitation)
    assert excitation.mz_z[at] == pytest.approx(0.0, abs=1e-3)
    assert abs(excitation.mz_xy[at]) == pytest.approx(1.0, abs=1e-3)

    inversion = simulate(_pulse(system, 180, "inversion"))
    at = _on_resonance(inversion)
    assert inversion.mz_z[at] == pytest.approx(-1.0, abs=1e-3)
    assert abs(inversion.mz_xy[at]) == pytest.approx(0.0, abs=1e-3)


def test_refocusing_efficiency_separates_a_180_from_a_90(system):
    """``ref_eff`` is the number ``Mz`` cannot give you.

    A 90 and a 180 both move Mz, but only the 180 refocuses transverse
    magnetisation -- which is the whole question for a spin-echo train, and
    the reason mx_xy and my_xy are simulated at all.
    """
    refocusing = simulate(_pulse(system, 180, "refocusing"))
    excitation = simulate(_pulse(system, 90, "excitation"))

    assert abs(refocusing.ref_eff[_on_resonance(refocusing)]) == pytest.approx(
        1.0, abs=1e-3
    )
    assert abs(excitation.ref_eff[_on_resonance(excitation)]) == pytest.approx(
        0.5, abs=1e-3
    )


@pytest.mark.parametrize(
    ("time_bw_product", "duration"),
    [(4, 2e-3), (8, 4e-3), (4, 1e-3), (6, 3e-3)],
)
def test_the_simulated_profile_is_as_wide_as_the_pulse_was_designed_to_be(
    system, time_bw_product, duration
):
    """FWHM of the excitation profile against the nominal ``TBW / duration``.

    This is the check that the integration is right and not merely
    self-consistent: the answer has to land on a number chosen before the
    simulator ran.
    """
    response = simulate(
        _pulse(
            system, 90, "excitation", time_bw_product=time_bw_product, duration=duration
        )
    )
    magnitude = np.abs(response.mz_xy)
    above = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
    measured = response.frequency[above[-1]] - response.frequency[above[0]]

    assert measured == pytest.approx(time_bw_product / duration, rel=0.03)


def test_a_frequency_offset_moves_the_profile_and_not_its_width(system):
    offset = 1500.0
    centred = simulate(_pulse(system, 90, "excitation"))
    shifted_pulse = _pulse(system, 90, "excitation")
    shifted_pulse.freq_offset = offset
    shifted = simulate(shifted_pulse)

    def centre_and_width(response):
        magnitude = np.abs(response.mz_xy)
        above = np.flatnonzero(magnitude >= 0.5 * magnitude.max())
        edges = response.frequency[above[[0, -1]]]
        return 0.5 * (edges[0] + edges[1]), edges[1] - edges[0]

    centre_a, width_a = centre_and_width(centred)
    centre_b, width_b = centre_and_width(shifted)
    assert centre_b - centre_a == pytest.approx(offset, rel=0.05)
    assert width_b == pytest.approx(width_a, rel=0.05)


def test_the_answers_come_back_in_matlabs_order(system):
    rf = _pulse(system, 90, "excitation")

    answer = pp.sim_rf(rf)
    named = simulate(rf)

    assert len(answer) == 6
    for value, name in zip(
        answer,
        ("mz_z", "mz_xy", "frequency", "ref_eff", "mx_xy", "my_xy"),
        strict=True,
    ):
        np.testing.assert_array_equal(value, getattr(named, name))


def test_the_rephase_factor_defaults_by_the_pulses_use(system):
    """A refocusing pulse is not rephased; an excitation is.

    Left unrephased, a slice-selective excitation carries the linear phase
    ramp the pulse leaves across the profile, so the transverse
    magnetisation cancels when it is summed -- which is exactly what the
    default is there to avoid.
    """
    rf = _pulse(system, 90, "excitation")
    rephased = np.abs(simulate(rf).mz_xy.sum())
    unrephased = np.abs(simulate(rf, 0.0).mz_xy.sum())
    assert rephased > 3.0 * unrephased
