"""Independent spin evolution and Figure 1 parity of the Busse constructor."""

import json
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp

# The schedules and their effective TE are computed by blochsim.trains.
pytest.importorskip("blochsim.trains")


@pytest.fixture(scope="module")
def reference():
    return json.loads(
        (Path(__file__).parent / "fixtures/busse_figure1.json").read_text()
    )


def test_complete_train_matches_working_figure1(reference):
    angles = pp.make_busse_schedule(106)
    np.testing.assert_allclose(
        np.rad2deg(angles), reference["angles_deg"], atol=1e-8, rtol=0
    )


def test_independent_isochromat_bloch_matches_reference_coherence(reference):
    angles = pp.make_busse_schedule(106)
    # Uniform phase quadrature is independent of the EPG state recursion.
    # 2048 samples resolve every coherence order in this finite train.
    phase = np.exp(2j * np.pi * np.arange(2048) / 2048)
    transverse = np.ones(2048, dtype=complex)
    longitudinal = np.zeros(2048)
    signal = []
    for angle in angles:
        transverse *= phase
        sine, cosine = np.sin(angle), np.cos(angle)
        imaginary = transverse.imag.copy()
        transverse = transverse.real + 1j * (cosine * imaginary - sine * longitudinal)
        longitudinal = sine * imaginary + cosine * longitudinal
        transverse *= phase
        signal.append(transverse.mean())
    np.testing.assert_allclose(signal, reference["static_signal"], atol=2e-12, rtol=0)


@pytest.mark.parametrize("acquired", [40, 80, 100, 120, 160])
def test_same_prescription_retains_actual_controls_and_relative_centre(acquired):
    angles = pp.make_busse_schedule(acquired + 6)
    centre = 6 + (acquired - 1) // 2
    assert len(angles) == acquired + 6
    np.testing.assert_allclose(
        np.rad2deg(angles[[np.argmin(angles), centre, -1]]), [25, 70, 120], atol=1e-6
    )
    np.testing.assert_allclose(angles[0], np.deg2rad(120), atol=1e-12)
    assert max(angles) <= np.deg2rad(120) + 1e-8
    assert angles[-1] > angles[centre] > min(angles)


def test_explicit_midpoint_is_identical_to_default():
    np.testing.assert_array_equal(
        pp.make_busse_schedule(106), pp.make_busse_schedule(106, centre_echo=55)
    )


def test_cached_schedules_are_independent_arrays():
    first = pp.make_busse_schedule(106)
    first[:] = 0
    assert pp.make_busse_schedule(106).min() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0},
        {"length": 106.5},
        {"length": True},
        {"startup_echoes": -1},
        {"startup_echoes": 106},
        {"centre_echo": 5},
        {"centre_echo": 105},
        {"centre_echo": 55.5},
        {"minimum_flip_angle": 0},
        {"minimum_flip_angle": np.nan},
        {"centre_flip_angle": np.deg2rad(20)},
        {"maximum_flip_angle": np.pi},
        {"initial_decay": 0},
        {"initial_decay": np.inf},
    ],
)
def test_invalid_prescriptions_raise(kwargs):
    arguments = {"length": 106}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        pp.make_busse_schedule(**arguments)


def test_batched_variable_trains_match_independent_scalar_shooting():
    from pypulseqpp._busse import _make_busse_schedules

    lengths = np.array([46, 51, 66, 86, 106])
    controls = np.array(
        [[25, 70, 120], [27, 68, 118], [30, 65, 115], [22, 74, 125], [20, 75, 130]]
    )
    profiles = _make_busse_schedules(lengths, controls, 25, 1.3)
    for row, (length, requested) in enumerate(zip(lengths, controls, strict=True)):
        reference = pp.make_busse_schedule(
            int(length),
            minimum_flip_angle=np.deg2rad(requested[0]),
            centre_flip_angle=np.deg2rad(requested[1]),
            maximum_flip_angle=np.deg2rad(requested[2]),
            centre_echo=25,
        )
        np.testing.assert_allclose(profiles[row, :length], reference, atol=2e-8, rtol=0)
        assert np.all(profiles[row, length:] == 0)


@pytest.fixture(scope="module")
def played():
    return json.loads(
        (Path(__file__).parent / "fixtures/fse3d_busse_trains.json").read_text()
    )["trains"]


def static_signal(angles):
    """Echoes of a train without relaxation, by isochromat summation."""
    phase = np.exp(2j * np.pi * np.arange(2048) / 2048)
    transverse = np.ones(2048, dtype=complex)
    longitudinal = np.zeros(2048)
    signal = []
    for angle in angles:
        transverse *= phase
        sine, cosine = np.sin(angle), np.cos(angle)
        imaginary = transverse.imag.copy()
        transverse = transverse.real + 1j * (cosine * imaginary - sine * longitudinal)
        longitudinal = sine * imaginary + cosine * longitudinal
        transverse *= phase
        signal.append(transverse.mean())
    return np.array(signal)


@pytest.mark.parametrize("name", ["te60", "te44"])
def test_played_trains_are_reproduced_in_angle_and_signal(played, name):
    train = played[name]
    startup = train["startup_echoes"]
    low, centre, high = np.deg2rad(train["played_controls_deg"])
    angles = pp.make_busse_schedule(
        train["echo_train_length"] + startup,
        minimum_flip_angle=low,
        centre_flip_angle=centre,
        maximum_flip_angle=high,
        centre_echo=startup + train["centre_echo"],
        startup_echoes=startup,
        initial_decay=train["initial_decay"],
    )
    reference = np.deg2rad(train["angles_deg"])
    np.testing.assert_allclose(
        np.rad2deg(angles), train["angles_deg"], atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        static_signal(angles), static_signal(reference), atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("name", ["te60", "te44"])
def test_played_centre_echo_and_its_effective_te_are_reproduced(played, name):
    from pypulseqpp._fse_contrast import _select_busse_centre

    train = played[name]
    startup, etl = train["startup_echoes"], train["echo_train_length"]
    timing = (train["echo_spacing"], train["first_echo_time"])
    controls = np.array(train["prescribed_controls_deg"])
    centre = _select_busse_centre(
        etl,
        etl,
        startup,
        controls,
        controls,
        train["initial_decay"],
        *timing,
        train["requested_effective_te"],
    )
    assert centre == train["centre_echo"]
    effective = pp.calc_fse_effective_te(
        np.deg2rad(train["angles_deg"]), timing[0], first_echo_time=timing[1]
    )
    assert effective[startup + centre] == pytest.approx(
        train["centre_effective_te"], abs=1e-9
    )


@pytest.mark.parametrize(
    "design",
    [
        lambda: pp.make_busse_schedule(47),
        lambda: pp.make_pd_schedule(27, 0.003416),
        lambda: pp.calc_fse_effective_te([np.pi] * 3, 0.004),
    ],
)
def test_a_design_without_blochsim_names_the_extra(monkeypatch, design):
    import sys

    from pypulseqpp._busse import _schedule
    from pypulseqpp._pd import _pd_schedule

    # A schedule designed earlier would be answered from the cache.
    _schedule.cache_clear()
    _pd_schedule.cache_clear()
    # A None entry makes ``import blochsim`` fail as if it were not installed.
    monkeypatch.setitem(sys.modules, "blochsim", None)
    with pytest.raises(ImportError, match=r"pypulseqpp\[design\]"):
        design()
