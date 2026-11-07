"""Independent spin evolution and Figure 1 parity of the Busse constructor."""

import json
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp


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
