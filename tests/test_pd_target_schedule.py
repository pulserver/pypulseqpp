"""Target-signal PD trains: inverse EPG of an explicit signal, bounded, with exact derivatives."""

import json
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp

pytest.importorskip("blochsim.trains")

from pypulseqpp._pd import (
    PD_CENTRE,
    PD_MAXIMUM_ANGLE_DEG,
    PD_PERIPHERY,
    PD_TARGET_PARAMETERS,
    _pd_target_trains,
    _pd_targets,
)

ESP, FIRST = 3.416e-3, 4.162e-3
FIGURE = json.loads(
    (Path(__file__).parent / "fixtures" / "buonincontri_figure1.json").read_text()
)["angles_deg"]


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.parametrize(
    ("name", "controls", "bound"),
    [("centre", PD_CENTRE, 1.5), ("periphery", PD_PERIPHERY, 3.0)],
)
def test_the_fitted_targets_reproduce_the_published_trains(name, controls, bound):
    published = FIGURE[name]
    angles, _ = _pd_target_trains(
        [len(published)], [controls], PD_MAXIMUM_ANGLE_DEG, ESP, FIRST
    )
    assert rms(np.rad2deg(angles[0]), published) < bound


def test_no_pulse_passes_the_maximum_angle():
    for length in (21, 40, 90, 160):
        for controls in (PD_CENTRE, PD_PERIPHERY):
            angles, _ = _pd_target_trains([length], [controls], 140.0, ESP, FIRST)
            assert np.rad2deg(angles).max() <= 140.0 + 1e-9


def test_echoes_meet_the_target_wherever_the_bound_is_not_reached():
    angles, signal = _pd_target_trains([40], [PD_CENTRE], 150.0, ESP, FIRST)
    target = _pd_targets([40], [PD_CENTRE])
    free = np.rad2deg(angles[0, 1:]) < 150.0 - 1e-9
    np.testing.assert_allclose(signal[0, 1:][free], target[0, 1:][free], atol=1e-12)


def test_a_batch_designs_each_train_as_alone_and_pads_with_zeros():
    lengths = [40, 21, 30]
    controls = [PD_CENTRE, PD_PERIPHERY, PD_CENTRE]
    batch, _ = _pd_target_trains(lengths, controls, 150.0, ESP, FIRST)
    for row, (length, control) in enumerate(zip(lengths, controls, strict=True)):
        alone, _ = _pd_target_trains([length], [control], 150.0, ESP, FIRST)
        np.testing.assert_allclose(batch[row, :length], alone[0], atol=1e-12)
        assert np.all(batch[row, length:] == 0)


@pytest.mark.parametrize("length", [30, 90])
def test_the_angle_jacobian_is_the_derivative_of_the_design(length):
    controls = np.array([PD_CENTRE, PD_PERIPHERY])
    lengths = [length, 24]
    _, _, jacobian = _pd_target_trains(
        lengths, controls, 150.0, ESP, FIRST, jacobian=True
    )
    step = 1e-6
    for parameter in range(8):
        moved = []
        for sign in (1, -1):
            changed = controls.copy()
            changed[:, parameter] += sign * step
            moved.append(_pd_target_trains(lengths, changed, 150.0, ESP, FIRST)[0])
        np.testing.assert_allclose(
            jacobian[parameter], (moved[0] - moved[1]) / (2 * step), atol=2e-5
        )


def test_the_public_schedule_is_the_designed_train():
    angles = pp.make_pd_target_schedule(40, ESP, first_echo_time=FIRST)
    expected, _ = _pd_target_trains([40], [PD_CENTRE], 150.0, ESP, FIRST)
    np.testing.assert_allclose(angles, expected[0], atol=1e-12)
    assert np.isclose(np.rad2deg(angles[0]), PD_CENTRE[0])


@pytest.mark.parametrize(
    "change",
    [
        {"length": 1},
        {"length": 2.5},
        {"first_flip_angle": np.deg2rad(160.0)},
        {"maximum_flip_angle": np.pi},
        {"plateau_signal": 0.0},
        {"plateau_fraction": 1.0},
        {"tail_signal": -0.1},
        {"tail_exponent": 0.0},
        {"initial_signal": 1.5},
        {"initial_decay": 0.0},
        {"reference_t2": 0.0},
    ],
)
def test_a_parameter_out_of_range_is_refused(change):
    arguments = {"length": 40, "echo_spacing": ESP} | change
    with pytest.raises(ValueError):
        pp.make_pd_target_schedule(**arguments)


def value(seq, name):
    return float(np.atleast_1d(seq.definitions[name])[0])


def build(**protocol):
    from pypulseqpp import sequences

    return sequences.fse3D_sequence(
        pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        n_x=32,
        n_y=16,
        n_z=8,
        etl=20,
        te=None,
        tr=0.5,
        n_dummy=1,
        flip_modulation="pd_target",
        **protocol,
    )


def played_flips(seq, startup, etl_max):
    """Every train's refocusing angles (deg), dummy excluded, by RF amplitude."""
    from test_zoo_fse import pulses

    refocusing = [a for _, use, a in pulses(seq) if use == "refocusing"]
    actual = np.reshape(refocusing, (-1, etl_max + startup))[1:]
    first = np.atleast_1d(seq.definitions["RefocusingFlipAngles"])[0]
    return actual * float(first) / actual[0, 0]


@pytest.mark.parametrize("ordering", ["buonincontri", "buonincontri_golden"])
def test_each_shot_of_a_transition_plays_the_train_of_its_own_target(ordering):
    from pypulseqpp import sequences

    seq = build(
        ordering=ordering,
        tr_periphery=0.4,
        etl_periphery=28,
        **{
            f"pd_target_{name.removesuffix('_deg')}_periphery"
            + ("_deg" if name.endswith("_deg") else ""): value
            for name, value in zip(PD_TARGET_PARAMETERS, PD_PERIPHERY, strict=True)
        },
    )
    assert seq.check_timing()[0]
    from test_zoo_fse import trains

    views = trains(seq)[1:]
    lengths, _, place = sequences.fse3D_sequence.shot_parameters(
        sum(len(train) for train in views), 20, 28, 0.5, 0.4
    )
    fraction = sequences.fse3D_sequence.cubic(place)[:, None]
    controls = np.asarray(PD_CENTRE) + fraction * (
        np.asarray(PD_PERIPHERY) - np.asarray(PD_CENTRE)
    )
    expected, _ = _pd_target_trains(
        np.asarray(lengths) + 6,
        controls,
        PD_MAXIMUM_ANGLE_DEG,
        value(seq, "EchoSpacing"),
        value(seq, "FirstEchoSpacing"),
    )
    actual = played_flips(seq, 6, 28)
    np.testing.assert_allclose(actual, np.rad2deg(expected), atol=1e-6)


def test_a_target_pd_scan_states_its_target_parameters():
    seq = build()
    assert seq.check_timing()[0]
    assert seq.definitions["PDControlColumns"] == " ".join(
        [
            "first_angle_deg",
            "plateau_signal",
            "plateau_slope",
            "plateau_fraction",
            "tail_signal",
            "tail_exponent",
            "initial_signal",
            "initial_decay",
        ]
    )
    np.testing.assert_allclose(
        np.atleast_1d(seq.definitions["PDEndpointControls"]),
        np.tile(PD_CENTRE, 2),
    )


def test_a_target_pd_scan_refuses_the_angle_optimizer():
    with pytest.raises(ValueError, match="pd_optimize"):
        build(pd_optimize=True)
