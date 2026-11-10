"""PD trains are relaxation-aware inverse EPG of a parametric target signal."""

import json
from pathlib import Path

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp._pd import (
    PD_CENTRE,
    PD_MAXIMUM_ANGLE_DEG,
    PD_PERIPHERY,
    _pd_targets,
    _pd_trains,
)

ESP, FIRST = 0.003416, 0.004162
FIGURE = json.loads(
    (Path(__file__).parent / "fixtures" / "buonincontri_figure1.json").read_text()
)["angles_deg"]


def designed(lengths, controls, maximum=PD_MAXIMUM_ANGLE_DEG):
    return _pd_trains(lengths, controls, maximum, ESP, FIRST)


@pytest.mark.parametrize(
    ("name", "bound"),
    [
        ("centre", 1.5),
        ("inter_3", 4.0),
        ("inter_2", 4.0),
        ("inter_1", 4.5),
        ("periphery", 3.0),
    ],
)
def test_the_default_trains_come_within_reading_error_of_the_published_figure(
    name, bound
):
    """Each published train against the target signal at its place in the transition."""
    published = np.asarray(FIGURE[name])
    first, last = len(FIGURE["centre"]), len(FIGURE["periphery"])
    place = (first - len(published)) / (first - last)
    controls = np.add(PD_CENTRE, place * np.subtract(PD_PERIPHERY, PD_CENTRE))
    angles = np.rad2deg(designed([len(published)], [controls])[0][0])
    assert np.sqrt(np.mean((angles - published) ** 2)) < bound


def test_independent_bloch_evolution_reaches_the_target_signal():
    angles, _ = designed([40], [PD_CENTRE])
    angles = angles[0]
    phase = np.exp(2j * np.pi * np.arange(2048) / 2048)
    transverse = np.ones(2048, complex)
    longitudinal = np.zeros(2048)
    signal = []
    for echo, angle in enumerate(angles):
        half = (FIRST if echo == 0 else ESP) / 2
        e1, e2 = np.exp(-half), np.exp(-half / 0.1)
        transverse *= phase * e2
        longitudinal = longitudinal * e1 + 1 - e1
        imag = transverse.imag.copy()
        transverse = transverse.real + 1j * (
            np.cos(angle) * imag - np.sin(angle) * longitudinal
        )
        longitudinal = np.sin(angle) * imag + np.cos(angle) * longitudinal
        transverse *= phase * e2
        longitudinal = longitudinal * e1 + 1 - e1
        signal.append(transverse.mean())
    target = _pd_targets([40], [PD_CENTRE])[0]
    np.testing.assert_allclose(np.real(signal)[1:], target[1:], atol=1e-10)


def test_a_pulse_the_target_would_push_past_the_maximum_plays_the_maximum():
    (angles,), (signal,) = designed([40], [PD_CENTRE], maximum=110.0)
    target = _pd_targets([40], [PD_CENTRE])[0]
    capped = np.isclose(np.rad2deg(angles), 110.0)
    assert np.rad2deg(angles).max() <= 110.0 + 1e-9
    assert capped.any()
    assert np.all(signal[capped] < target[capped])


def test_a_batch_designs_each_train_as_alone_and_pads_with_zero():
    lengths = [51, 40, 26]
    controls = [PD_CENTRE, np.add(PD_CENTRE, PD_PERIPHERY) / 2, PD_PERIPHERY]
    batch, _ = designed(lengths, controls)
    for row, length, control in zip(batch, lengths, controls, strict=True):
        alone = pp.make_pd_schedule(
            length,
            ESP,
            first_echo_time=FIRST,
            first_flip_angle=np.deg2rad(control[0]),
            plateau_signal=control[1],
            plateau_slope=control[2],
            plateau_fraction=control[3],
            tail_signal=control[4],
            tail_exponent=control[5],
            initial_signal=control[6],
            initial_decay=control[7],
        )
        np.testing.assert_allclose(row[:length], alone, atol=1e-12)
        assert np.all(row[length:] == 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"length": 1},
        {"first_flip_angle": np.deg2rad(160.0)},
        {"maximum_flip_angle": np.pi},
        {"plateau_signal": 0.0},
        {"plateau_fraction": 1.0},
        {"tail_exponent": 0.0},
        {"initial_decay": 0.0},
        {"echo_spacing": 0.0},
    ],
)
def test_invalid_pd_parameters_raise(changes):
    with pytest.raises(ValueError):
        pp.make_pd_schedule(**({"length": 26, "echo_spacing": ESP} | changes))


def test_pd_fse_keeps_common_rf_slots_and_acquired_echo_times():
    from test_zoo_fse import blocks, echo_times, trains

    settings = {
        "system": pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 20,
        "etl_periphery": 28,
        "tr": 0.8,
        "tr_periphery": 0.4,
        "te": None,
        "n_dummy": 0,
        "flip_modulation": "pd",
        "pd_effective_te": 0.028,
        "pd_plateau_signal_periphery": PD_PERIPHERY[1],
    }
    seq = sequences.fse3D_sequence(**settings)
    views = trains(seq)
    lengths, _, _ = sequences.fse3D_sequence.shot_parameters(
        sum(map(len, views)), 20, 28, 0.8, 0.4
    )
    flips = []
    for block in blocks(seq):
        if block.rf is not None:
            if block.rf.use == "excitation":
                flips.append([])
            else:
                flips[-1].append(
                    abs(np.trapezoid(block.rf.signal, block.rf.t) * 2 * np.pi)
                )
    assert all(len(row) == 34 for row in flips)
    for row, length in zip(flips, lengths, strict=True):
        assert np.all(np.asarray(row[length + 6 :]) == 0)
    physical = float(seq.definitions["TE"][0])
    effective = float(seq.definitions["EffectiveEchoTime"][0])
    assert physical > effective
    all_times = echo_times(seq)
    offset = 0
    for train in views:
        times = all_times[offset : offset + len(train)]
        echoes = np.array([echo for _, _, echo in train])
        np.testing.assert_allclose(
            times,
            physical
            + (echoes - (6 + int(seq.definitions["PDCentreEcho"][0])))
            * float(seq.definitions["EchoSpacing"][0]),
            atol=1e-10,
        )
        offset += len(train)
    assert seq.definitions["ResolvedViewOrdering"] == "buonincontri"
    assert seq.check_timing()[0]


def test_pd_fit_moves_only_the_tail_onset_and_keeps_energy_and_effective_te():
    pytest.importorskip("blochsim.simulators")
    settings = {
        "system": pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 20,
        "tr": 0.8,
        "te": None,
        "n_dummy": 0,
        "flip_modulation": "pd",
        "pd_effective_te": 0.028,
    }
    baseline = sequences.fse3D_sequence(**settings)
    fitted = sequences.fse3D_sequence(**settings, pd_optimize=True)
    a = np.asarray(fitted.definitions["RefocusingFlipAngles"])
    b = np.asarray(baseline.definitions["RefocusingFlipAngles"])
    before = np.reshape(baseline.definitions["PDEndpointControls"], (2, 8))[0]
    after = np.reshape(fitted.definitions["PDEndpointControls"], (2, 8))[0]
    moved = np.flatnonzero(before != after)
    assert set(moved) <= {3}
    assert a[0] == pytest.approx(PD_CENTRE[0], abs=1e-9)
    assert np.sum(a * a) <= np.sum(b * b) * 1.00011
    assert float(fitted.definitions["EffectiveEchoTime"][0]) == pytest.approx(
        float(baseline.definitions["EffectiveEchoTime"][0]), abs=1.01e-4
    )


def test_a_dense_transition_changes_little_between_neighbouring_shots():
    place = np.linspace(0, 1, 586)
    fraction = place**2 * (3 - 2 * place)
    lengths = np.rint(51 - 25 * fraction).astype(int)
    controls = np.add(
        PD_CENTRE, fraction[:, None] * np.subtract(PD_PERIPHERY, PD_CENTRE)
    )
    profiles, _ = designed(lengths, controls)
    active = np.arange(51)[None, :] < lengths[:, None]
    assert np.all(profiles[~active] == 0)
    same_length = lengths[1:] == lengths[:-1]
    assert np.rad2deg(abs(np.diff(profiles, axis=0))[same_length]).max() < 2.0
