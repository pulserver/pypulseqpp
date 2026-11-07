"""Prescribed PD signal envelopes, independent Bloch evolution and padded FSE trains."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences


@pytest.mark.parametrize("acquired", [20, 40, 45, 80, 120, 160])
def test_pd_first_peak_final_are_independent_actual_angles(acquired):
    angles = np.rad2deg(
        pp.make_pd_schedule(acquired + 6, 0.003416, first_echo_time=0.004162)
    )
    np.testing.assert_allclose(angles[[0, -1]], [120, 100], atol=1e-4)
    assert angles.max() == pytest.approx(140, abs=1e-4)
    assert 6 <= np.argmax(angles) < len(angles) - 1
    assert np.all(angles > 0)


def test_equal_peak_and_final_reaches_a_late_rf_plateau():
    angles = pp.make_pd_schedule(
        51, 0.003416, first_echo_time=0.004162, final_flip_angle=np.deg2rad(140)
    )
    assert angles[-1] == pytest.approx(angles.max(), abs=1e-6)
    assert np.rad2deg(angles.min()) < 40


def test_independent_bloch_evolution_tracks_relaxation_aware_target():
    from pypulseqpp._pd import _pd_schedule

    angles, levels = _pd_schedule(
        26, 120.0, 140.0, 100.0, 0.7, 6, 0.003416, 0.004162, 1.0, 0.1, 1.3
    )
    phase = np.exp(2j * np.pi * np.arange(2048) / 2048)
    transverse = np.ones(2048, complex)
    longitudinal = np.zeros(2048)
    signal = []
    for echo, angle in enumerate(angles):
        half = (0.004162 if echo == 0 else 0.003416) / 2
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
    x = np.arange(26)
    cut = 6 + 0.7 * (26 - 1 - 6)
    tail = np.clip((x - cut) / (25 - cut), 0, 1)
    tail = tail**2
    initial = np.sin(np.deg2rad(120) / 2) ** 2 * np.exp(-0.004162 / 0.1)
    target = levels[0] * (1 + (levels[1] - 1) * tail) + (initial - levels[0]) * np.exp(
        -x / 1.3
    )
    np.testing.assert_allclose(np.real(signal), target, atol=2e-11)
    assert max(target[6:18]) - min(target[6:18]) < 0.003
    assert target[-1] < target[18]


def test_pd_cached_arrays_are_independent():
    a = pp.make_pd_schedule(26, 0.003416)
    a[:] = 0
    assert pp.make_pd_schedule(26, 0.003416).min() > 0


@pytest.mark.parametrize(
    "changes",
    [
        {"length": 6},
        {"length": 26.5},
        {"length": True},
        {"startup_echoes": -1},
        {"first_flip_angle": 3.0},
        {"final_flip_angle": 3.0},
        {"maximum_flip_angle": np.pi},
        {"plateau_fraction": 0.0},
        {"plateau_fraction": 1.0},
        {"echo_spacing": 0.0},
        {"reference_t1": np.inf},
        {"reference_t2": -1.0},
        {"initial_decay": np.nan},
    ],
)
def test_invalid_pd_controls_raise(changes):
    with pytest.raises(ValueError):
        pp.make_pd_schedule(**({"length": 26, "echo_spacing": 0.003416} | changes))


def test_batched_variable_pd_profiles_match_scalar_and_pad_with_zero():
    from pypulseqpp._pd import _make_pd_schedules

    lengths = np.array([26, 31, 41, 51])
    controls = np.array(
        [
            [120, 140, 100, 0.7],
            [121, 140, 110, 0.71],
            [119, 140, 130, 0.69],
            [120, 140, 140, 0.7],
        ]
    )
    profiles = _make_pd_schedules(lengths, controls, 6, 0.003416, 0.004162)
    for row, (length, control) in enumerate(zip(lengths, controls, strict=True)):
        actual = pp.make_pd_schedule(
            int(length),
            0.003416,
            first_echo_time=0.004162,
            first_flip_angle=np.deg2rad(control[0]),
            maximum_flip_angle=np.deg2rad(control[1]),
            final_flip_angle=np.deg2rad(control[2]),
            plateau_fraction=control[3],
        )
        np.testing.assert_allclose(profiles[row, :length], actual, atol=2e-7)
        assert np.all(profiles[row, length:] == 0)


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
        "pd_final_angle_periphery_deg": 140.0,
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


def test_pd_fit_preserves_first_peak_final_energy_and_effective_te():
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
    assert a[0] == pytest.approx(120, abs=1e-10)
    assert a.max() == pytest.approx(140, abs=1e-4)
    assert a[-1] == pytest.approx(100, abs=1e-4)
    assert np.sum(a * a) <= np.sum(b * b) * 1.00011
    assert float(fitted.definitions["EffectiveEchoTime"][0]) == pytest.approx(
        float(baseline.definitions["EffectiveEchoTime"][0]), abs=1.01e-4
    )


def test_dense_pd_transition_is_continuous_at_equal_peak_and_final():
    from pypulseqpp._pd import _make_pd_schedules

    place = np.linspace(0, 1, 128)
    fraction = place**2 * (3 - 2 * place)
    lengths = np.rint(51 - 25 * fraction).astype(int)
    controls = np.array([120, 140, 140, 0.7]) + fraction[:, None] * np.array(
        [5, 0, -40, 0.01]
    )
    profiles = _make_pd_schedules(lengths, controls, 6, 0.003416, 0.004162)
    active = np.arange(51)[None, :] < lengths[:, None]
    assert np.all(profiles[~active] == 0)
    np.testing.assert_allclose(np.rad2deg(profiles.max(1)), 140, atol=1e-4)
    np.testing.assert_allclose(
        np.rad2deg(profiles[np.arange(128), lengths - 1]), controls[:, 2], atol=1e-4
    )
    same_length = lengths[1:] == lengths[:-1]
    assert np.rad2deg(abs(np.diff(profiles, axis=0))[same_length]).max() < 2.0
