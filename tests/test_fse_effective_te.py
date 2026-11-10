"""Reference contrast TE agrees with spin-echo timing and independent Bloch evolution."""

from contextlib import suppress

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

# The schedules and their effective TE are computed by blochsim.trains.
pytest.importorskip("blochsim.trains")


@pytest.mark.parametrize("t1,t2", [(1.0, 0.1), (2.0, 0.05)])
def test_180_degree_refocusing_has_physical_echo_times(t1, t2):
    result = pp.calc_fse_effective_te(
        np.full(20, np.pi),
        0.003416,
        first_echo_time=0.004162,
        reference_t1=t1,
        reference_t2=t2,
    )
    np.testing.assert_allclose(result, 0.004162 + np.arange(20) * 0.003416, atol=1e-14)


def test_relaxation_normalized_epg_matches_independent_isochromats():
    angles = pp.make_busse_schedule(86)
    phase = np.exp(2j * np.pi * np.arange(2048) / 2048)
    signals = []
    for t1, t2 in ((1.0, 0.1), (np.inf, np.inf)):
        transverse = np.ones(2048, dtype=complex)
        longitudinal = np.zeros(2048)
        signal = []
        for echo, angle in enumerate(angles):
            half = (0.004162 if echo == 0 else 0.003416) / 2
            e1, e2 = np.exp(-half / t1), np.exp(-half / t2)
            transverse *= phase * e2
            longitudinal = longitudinal * e1 + 1 - e1
            imaginary = transverse.imag.copy()
            transverse = transverse.real + 1j * (
                np.cos(angle) * imaginary - np.sin(angle) * longitudinal
            )
            longitudinal = np.sin(angle) * imaginary + np.cos(angle) * longitudinal
            transverse *= phase * e2
            longitudinal = longitudinal * e1 + 1 - e1
            signal.append(abs(transverse.mean()))
        signals.append(np.asarray(signal))
    independent = -0.1 * np.log(signals[0] / signals[1])
    result = pp.calc_fse_effective_te(angles, 0.003416, first_echo_time=0.004162)
    np.testing.assert_allclose(result, independent, atol=2e-13)
    assert result[45] == pytest.approx(0.06584688767461871, abs=1e-12)


def test_batched_contrast_te_is_independent_for_each_train():
    trains = np.array([pp.make_busse_schedule(46), np.full(46, np.pi)])
    batch = pp.calc_fse_effective_te(trains, 0.004)
    for row, angles in enumerate(trains):
        np.testing.assert_allclose(batch[row], pp.calc_fse_effective_te(angles, 0.004))


def test_zero_coherence_has_no_defined_effective_te():
    assert np.isnan(pp.calc_fse_effective_te([0.0], 0.004)[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"flip_angles": []},
        {"flip_angles": [np.nan]},
        {"flip_angles": [-1]},
        {"flip_angles": [4]},
        {"echo_spacing": 0},
        {"first_echo_time": -1},
        {"reference_t1": np.inf},
        {"reference_t2": 0},
    ],
)
def test_invalid_effective_te_models_raise(kwargs):
    with pytest.raises(ValueError):
        pp.calc_fse_effective_te(
            **({"flip_angles": [1.0], "echo_spacing": 0.004} | kwargs)
        )


@pytest.mark.parametrize("target", [0.02, 0.054, 0.08])
def test_centre_search_matches_exhaustive_feasible_search(target):
    from pypulseqpp._fse_contrast import _select_busse_centre

    etl, startup = 40, 6
    controls = (25.0, 70.0, 120.0)
    feasible = {}
    for centre in range(etl - 1):
        with suppress(ValueError):
            angles = pp.make_busse_schedule(
                etl + startup,
                centre_echo=startup + centre,
                startup_echoes=startup,
            )
            feasible[centre] = pp.calc_fse_effective_te(
                angles[: startup + centre + 1], 0.003416, first_echo_time=0.004162
            )[-1]
    expected = min(feasible, key=lambda c: (abs(feasible[c] - target), c))
    assert (
        _select_busse_centre(
            etl, etl, startup, controls, controls, 1.3, 0.003416, 0.004162, target
        )
        == expected
    )


def test_played_centre_echo_uses_effective_te_without_changing_spacing():
    seq = sequences.fse3D_sequence(
        pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        n_x=32,
        n_y=16,
        n_z=8,
        etl=64,
        tr=1.8,
        te=None,
        n_dummy=0,
        flip_modulation="busse",
        busse_effective_te=0.054,
    )
    physical = float(seq.definitions["TE"][0])
    effective = float(seq.definitions["EffectiveEchoTime"][0])
    assert physical > effective
    assert effective == pytest.approx(0.054, abs=0.002)
    from test_zoo_fse import echo_times, places

    line, partition, _ = places(seq)
    centre = np.flatnonzero((line == 8) & (partition == 4))
    np.testing.assert_allclose(echo_times(seq)[centre], physical, atol=1e-10)
    assert seq.check_timing()[0]


def test_control_fit_preserves_selected_effective_te():
    pytest.importorskip("blochsim.simulators")
    settings = {
        "system": pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 40,
        "tr": 1.8,
        "te": None,
        "n_dummy": 0,
        "flip_modulation": "busse",
        "busse_effective_te": 0.054,
    }
    baseline = sequences.fse3D_sequence(**settings)
    fitted = sequences.fse3D_sequence(**settings, busse_optimize=True)
    assert int(fitted.definitions["BusseCentreEcho"][0]) == int(
        baseline.definitions["BusseCentreEcho"][0]
    )
    assert float(fitted.definitions["EffectiveEchoTime"][0]) == pytest.approx(
        float(baseline.definitions["EffectiveEchoTime"][0]), abs=1.01e-4
    )
    assert float(fitted.definitions["TE"][0]) == float(baseline.definitions["TE"][0])
    assert fitted.check_timing()[0]


def test_short_tr_periphery_fit_converges_at_effective_te_centre():
    pytest.importorskip("blochsim.simulators")
    from pypulseqpp._busse_opt import optimize_controls

    module = sequences.fse3D_sequence
    shape = (192, 160)
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (24, 24), caipi_shift=1, elliptical=True, train_length=1
    )
    sampled = sorted({*calibration, *imaging})
    coords = (np.asarray(sampled) - np.array(shape) // 2) / shape
    plane = np.full(shape, -1, dtype=np.int32)
    for train in pp.make_pclo_order(coords, 20, center_echo=11, pad=True):
        for echo, index in enumerate(train):
            if index is not None:
                plane[sampled[index]] = echo
    fitted = optimize_controls(
        26,
        minimum_flip_angle=np.deg2rad(25),
        centre_flip_angle=np.deg2rad(70),
        maximum_flip_angle=np.deg2rad(120),
        centre_echo=17,
        startup_echoes=6,
        initial_decay=1.3,
        esp=0.003416,
        first_echo_time=0.004162,
        tr=0.177152,
        echo_map=plane,
        t1_ms=module.DESIGN_T1_MS,
        t2_ms=module.DESIGN_T2_MS,
        b1_scales=module.DESIGN_B1_SCALES,
        contrast_pair=module.DESIGN_CONTRAST,
    )
    assert np.all(np.isfinite(fitted))
    assert len(fitted) == 26
    assert np.rad2deg(fitted[-1]) == pytest.approx(120, abs=1e-6)
