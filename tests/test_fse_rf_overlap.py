"""Gradient continuity and timing of nonselective 3D FSE with RF overlap."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences


@pytest.fixture
def system():
    return pp.cap_system(
        pp.apply_system_derates(
            pp.Opts(
                gamma=42575900.0,
                B0=7.001091,
                max_grad=100.0,
                grad_unit="mT/m",
                max_slew=115.0,
                slew_unit="T/m/s",
                max_b1=25.0,
                b1_unit="uT",
                rf_dead_time=232e-6,
                rf_ringdown_time=4e-6,
                rf_raster_time=2e-6,
                grad_raster_time=4e-6,
                adc_raster_time=2e-6,
                block_duration_raster=4e-6,
            )
        ),
        max_grad=80.0,
        max_slew=200.0,
    )


def readout(system, **kwargs):
    excitation = sequences.SpatialSelectiveExcitation(
        system, 90.0, 0.112, duration_s=0.001, time_bw_product=4.0, is_slab=True
    )
    refocusing = sequences.NonSelectiveRefocusing(system, duration_s=504e-6)
    return sequences.FseReadout3D(
        system,
        excitation.rf,
        excitation.gz,
        rf_ref=refocusing.rf_ref,
        fov=(0.21, 0.1596, 0.112),
        matrix=(276, 276, 140),
        etl=3,
        oversampling=2.0,
        readout_bandwidth_hz=500000.0,
        spoiling_cycles=0.5,
        **kwargs,
    )


def test_rf_overlap_preserves_echo_locations_and_rf_midpoints(system):
    fse = readout(system, rf_overlap=True)
    assert fse.esp == pytest.approx(0.003416)
    assert fse.esp_first == pytest.approx(0.004160)
    assert fse.check_timing()[0]
    k, _, excitation, refocusing, time = fse.calculate_kspace()
    lines = np.asarray(k).reshape(3, 3, 552)
    centres = (lines[:, :, 275] + lines[:, :, 276]) / 2
    np.testing.assert_allclose(centres[0], 0.0, atol=0.01)
    np.testing.assert_allclose(centres[1], 276 / (2 * 0.1596), atol=0.01)
    np.testing.assert_allclose(centres[2], 140 / (2 * 0.112), atol=0.01)
    echoes = np.asarray(time).reshape(3, 552)[:, [275, 276]].mean(1)
    np.testing.assert_allclose(np.diff(echoes), fse.esp, atol=1e-12)
    previous = np.r_[float(excitation[0]), echoes[:-1]]
    np.testing.assert_allclose(refocusing, (previous + echoes) / 2, atol=2e-6)
    gradients = fse.seq.rf_gradients()
    np.testing.assert_allclose(np.asarray(gradients.gradient)[1:], 0.0, atol=0.01)


def test_reusing_overlap_templates_starts_each_train_at_zero_phase_gradient(system):
    fse = readout(system, rf_overlap=True)
    seq = pp.Sequence(system)
    for _ in range(2):
        fse.layout.excitation(seq, [])
        previous = (0.0, 0.0)
        for echo, (ky, kz) in enumerate(((0.5, -0.25), (-0.8, 0.6), (0.0, 0.0))):
            fse.layout.refocusing(seq, echo == 0, *previous)
            if echo == 0:
                seq.add_block(pp.make_delay((fse.esp_first - fse.esp) / 2))
            fse.layout.encoding(seq, ky, kz)
            seq.add_block(fse.gx, fse.adc)
            fse.layout.rewinding(seq, ky, kz, echo == 2)
            previous = (ky, kz)
        seq.add_block(pp.make_delay(0.001))
    ok, errors = seq.check_timing()
    assert ok, errors


def test_rf_overlap_requested_spacing_is_the_played_spacing(system):
    fse = readout(system, rf_overlap=True, esp=0.0038)
    assert fse.esp == pytest.approx(0.0038)
    assert fse.check_timing()[0]


def test_rf_overlap_requires_full_echoes(system):
    with pytest.raises(ValueError, match="full echoes"):
        readout(system, rf_overlap=True, partial_echo=0.75)


def test_refocusing_cost_uses_the_first_pulse_and_first_echo(monkeypatch):
    torch = pytest.importorskip("torch")
    simulators = pytest.importorskip("blochsim.simulators")
    from blochsim._execution import execution

    original = simulators.FSESimulator.simulate
    seen = []

    def record(self, **kwargs):
        seen.append(
            (kwargs["flip"].detach().clone(), self.layout(flip=kwargs["flip"], ESP=4.0))
        )
        return original(self, **kwargs)

    monkeypatch.setattr(simulators.FSESimulator, "simulate", record)
    with execution("cpu"):
        flips = sequences.fse3D_sequence.design_trains(
            100.0,
            False,
            [8],
            [1.0],
            np.zeros(1),
            5,
            8,
            0.004,
            first_refocusing_angle_deg=180.0,
            first_echo_time=0.008,
        )
    assert flips[0, 0] == 180.0
    assert seen
    for flip, program in seen:
        assert torch.all(flip[..., 0] == 180.0)
        assert program[1][0] == pytest.approx(0.004)
        assert program[2][0] == pytest.approx(0.008)
        assert program[4][0] - program[2][0] == pytest.approx(0.004)


def test_end_angle_optimization_keeps_the_starting_profiles_rf_budget():
    pytest.importorskip("blochsim.simulators")
    from blochsim._execution import execution
    from scipy.interpolate import CubicHermiteSpline

    with execution("cpu"):
        flips = sequences.fse3D_sequence.design_trains(
            100.0,
            False,
            [80],
            [3.431],
            np.zeros(1),
            16,
            80,
            0.003416,
            first_refocusing_angle_deg=180.0,
            first_echo_time=0.004162,
            b1_scales=(0.8, 1.0, 1.2),
            optimize_end_angle=True,
        )[0]
    # Independent Hermite interpolation of the stated initial control points.
    initial = CubicHermiteSpline([1, 5, 17, 80], [100, 60, 100, 100], [0, 0, 0, 0])(
        np.arange(1, 81)
    )
    initial[0] = 180.0
    assert flips[0] == 180.0
    assert flips[16] == pytest.approx(100.0)
    assert np.all((flips[1:] >= 20.0) & (flips[1:] <= 100.0))
    assert np.sum(flips**2) <= np.sum(initial**2) * (1 + 1e-6)
