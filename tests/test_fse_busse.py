"""Prescribed-PSS schedules are the angles and startup echoes actually played."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences


def value(seq, name):
    return float(np.atleast_1d(seq.definitions[name])[0])


def build(**protocol):
    return sequences.fse3D_sequence(
        pp.Opts(rf_dead_time=100e-6, rf_ringdown_time=20e-6),
        n_x=32,
        n_y=16,
        n_z=8,
        etl=20,
        te=None,
        tr=0.5,
        n_dummy=1,
        flip_modulation="busse",
        **protocol,
    )


@pytest.mark.parametrize("overlap", [False, True])
def test_played_angles_startup_acquisitions_and_centre_match_prescription(overlap):
    seq = build(rf_overlap=overlap)
    assert seq.check_timing()[0]
    expected = pp.make_busse_schedule(26)
    time = 0.0
    flips, echoes, acquisitions, excitation = [], [], [], None
    last_rf = -1
    for i in range(1, len(seq.block_events) + 1):
        block = seq.get_block(i)
        if block.rf is not None:
            rf = block.rf
            if rf.use == "excitation":
                flips.append([])
                acquisitions.append([])
                excitation = time + rf.delay + rf.center
                last_rf = -1
            else:
                last_rf += 1
                # RF integral rather than metadata measures the played flip.
                flips[-1].append(abs(np.trapezoid(rf.signal, rf.t) * 2 * np.pi))
        if block.adc is not None:
            acquisitions[-1].append(last_rf)
            centre = value(seq, "kSpaceCenterSample")
            echoes.append(
                time + block.adc.delay + centre * block.adc.dwell - excitation
            )
        time += block.block_duration
    np.testing.assert_allclose(flips, np.tile(expected, (len(flips), 1)), atol=1e-10)
    assert acquisitions[0] == []  # Dummy train.
    assert all(train == list(range(6, 26)) for train in acquisitions[1:])
    assert len(flips) == int(value(seq, "NumShots")) + 1
    assert value(seq, "StartupEchoes") == 6
    assert value(seq, "BusseCentreEcho") == 9
    echo = value(seq, "FirstEchoSpacing") + np.arange(26) * value(seq, "EchoSpacing")
    np.testing.assert_allclose(
        np.asarray(echoes).reshape(-1, 20)[0], echo[6:], atol=1e-9
    )
    assert value(seq, "TE") == pytest.approx(echo[15])
    np.testing.assert_allclose(
        seq.definitions["RefocusingFlipAngles"], np.rad2deg(expected)
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"te": 0.028}, "te=None"),
        ({"etl_periphery": 8}, "final acquired echo"),
        ({"busse_centre_echo": 20}, "acquired echo"),
    ],
)
def test_incompatible_prescriptions_raise(kwargs, message):
    arguments = {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "etl": 20,
        "te": None,
        "flip_modulation": "busse",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        sequences.fse3D_sequence(**arguments)


def test_outer_fit_keeps_family_and_improves_forward_psf(monkeypatch):
    torch = pytest.importorskip("torch")
    simulators = pytest.importorskip("blochsim.simulators")
    from pypulseqpp._busse_opt import optimize_controls, psf_weights

    module = sequences.fse3D_sequence
    acquired = 40
    centre = 6 + (acquired - 1) // 2
    cal, img = pp.make_cartesian_plane_sampling(
        (32, 16), elliptical=True, train_length=acquired
    )
    sampled = sorted({*cal, *img})
    coordinates = (np.asarray(sampled) - [16, 8]) / [32, 16]
    lengths, _, place = module.shot_parameters(
        len(sampled), acquired, acquired, 3.431, 3.431
    )
    order = module.deal_trains(
        coordinates, lengths, np.zeros_like(place), centre - 6, acquired
    )
    echo_map = np.full((32, 16), -1, dtype=np.int32)
    for train in order:
        for echo, index in enumerate(train):
            if index is not None:
                echo_map[sampled[index]] = echo
    seen = []
    simulate = simulators.FSESimulator.simulate

    def record(self, **protocol):
        layout = self.layout(flip=protocol["flip"], ESP=3.416)
        # Event times are seconds even though the simulator's ESP is ms.
        np.testing.assert_allclose(
            [layout[1][0], layout[2][0], layout[4][0]],
            [0.004162 / 2, 0.004162, 0.004162 + 0.003416],
        )
        assert torch.allclose(
            protocol["flip"][..., 0], torch.full_like(protocol["flip"][..., 0], 120)
        )
        result = simulate(self, **protocol)
        seen.append((protocol["flip"].numpy().copy(), result.abs().numpy().copy()))
        return result

    monkeypatch.setattr(simulators.FSESimulator, "simulate", record)
    angles = optimize_controls(
        acquired + 6,
        minimum_flip_angle=np.deg2rad(25),
        centre_flip_angle=np.deg2rad(70),
        maximum_flip_angle=np.deg2rad(120),
        centre_echo=centre,
        startup_echoes=6,
        initial_decay=1.3,
        esp=0.003416,
        first_echo_time=0.004162,
        tr=3.431,
        echo_map=echo_map,
        t1_ms=module.DESIGN_T1_MS,
        t2_ms=module.DESIGN_T2_MS,
        b1_scales=module.DESIGN_B1_SCALES,
        contrast_pair=module.DESIGN_CONTRAST,
    )
    reference = pp.make_busse_schedule(acquired + 6)
    low, middle = np.rad2deg(angles[[np.argmin(angles), centre]])
    assert 20 - 1e-6 <= low <= 30 + 1e-6 and 65 - 1e-6 <= middle <= 75 + 1e-6
    np.testing.assert_allclose(
        angles,
        pp.make_busse_schedule(
            acquired + 6,
            minimum_flip_angle=np.deg2rad(low),
            centre_flip_angle=np.deg2rad(middle),
        ),
        atol=1e-8,
    )
    assert angles[0] == reference[0] and angles[-1] == pytest.approx(reference[-1])
    assert sum(angles**2) <= sum(reference**2) * (1 + 1e-6)
    baseline = seen[0][1].reshape(3, 3, acquired + 6)
    final = seen[-1][1][0].reshape(3, 3, acquired + 6)
    np.testing.assert_allclose(np.rad2deg(angles), seen[-1][0][0])
    contrast = (final[:, 2, centre] - final[:, 0, centre]) / (
        baseline[:, 2, centre] - baseline[:, 0, centre]
    )
    assert contrast.min() >= 0.955 - 1e-4
    energy, moment = psf_weights(echo_map, acquired)

    def radius(signal):
        data = signal.reshape(-1, acquired + 6)[:, 6:]
        return np.sqrt(
            np.einsum("bi,ij,bj->b", data, moment, data)
            / np.einsum("bi,ij,bj->b", data, energy, data)
        ).mean()

    assert radius(final) <= radius(baseline) * (1 + 1e-4)


@pytest.mark.parametrize("ordering", ["busse_radial", "adaptive_radial"])
def test_busse_and_adaptive_orders_keep_caipi_support_and_declared_centre(ordering):
    from test_zoo_fse import definition, echo_times, places

    centre = 0 if ordering == "busse_radial" else 9
    seq = build(
        ordering=ordering,
        busse_centre_echo=centre,
        ry=2,
        rz=2,
        caipi_shift=1,
        n_acs_y=4,
        n_acs_z=2,
    )
    assert seq.check_timing()[0]
    line, partition, echo = places(seq)
    centre_view = np.flatnonzero((line == 8) & (partition == 4))
    assert len(centre_view) == 1
    assert echo[centre_view[0]] == centre + 6
    np.testing.assert_allclose(
        echo_times(seq)[centre_view], definition(seq, "TE"), atol=1e-9
    )
    calibration, imaging = pp.make_cartesian_plane_sampling(
        (16, 8), (2, 2), (4, 2), caipi_shift=1, elliptical=True, train_length=20
    )
    assert set(zip(line, partition, strict=True)) == {*calibration, *imaging}
    if ordering == "busse_radial":
        distance = np.hypot((line - 8) / 16, (partition - 4) / 8)
        assert all(
            max(distance[echo == e]) <= min(distance[echo == e + 1]) + 1e-12
            for e in range(6, 25)
        )


def test_variable_busse_controls_follow_cubic_transition_and_zero_padding():
    from test_zoo_fse import echo_times, pulses, trains

    seq = build(
        ordering="adaptive_radial",
        tr_periphery=0.4,
        etl_periphery=28,
        busse_minimum_angle_periphery_deg=30,
        busse_centre_angle_periphery_deg=65,
    )
    assert seq.check_timing()[0]
    views = trains(seq)[1:]  # Dummy excluded.
    lengths, _, place = sequences.fse3D_sequence.shot_parameters(
        sum(len(train) for train in views), 20, 28, 0.5, 0.4
    )
    refocusing = [a for _, use, a in pulses(seq) if use == "refocusing"]
    # All pulses use the same hard-pulse shape and calibration amplitude.
    actual = np.reshape(refocusing, (-1, 34))[1:]
    actual *= 120.0 / actual[0, 0]
    controls = np.array([25.0, 70.0, 120.0]) + sequences.fse3D_sequence.cubic(place)[
        :, None
    ] * [5.0, -5.0, 0.0]
    for row, (length, target) in enumerate(zip(lengths, controls, strict=True)):
        expected = np.rad2deg(
            pp.make_busse_schedule(
                length + 6,
                minimum_flip_angle=np.deg2rad(target[0]),
                centre_flip_angle=np.deg2rad(target[1]),
                maximum_flip_angle=np.deg2rad(target[2]),
                centre_echo=15,
            )
        )
        np.testing.assert_allclose(actual[row, : length + 6], expected, atol=2e-6)
        assert np.all(actual[row, length + 6 :] == 0)
        assert all(6 <= echo < length + 6 for _, _, echo in views[row])
    played_echo_times = echo_times(seq)
    offset = 0
    spacing = value(seq, "EchoSpacing")
    for train in views:
        count = len(train)
        acquired_times = played_echo_times[offset : offset + count]
        np.testing.assert_allclose(np.diff(acquired_times), spacing, atol=1e-12)
        expected_times = value(seq, "TE") + spacing * (
            np.array([echo for _, _, echo in train]) - 15
        )
        np.testing.assert_allclose(acquired_times, expected_times, atol=1e-10)
        offset += count
    assert np.allclose(np.atleast_1d(seq.definitions["TRPeriphery"]), 0.4)


def test_variable_trains_require_adaptive_order_and_busseradial_requires_first_echo():
    with pytest.raises(ValueError, match="radial"):
        build(ordering="busse_radial", busse_centre_echo=0, tr_periphery=0.4)
    with pytest.raises(ValueError, match="centre echo 0"):
        build(ordering="busse_radial")


def test_single_shot_variable_train_pads_to_the_common_timing_layout():
    seq = build(ry=16, rz=8, n_acs_y=0, n_acs_z=0, etl_periphery=28)
    assert seq.check_timing()[0]
    assert int(value(seq, "NumShots")) == 1
