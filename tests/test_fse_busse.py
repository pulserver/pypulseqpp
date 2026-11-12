"""Prescribed-PSS schedules are the angles and startup echoes actually played."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences

# The schedules and their effective TE are computed by blochsim.trains.
pytest.importorskip("blochsim.trains")


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


def test_played_angles_startup_acquisitions_and_centre_match_prescription():
    seq = build()
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
    order = pp.make_pclo_order(coordinates, acquired, center_echo=centre - 6, pad=True)
    echo_map = np.full((32, 16), -1, dtype=np.int32)
    for train in order:
        for echo, index in enumerate(train):
            if index is not None:
                echo_map[sampled[index]] = echo
    from pypulseqpp._busse_opt import _ForwardSignal

    seen = []
    simulate = simulators.FSESimulator.simulate
    magnitudes = _ForwardSignal.__call__
    slopes = _ForwardSignal.slopes

    def timed(self, **protocol):
        layout = self.layout(flip=torch.full((4,), 120.0), ESP=3.416)
        # Event times are seconds even though the simulator's ESP is ms.
        np.testing.assert_allclose(
            [layout[1][0], layout[2][0], layout[4][0]],
            [0.004162 / 2, 0.004162, 0.004162 + 0.003416],
        )
        return simulate(self, **protocol)

    def record(self, flips):
        result = magnitudes(self, flips)
        seen.append((np.rad2deg(flips), result))
        return result

    def record_slopes(self, flips, directions):
        result = slopes(self, flips, directions)
        seen.append((np.rad2deg(flips), result[0]))
        return result

    monkeypatch.setattr(simulators.FSESimulator, "simulate", timed)
    monkeypatch.setattr(_ForwardSignal, "__call__", record)
    monkeypatch.setattr(_ForwardSignal, "slopes", record_slopes)
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
    assert all(flips[0] == pytest.approx(120) for flips, _ in seen)
    baseline = seen[0][1].reshape(3, 3, acquired + 6)
    final = seen[-1][1].reshape(3, 3, acquired + 6)
    np.testing.assert_allclose(np.rad2deg(angles), seen[-1][0])
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


@pytest.mark.parametrize("ordering", ["busse_radial", "pclo"])
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
        ordering="buonincontri",
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


@pytest.mark.parametrize("ordering", ["pclo", "shuffling", "busse_radial"])
def test_fixed_orderings_refuse_parameter_transitions(ordering):
    with pytest.raises(ValueError, match="Buonincontri"):
        build(ordering=ordering, busse_centre_echo=0, tr_periphery=0.4)


def test_shared_busse_uses_pclo_without_joint_shot_assignment(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("shared trains must not use Buonincontri assignment")

    monkeypatch.setattr(sequences.fse3D_sequence, "deal_trains", unexpected)
    auto = build()
    explicit = build(ordering="pclo")
    from test_zoo_fse import places

    for actual, expected in zip(places(auto), places(explicit), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert auto.definitions["ResolvedViewOrdering"] == "pclo"


def test_buonincontri_is_reserved_for_parameter_transitions():
    with pytest.raises(ValueError, match="centre/periphery parameter transition"):
        build(ordering="buonincontri")
    seq = build(tr_periphery=0.4)
    assert seq.definitions["ResolvedViewOrdering"] == "buonincontri"


def test_busse_schedule_slopes_are_the_derivative_of_the_schedule():
    from pypulseqpp._busse import _schedule_slopes

    angles, slopes = _schedule_slopes(46, 25.0, 70.0, 120.0, 25, 1.3)
    np.testing.assert_allclose(
        angles, pp.make_busse_schedule(46, centre_echo=25), atol=1e-12
    )
    step = 1e-6
    for column, name in enumerate(
        ("minimum_flip_angle", "centre_flip_angle", "maximum_flip_angle")
    ):
        prescription = {
            "minimum_flip_angle": np.deg2rad(25.0),
            "centre_flip_angle": np.deg2rad(70.0),
            "maximum_flip_angle": np.deg2rad(120.0),
        }
        moved = []
        for sign in (1, -1):
            changed = dict(prescription)
            changed[name] += sign * step
            moved.append(pp.make_busse_schedule(46, centre_echo=25, **changed))
        np.testing.assert_allclose(
            slopes[:, column], (moved[0] - moved[1]) / (2 * step), atol=1e-6
        )


def test_psf_radius_slopes_are_its_derivative():
    from pypulseqpp._busse_opt import _psf_radius

    rng = np.random.default_rng(5)
    forms = rng.standard_normal((2, 12, 12))
    energy, moment = (form @ form.T for form in forms)
    signal = rng.random((4, 12))
    directions = rng.standard_normal((2, 4, 12))
    _, slopes = _psf_radius(signal, directions, energy, moment)
    step = 1e-7
    for k in range(2):
        moved = [
            _psf_radius(
                signal + sign * step * directions[k], directions, energy, moment
            )[0]
            for sign in (1, -1)
        ]
        np.testing.assert_allclose(
            slopes[k], (moved[0] - moved[1]) / (2 * step), rtol=1e-6
        )


def test_ensemble_echo_slopes_are_the_derivative_of_the_simulated_echoes():
    pytest.importorskip("blochsim.simulators")
    from blochsim._execution import execution

    from pypulseqpp._busse_opt import _ForwardSignal

    module = sequences.fse3D_sequence
    flips = pp.make_busse_schedule(46, centre_echo=25)
    rng = np.random.default_rng(6)
    directions = 0.05 * rng.standard_normal((2, 46))
    with execution("cpu"):
        forward = _ForwardSignal(
            46,
            esp=0.003416,
            first_echo_time=0.004162,
            tr=1.5,
            t1_ms=module.DESIGN_T1_MS,
            t2_ms=module.DESIGN_T2_MS,
            b1_scales=module.DESIGN_B1_SCALES,
        )
        signal, slopes = forward.slopes(flips, directions)
        np.testing.assert_allclose(signal, forward(flips), rtol=1e-6)
        # The simulator runs in single precision, so the difference is taken
        # over a step its round-off does not swamp.
        step = 1e-2
        for k in range(2):
            moved = [forward(flips + sign * step * directions[k]) for sign in (1, -1)]
            np.testing.assert_allclose(
                slopes[k], (moved[0] - moved[1]) / (2 * step), atol=2e-4
            )


def _transition_deal(golden):
    deal = sequences.fse3D_sequence.deal_trains
    lengths, _, place = sequences.fse3D_sequence.shot_parameters(96, 8, 12, 0.5, 0.4)
    y, z = np.meshgrid(np.arange(-8, 8), np.arange(-8, 8), indexing="ij")
    coords = np.column_stack((y.ravel(), z.ravel())).astype(float)
    coords = coords[np.argsort(np.hypot(*coords.T), kind="stable")][: sum(lengths)]
    trains = deal(coords, list(lengths), place, 3, max(lengths), golden=golden)
    return coords, place, trains


def test_golden_ordering_deals_the_same_slots_and_views_as_the_sweep():
    _, _, sweep = _transition_deal(golden=False)
    _, _, golden = _transition_deal(golden=True)
    occupied = [[view is not None for view in train] for train in sweep]
    assert [[view is not None for view in train] for train in golden] == occupied
    dealt = sorted(view for train in golden for view in train if view is not None)
    assert dealt == sorted(v for train in sweep for v in train if v is not None)


def test_golden_ordering_gives_each_radial_section_in_the_order_of_the_shots_turns():
    coords, place, trains = _transition_deal(golden=True)
    radius = np.hypot(coords[:, 0], coords[:, 1])
    angle = np.arctan2(coords[:, 1], coords[:, 0])
    rank = np.argsort(np.argsort(place, kind="stable"), kind="stable")
    turn = (rank * sequences.fse3D_sequence._GOLDEN) % 1.0
    shot_of = {v: s for s, train in enumerate(trains) for v in train if v is not None}
    views = sorted(range(len(coords)), key=lambda i: (radius[i], angle[i]))
    views = views[1:] if radius[views[0]] == 0.0 else views
    for start in range(0, len(views), len(trains)):
        section = sorted(views[start : start + len(trains)], key=lambda i: angle[i])
        turns = [turn[shot_of[view]] for view in section]
        assert turns == sorted(turns)


def test_golden_ordering_is_reserved_for_parameter_transitions():
    with pytest.raises(ValueError, match="centre/periphery parameter transition"):
        build(ordering="buonincontri_golden")
    seq = build(ordering="buonincontri_golden", tr_periphery=0.4)
    assert seq.check_timing()[0]
    assert seq.definitions["ResolvedViewOrdering"] == "buonincontri_golden"
