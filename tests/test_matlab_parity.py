"""The routines MATLAB Pulseq defines and upstream PyPulseq has not ported.

Ported from the reference toolbox's `test_rotate3_d`, `test_sim_rf` and
`test_make_hexagon_gradient_area`.
"""

import math

import numpy as np
import pytest

import pypulseqpp as pp


@pytest.fixture
def system():
    return pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


# -- rotate_3d --------------------------------------------------------------


def about(axis, angle):
    """A turn of ``angle`` about one of the three axes, as a matrix."""
    c, s = math.cos(angle), math.sin(angle)
    return {
        "x": np.array(((1, 0, 0), (0, c, -s), (0, s, c))),
        "y": np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))),
        "z": np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))),
    }[axis]


def areas(events):
    """What each axis is left carrying."""
    found = dict.fromkeys("xyz", 0.0)
    for event in events:
        if hasattr(event, "channel"):
            found[event.channel] = event.area
    return found


def test_the_identity_leaves_every_gradient_where_it_was(system):
    rotated = pp.rotate_3d(
        np.eye(3),
        pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
        pp.make_trapezoid("y", area=2000, duration=2e-3, system=system),
    )

    assert len(rotated) == 2
    assert sorted(abs(event.area) for event in rotated) == pytest.approx(
        [1000, 2000], abs=10
    )


@pytest.mark.parametrize(
    ("axis", "source", "target"), [("z", "x", "y"), ("x", "y", "z"), ("y", "z", "x")]
)
def test_a_quarter_turn_carries_a_gradient_onto_the_next_axis(
    system, axis, source, target
):
    gradient = pp.make_trapezoid(source, area=1000, duration=2e-3, system=system)

    turned = areas(pp.rotate_3d(about(axis, math.pi / 2), gradient))

    assert abs(turned[target]) == pytest.approx(abs(gradient.area), abs=10)


@pytest.mark.parametrize(
    ("axis", "source", "created"), [("z", "x", "y"), ("x", "y", "z"), ("y", "z", "x")]
)
def test_an_eighth_turn_splits_a_gradient_between_two_axes(
    system, axis, source, created
):
    gradient = pp.make_trapezoid(source, area=1000, duration=2e-3, system=system)

    turned = areas(pp.rotate_3d(about(axis, math.pi / 4), gradient))

    assert turned[source] == pytest.approx(
        gradient.area * math.cos(math.pi / 4), abs=10
    )
    assert turned[created] == pytest.approx(
        gradient.area * math.sin(math.pi / 4), abs=10
    )


def test_an_oblique_turn_keeps_the_area_and_undoes_itself(system):
    gx = pp.make_trapezoid("x", area=1000, duration=2e-3, system=system)
    axis = np.array((0.5, 0.3, 0.6))
    axis /= np.linalg.norm(axis)
    angle = math.radians(70)
    cross = np.array(
        ((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0))
    )
    rotation = (
        np.eye(3) * math.cos(angle)
        + (1 - math.cos(angle)) * np.outer(axis, axis)
        + cross * math.sin(angle)
    )

    forward = pp.rotate_3d(rotation, gx)
    turned = areas(forward)

    assert all(abs(turned[channel]) > 1 for channel in "xyz")
    assert np.linalg.norm([turned[channel] for channel in "xyz"]) == pytest.approx(
        abs(gx.area), abs=10
    )

    back = areas(pp.rotate_3d(rotation.T, *forward))

    assert back["x"] == pytest.approx(gx.area, abs=15)
    assert abs(back["y"]) == pytest.approx(0, abs=15)
    assert abs(back["z"]) == pytest.approx(0, abs=15)


@pytest.mark.parametrize(
    "named",
    [
        about("z", math.pi / 2),
        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
        [math.pi / 2],
        [math.pi / 2, 0.0],
    ],
    ids=["matrix", "quaternion", "one angle", "two angles"],
)
def test_one_turn_named_four_ways_is_one_turn(system, named):
    gx = pp.make_trapezoid("x", area=1000, duration=2e-3, system=system)

    turned = areas(pp.rotate_3d(named, gx))

    assert abs(turned["y"]) == pytest.approx(abs(gx.area), abs=10)


def test_what_is_not_a_gradient_is_not_rotated(system):
    rotated = pp.rotate_3d(
        np.eye(3),
        pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
        pp.make_adc(num_samples=128, duration=1e-3, system=system),
    )

    assert [event.type for event in rotated] == ["adc", "trap"]


def test_an_axis_carries_one_gradient_at_a_time(system):
    with pytest.raises(ValueError, match="one gradient"):
        pp.rotate_3d(
            np.eye(3),
            pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
            pp.make_trapezoid("x", area=500, duration=2e-3, system=system),
        )


def test_a_rotation_has_to_be_one_of_the_forms_that_names_one(system):
    with pytest.raises(ValueError, match="rotation"):
        pp.rotate_3d([1, 2, 3], pp.make_trapezoid("x", area=1000, duration=2e-3))


# -- sim_rf -----------------------------------------------------------------


def on_resonance(f):
    return int(np.argmin(abs(f)))


def test_a_hard_ninety_takes_z_into_the_transverse_plane():
    rf = pp.make_block_pulse(math.pi / 2, duration=0.5e-3)

    mz_z, mz_xy, f = pp.sim_rf(rf)[:3]

    assert mz_z[on_resonance(f)] == pytest.approx(0, abs=0.15)
    assert abs(mz_xy[on_resonance(f)]) == pytest.approx(1, abs=0.15)


def test_a_hard_one_eighty_inverts_it():
    rf = pp.make_block_pulse(math.pi, duration=0.7e-3)

    mz_z, _, f = pp.sim_rf(rf)[:3]

    assert mz_z[on_resonance(f)] == pytest.approx(-1, abs=0.15)


def test_every_answer_is_one_value_per_frequency():
    rf = pp.make_block_pulse(math.pi / 2, duration=0.5e-3)

    mz_z, mz_xy, f, ref_eff, mx_xy, my_xy = pp.sim_rf(rf)

    assert {len(part) for part in (mz_z, mz_xy, ref_eff, mx_xy, my_xy)} == {len(f)}


def test_a_slice_selective_sinc_excites_the_slice_it_selects():
    rf = pp.make_sinc_pulse(math.pi / 2, duration=4e-3, time_bw_product=4)

    mz_z, mz_xy, f = pp.sim_rf(rf, 0)[:3]
    bandwidth = pp.calc_rf_bandwidth(rf)

    assert mz_z[on_resonance(f)] < 0.3
    # And leaves the magnetisation alone well outside the band.
    far = int(np.argmin(abs(f - 3 * bandwidth)))
    assert mz_z[far] > 0.9
    assert abs(mz_xy[far]) < 0.1


def test_a_refocusing_pulse_is_not_rephased_after_itself():
    """What rephase_factor defaults to is read off what the pulse is for."""
    rf = pp.make_sinc_pulse(math.pi, duration=4e-3, time_bw_product=4, use="refocusing")

    ref_eff, f = pp.sim_rf(rf)[3], pp.sim_rf(rf)[2]

    assert abs(ref_eff[on_resonance(f)]) > 0.9


# -- make_hexagon_gradient_area ---------------------------------------------


HEXAGON_CASES = [
    ("x", 0.0, 0.0, 5000, True),
    ("y", 0.0, 0.0, -5000, True),
    ("z", 0.0, 0.0, 100, True),
    ("x", 0.0, 0.0, 50000, True),
    ("x", 0.5, 0.0, 3000, True),
    ("y", 0.0, 0.3, 4000, True),
    ("z", 0.2, 0.4, 8000, True),
    ("x", -0.3, -0.1, -6000, True),
    ("y", 0.2, -0.2, 2000, True),
    ("z", -0.3, 0.1, -3000, True),
    ("x", 0.25, 0.25, 7000, True),
    ("y", 0.8, 0.8, 500, False),
    ("x", 0.3, 0.3, 0, True),
    ("y", 0.3, -0.3, 0, True),
    ("z", 0.3, 0.0, 0, True),
    ("x", -0.3, -0.075, 0, True),
    ("y", 0.8, 0.8, 0, True),
]


def carries(gradient, area, first, last, system):
    """That a waveform encloses the area asked for, inside the limits."""
    assert gradient.area == pytest.approx(area, abs=1)
    assert gradient.waveform[0] == pytest.approx(first, abs=1e-3)
    assert gradient.waveform[-1] == pytest.approx(last, abs=1e-3)
    slew = np.diff(gradient.waveform) / np.diff(gradient.tt)
    assert np.max(abs(slew)) <= system.max_slew * 1.01
    assert np.max(abs(gradient.waveform)) <= system.max_grad * 1.01


@pytest.mark.parametrize(
    ("channel", "start", "end", "area", "shorter"),
    HEXAGON_CASES,
    ids=[f"{c}:{s}->{e}:{a}" for c, s, e, a, _ in HEXAGON_CASES],
)
def test_a_hexagon_carries_the_area_between_two_amplitudes(
    system, channel, start, end, area, shorter
):
    first = start * system.max_grad
    last = end * system.max_grad

    hexagon, _, _ = pp.make_hexagon_gradient_area(
        channel, first, last, area, system=system
    )

    carries(hexagon, area, first, last, system)


@pytest.mark.parametrize(
    ("channel", "start", "end", "area"),
    [case[:4] for case in HEXAGON_CASES if case[4]],
    ids=[f"{c}:{s}->{e}:{a}" for c, s, e, a, keep in HEXAGON_CASES if keep],
)
def test_a_hexagon_is_never_longer_than_the_trapezoid_it_stands_in_for(
    system, channel, start, end, area
):
    first = start * system.max_grad
    last = end * system.max_grad

    trapezoid, _, _ = pp.make_extended_trapezoid_area(
        area, channel, first, last, system=system
    )
    hexagon, _, _ = pp.make_hexagon_gradient_area(
        channel, first, last, area, system=system
    )

    carries(trapezoid, area, first, last, system)
    assert hexagon.shape_dur <= trapezoid.shape_dur + 1e-9


def test_an_end_amplitude_over_the_limit_is_refused(system):
    with pytest.raises(ValueError, match="grad_start"):
        pp.make_hexagon_gradient_area(
            "x", 2 * system.max_grad, 0.0, 1000, system=system
        )
