"""Waveform corners and RF/ADC timing compared with the reference toolbox."""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert

import pypulseqpp as pp


@pytest.fixture
def both(reference_name, build_reference):
    """The same sequence as the toolbox holds it and as the core holds it."""
    theirs = build_reference()
    ours = pp.Sequence(theirs.system)
    ours._native = convert.to_core(theirs)
    return theirs, ours


def assert_same(expected, got, where):
    expected, got = np.asarray(expected), np.asarray(got)
    assert expected.shape == got.shape, f"{where}: shape"
    if expected.size:
        assert np.allclose(expected, got, rtol=1e-9, atol=1e-12), f"{where}: values"


def test_the_gradient_waveforms_are_the_toolboxs(both):
    theirs, ours = both

    for axis, (expected, got) in enumerate(
        zip(theirs.waveforms(), ours.waveforms(), strict=True)
    ):
        assert_same(expected, got, f"axis {axis}")


def test_the_rf_envelope_is_the_toolboxs(both):
    """The fourth channel, when it is asked for."""
    theirs, ours = both

    expected = theirs.waveforms(append_RF=True)
    got = ours.waveforms(append_RF=True)

    assert len(got) == len(expected) == 4
    assert_same(expected[3], got[3], "rf")


def test_when_each_pulse_acts_is_the_toolboxs(both):
    theirs, ours = both

    _, their_excitation, their_refocusing, _, _, _ = theirs.waveforms_and_times()
    found = ours.waveforms_and_times(compat=False).rf

    assert_same(their_excitation, found.of("excitation", "undefined").tfp, "excitation")
    assert_same(their_refocusing, found.of("refocusing").tfp, "refocusing")


def test_when_each_sample_is_taken_is_the_toolboxs(both):
    """The toolbox's `fp_adc` is per sample, where upstream's is per window."""
    theirs, ours = both

    *_, their_t, their_fp, their_pm = theirs.waveforms_and_times()
    found = ours.waveforms_and_times(compat=False).adc

    assert_same(their_t, found.t, "t_adc")
    assert_same(
        their_fp, np.vstack((found.sample_frequency, found.sample_phase)), "fp_adc"
    )
    assert_same(their_pm, found.phase_modulation, "pm_adc")


def test_the_five_values_a_drop_in_caller_unpacks_are_upstreams(both):
    """`compat` is the default, and it is upstream's tuple, not the toolbox's."""
    _, ours = both

    got = ours.waveforms_and_times()

    assert len(got) == 5
    channels, excitation, refocusing, _t_adc, fp_adc = got
    assert len(channels) == 3
    assert excitation.shape[0] == refocusing.shape[0] == 3
    # Per ADC window, which is what upstream packs it as.
    assert fp_adc.ndim == 2 and fp_adc.shape[1] == 2
    assert fp_adc.shape[0] == len(ours.waveforms_and_times(compat=False).adc.block)


def test_every_use_a_pulse_can_have_is_reported():
    """Upstream's two buckets drop five of Pulseq's seven uses."""
    system = pp.Opts()
    sequence = pp.Sequence(system)
    uses = ("excitation", "refocusing", "inversion", "saturation", "preparation")
    for use in uses:
        sequence.add_block(
            pp.make_block_pulse(0.5, duration=1e-3, system=system, use=use)
        )

    found = sequence.waveforms_and_times(compat=False).rf

    assert found.use == uses
    assert list(found.block) == [1, 2, 3, 4, 5]
    assert len(found.of("inversion")) == 1
    # ...where the tuple a drop-in caller unpacks carries two of the five.
    _, excitation, refocusing, _, _ = sequence.waveforms_and_times()
    assert excitation.shape[1] + refocusing.shape[1] == 2


def test_the_adc_times_are_the_toolboxs(both):
    theirs, ours = both

    for i, (expected, got) in enumerate(
        zip(theirs.adc_times(), ours.adc_times(), strict=True)
    ):
        assert_same(expected, got, f"adc_times[{i}]")


def test_the_rf_times_are_the_toolboxs(both):
    theirs, ours = both

    for i, (expected, got) in enumerate(
        zip(theirs.rf_times(), ours.rf_times(), strict=True)
    ):
        assert_same(expected, got, f"rf_times[{i}]")


def test_the_gradients_as_polynomials_are_the_toolboxs(both):
    theirs, ours = both

    for axis, (expected, got) in enumerate(
        zip(theirs.get_gradients(), ours.get_gradients(), strict=True)
    ):
        assert (expected is None) == (got is None), f"axis {axis}: presence"
        if expected is None:
            continue
        assert_same(expected.x, got.x, f"axis {axis}: knots")
        assert np.allclose(expected.c, got.c, rtol=1e-7, atol=1e-9), f"axis {axis}"


# -- asking for part of a sequence -----------------------------------------


@pytest.mark.parametrize("window", [[2, 5], [1, np.inf], [3, 3]])
def test_a_range_of_blocks_expands_as_the_toolbox_expands_it(both, window):
    """The toolbox spells this one `blockRange`; here it is `block_range`."""
    theirs, ours = both

    for axis, (expected, got) in enumerate(
        zip(
            theirs.waveforms(blockRange=window),
            ours.waveforms(block_range=window),
            strict=True,
        )
    ):
        assert_same(expected, got, f"axis {axis}")


def test_a_range_of_time_expands_as_the_toolbox_expands_it(both):
    theirs, ours = both
    total = sum(theirs.block_durations.values())
    window = {"time_range": [0.2 * total, 0.6 * total]}

    for axis, (expected, got) in enumerate(
        zip(theirs.waveforms(**window), ours.waveforms(**window), strict=True)
    ):
        assert_same(expected, got, f"axis {axis}")


def test_a_block_range_and_a_time_range_are_not_both_accepted():
    with pytest.raises(ValueError, match="not both"):
        pp.Sequence(pp.Opts()).waveforms(block_range=[1, 2], time_range=[0, 1])


def test_a_time_range_runs_forwards():
    with pytest.raises(ValueError, match="must be after"):
        pp.Sequence(pp.Opts()).waveforms(time_range=[1.0, 0.5])


# -- what a waveform is ----------------------------------------------------


def test_a_trapezoid_is_its_four_corners():
    """However long the flat top, a trapezoid is four points."""
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_trapezoid("x", amplitude=1000, flat_time=50e-3, rise_time=1e-4)
    )

    times, values = sequence.waveforms()[0]

    assert list(values) == pytest.approx([0.0, 1000.0, 1000.0, 0.0])
    assert times == pytest.approx([0.0, 1e-4, 50.1e-3, 50.2e-3])


def test_a_trapezoid_with_no_flat_top_is_three():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_trapezoid("y", amplitude=1000, flat_time=0.0, rise_time=1e-4)
    )

    times, values = sequence.waveforms()[1]

    assert list(values) == pytest.approx([0.0, 1000.0, 0.0])
    assert times == pytest.approx([0.0, 1e-4, 2e-4])


def test_an_axis_that_plays_nothing_is_empty():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(pp.make_trapezoid("x", area=1000, duration=1e-3))

    assert sequence.waveforms()[1].shape == (2, 0)
    assert sequence.waveforms()[2].shape == (2, 0)


# -- a rotated block -------------------------------------------------------
#
# A rotation sends each axis's gradient onto all three, so what a rotated
# block plays on one axis is a sum of the three it was given. The waveforms
# are piecewise linear, so that sum is exact on the union of their corners --
# a linear combination of straight lines is a straight line between the same
# points -- and no resampling is needed to take it.


def rotated(angle_deg, *events):
    """One block playing `events`, rotated about z."""
    from scipy.spatial.transform import Rotation

    system = pp.Opts()
    sequence = pp.Sequence(system)
    sequence.add_block(
        *events, pp.make_rotation(Rotation.from_euler("z", angle_deg, degrees=True))
    )
    return sequence


def test_rotating_by_nothing_leaves_the_waveform_alone():
    read = pp.make_trapezoid("x", area=1000, duration=1e-3, system=pp.Opts())
    plain = pp.Sequence(pp.Opts())
    plain.add_block(read)

    turned = rotated(0, read)

    assert_same(plain.waveforms()[0], turned.waveforms()[0], "x")
    assert turned.waveforms()[1].shape == (2, 0)


def test_a_quarter_turn_about_z_moves_the_readout_onto_the_other_axis():
    read = pp.make_trapezoid("x", area=1000, duration=1e-3, system=pp.Opts())
    plain = pp.Sequence(pp.Opts())
    plain.add_block(read)

    turned = rotated(90, read)

    # What x played is now on y, and x plays nothing worth keeping.
    assert_same(plain.waveforms()[0][0], turned.waveforms()[1][0], "y times")
    assert turned.waveforms()[1][1] == pytest.approx(plain.waveforms()[0][1])
    assert turned.waveforms()[0].shape == (2, 0)


def test_a_half_turn_about_z_inverts_both_axes_in_the_plane():
    system = pp.Opts()
    read = pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
    encode = pp.make_trapezoid("y", area=500, duration=1e-3, system=system)
    plain = pp.Sequence(system)
    plain.add_block(read, encode)

    turned = rotated(180, read, encode)

    for axis in (0, 1):
        assert turned.waveforms()[axis][1] == pytest.approx(-plain.waveforms()[axis][1])


def test_a_rotation_keeps_the_axis_it_turns_about_alone():
    system = pp.Opts()
    select = pp.make_trapezoid("z", area=1000, duration=1e-3, system=system)
    plain = pp.Sequence(system)
    plain.add_block(select)

    turned = rotated(37, select)

    assert_same(plain.waveforms()[2], turned.waveforms()[2], "z")


def test_rf_times_reports_every_use_when_asked():
    system = pp.Opts()
    sequence = pp.Sequence(system)
    for use in ("excitation", "inversion", "refocusing"):
        sequence.add_block(
            pp.make_block_pulse(0.5, duration=1e-3, system=system, use=use)
        )

    t_excitation, fp_excitation, t_refocusing, fp_refocusing = sequence.rf_times()
    pulses = sequence.rf_times(compat=False)

    # Upstream's four values carry two of the three pulses...
    assert t_excitation.size == 1
    assert t_refocusing.size == 1
    assert fp_excitation.shape == (2, 1)
    assert fp_refocusing.shape == (2, 1)
    # ...and the inversion is only in the other answer.
    assert len(pulses) == 3
    assert pulses.use == ("excitation", "inversion", "refocusing")
    np.testing.assert_allclose(pulses.of("excitation").t, t_excitation)


def test_asking_for_a_use_that_is_not_one_is_refused():
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(
        pp.make_block_pulse(0.5, duration=1e-3, system=pp.Opts(), use="excitation")
    )

    with pytest.raises(ValueError, match="unknown RF use"):
        sequence.rf_times(compat=False).of("recalibration")
