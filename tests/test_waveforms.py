"""What a sequence plays, held against the toolbox that defines it.

`waveforms_and_times` is the routine every analysis starts from: it turns a
block table into one waveform per axis over the whole scan, on a time base
shared by all of them, and says when each pulse acts and each sample is
taken. Everything else here reads that -- k-space, the gradient maxima, the
report, a plot -- so what it produces has to be what the toolbox produces,
point for point.

The comparison is the arrays, not a tolerance on some summary of them: a
waveform that agrees in shape and disagrees in a corner is a different
gradient.
"""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert

import pypulseqpp as pp

#: Rotations remap a block's gradients onto other axes, which is a different
#: waveform on each rather than this one moved; not expanded yet.
ROTATED = {"rotated_radial"}


@pytest.fixture
def both(reference_name, build_reference):
    """The same sequence as the toolbox holds it and as the core holds it."""
    if reference_name in ROTATED:
        pytest.skip("a rotated block is not expanded yet")
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
    _, our_excitation, our_refocusing, _, _, _ = ours.waveforms_and_times()

    assert_same(their_excitation, our_excitation, "tfp_excitation")
    assert_same(their_refocusing, our_refocusing, "tfp_refocusing")


def test_when_each_sample_is_taken_is_the_toolboxs(both):
    theirs, ours = both

    *_, their_t, their_fp, their_pm = theirs.waveforms_and_times()
    *_, our_t, our_fp, our_pm = ours.waveforms_and_times()

    assert_same(their_t, our_t, "t_adc")
    assert_same(their_fp, our_fp, "fp_adc")
    assert_same(their_pm, our_pm, "pm_adc")


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


@pytest.mark.parametrize(
    "window",
    [{"blockRange": [2, 5]}, {"blockRange": [1, np.inf]}, {"blockRange": [3, 3]}],
)
def test_a_range_of_blocks_expands_as_the_toolbox_expands_it(both, window):
    theirs, ours = both

    for axis, (expected, got) in enumerate(
        zip(theirs.waveforms(**window), ours.waveforms(**window), strict=True)
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
        pp.Sequence(pp.Opts()).waveforms(blockRange=[1, 2], time_range=[0, 1])


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


def test_a_rotated_block_is_refused_rather_than_expanded_wrongly():
    from scipy.spatial.transform import Rotation

    system = pp.Opts()
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_trapezoid("x", area=1000, duration=1e-3, system=system),
        pp.make_rotation(Rotation.from_euler("z", 30, degrees=True)),
    )

    with pytest.raises(NotImplementedError, match="rotated block"):
        sequence.waveforms()
