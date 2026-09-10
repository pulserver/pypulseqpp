"""K-space integration parity and invariance under equivalent waveform layouts."""

import math

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert

import pypulseqpp as pp

#: What the toolbox's `calculate_kspacePP` reports, by position, against the
#: name this package reports it under.
REPORTED = [
    ("k_traj_adc", 0),
    ("t_adc", 1),
    ("k_traj", 2),
    ("t_ktraj", 3),
    ("t_excitation", 4),
    ("t_refocusing", 5),
    ("slicepos", 6),
    ("t_slicepos", 7),
    ("pm_adc", 9),
]


@pytest.fixture
def both(reference_name, build_reference):
    """The same sequence as the toolbox holds it and as the core holds it."""
    theirs = build_reference()
    ours = pp.Sequence(theirs.system)
    ours._native = convert.to_core(theirs)
    return theirs, ours


#: How close two moments have to be before they are the same moment.
COINCIDENT = 1e-11

#: How far the toolbox's trajectory may sit from this one, in 1/m.
#:
#: It holds an axis at zero in front of what it plays by putting a knot a
#: picosecond ahead of the axis's own first corner. Its rule for merging
#: coincident knots works to a nanosecond, so the corner behind the pad is
#: lost to it and the ramp behind *that* is a picosecond longer than the
#: sequence asked for -- an area of amplitude times half a picosecond, every
#: time an axis starts anywhere but the beginning.
#:
#: Four parts in a hundred million of a phase encode, and it does not
#: accumulate, which is why it went unnoticed. It is not nothing: a phase
#: encode and its rewinder no longer cancel, and a sequence whose repetitions
#: are identical no longer reads as though they are. Where the two differ, it
#: is this package that returns k to zero -- `test_a_balanced_pair_returns_k_
#: to_where_it_found_it` is the same question asked without the toolbox.
LEAK = 1e-5


def assert_same(expected, got, where, atol=1e-7):
    expected = np.asarray(expected, dtype=float)
    got = np.asarray(got, dtype=float)
    assert expected.shape == got.shape, f"{where}: shape"
    if expected.size:
        assert np.allclose(expected, got, rtol=1e-7, atol=atol, equal_nan=True), (
            f"{where}: values"
        )


def assert_same_curve(their_t, their_k, our_t, our_k, where):
    """Compare trajectories at shared moments; the reference may add padding-induced knots."""
    their_t = np.asarray(their_t, dtype=float)
    our_t = np.asarray(our_t, dtype=float)
    if their_t.size == 0:
        assert our_t.size == 0, f"{where}: this reports moments the toolbox does not"
        return

    their_k = np.atleast_2d(np.asarray(their_k, dtype=float))
    our_k = np.atleast_2d(np.asarray(our_k, dtype=float))

    # Every moment of ours is one of theirs: a moment they do not report would
    # be a real difference rather than scaffolding of their own.
    after = np.searchsorted(their_t, our_t).clip(0, their_t.size - 1)
    before = (after - 1).clip(0, their_t.size - 1)
    theirs_at = np.where(
        np.abs(their_t[after] - our_t) <= np.abs(their_t[before] - our_t), after, before
    )
    assert np.abs(their_t[theirs_at] - our_t).max() <= COINCIDENT, (
        f"{where}: this reports a moment the toolbox does not"
    )

    # And where both report a moment, both say the same thing about it, to
    # within what the toolbox's own padding costs it.
    for axis in range(their_k.shape[0]):
        assert_same(
            their_k[axis][theirs_at], our_k[axis], f"{where}: axis {axis}", atol=LEAK
        )


def test_the_trajectory_is_the_toolboxs(both):
    theirs, ours = both

    reported = theirs.calculate_kspacePP()
    found = ours._kspace()

    for name, position in REPORTED:
        if name == "k_traj":
            assert_same_curve(
                reported[3], reported[2], found["t_ktraj"], found["k_traj"], name
            )
        elif name == "k_traj_adc":
            assert_same(reported[position], found[name], name, atol=LEAK)
        elif name != "t_ktraj":
            assert_same(reported[position], found[name], name)


def test_the_gradients_it_integrated_are_the_toolboxs(both):
    theirs, ours = both

    reported = theirs.calculate_kspacePP()[8]
    found = ours._kspace()["gw_pp"]

    for axis, (expected, got) in enumerate(zip(reported, found, strict=True)):
        assert (expected is None) == (got is None), f"axis {axis}: presence"
        if expected is None:
            continue
        assert_same(expected.x, got.x, f"axis {axis}: knots")


def test_a_delayed_trajectory_is_the_toolboxs(both):
    """A gradient that plays late puts the samples somewhere else."""
    theirs, ours = both

    reported = theirs.calculate_kspacePP(trajectory_delay=1e-5)
    found = ours._kspace(trajectory_delay=1e-5)

    assert_same(reported[0], found["k_traj_adc"], "k_traj_adc")


def test_a_background_gradient_is_the_toolboxs(both):
    theirs, ours = both

    reported = theirs.calculate_kspacePP(gradient_offset=50.0)
    found = ours._kspace(gradient_offset=50.0)

    assert_same(reported[0], found["k_traj_adc"], "k_traj_adc")


# -- what the five values are ----------------------------------------------


def test_the_five_values_are_upstreams(both):
    """`calculate_kspace` reports what upstream reports, in that order."""
    _, ours = both

    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = ours.calculate_kspace()
    everything = ours._kspace()

    assert_same(everything["k_traj_adc"], k_traj_adc, "k_traj_adc")
    assert_same(everything["k_traj"], k_traj, "k_traj")
    assert_same(everything["t_excitation"], t_excitation, "t_excitation")
    assert_same(everything["t_refocusing"], t_refocusing, "t_refocusing")
    assert_same(everything["t_adc"], t_adc, "t_adc")


def test_the_pp_spelling_is_the_same_calculation(both):
    _, ours = both

    assert_same(ours.calculate_kspace()[0], ours.calculate_kspacePP()[0], "k_traj_adc")


def test_there_is_one_sample_position_per_sample(both):
    theirs, ours = both

    k_traj_adc, *_, t_adc = ours.calculate_kspace()

    assert k_traj_adc.shape == (3, len(t_adc))
    assert len(t_adc) == len(theirs.adc_times()[0])


# -- what a trajectory does ------------------------------------------------


def gradient_echo(lines=4, area=1000.0):
    """A readout that sweeps k, with the phase encode stepped."""
    system = pp.Opts()
    sequence = pp.Sequence(system)
    pulse = pp.make_block_pulse(
        math.pi / 8, duration=1e-3, system=system, use="excitation"
    )
    read = pp.make_trapezoid("x", area=area, duration=2e-3, system=system)
    window = pp.make_adc(
        num_samples=32, duration=1.8e-3, delay=read.rise_time, system=system
    )
    encode = pp.make_trapezoid("y", area=area / 2, duration=1e-3, system=system)
    for line in range(lines):
        sequence.add_block(pulse)
        sequence.add_block(pp.scale_grad(encode, (line / lines) or 1e-9))
        sequence.add_block(read, window)
    return sequence


def test_an_excitation_puts_the_trajectory_back_at_the_origin():
    """Nothing is remembered across a pulse: that is what excitation means."""
    sequence = gradient_echo(lines=3)

    _, k_traj, t_excitation, _, _ = sequence.calculate_kspace()
    everything = sequence._kspace()
    at_pulse = [
        int(np.argmin(np.abs(everything["t_ktraj"] - when))) for when in t_excitation
    ]

    for index in at_pulse:
        assert k_traj[:, index] == pytest.approx(np.zeros(3), abs=1e-6)


def test_a_readout_sweeps_k_from_where_the_prewinder_left_it():
    """The prewinder is what puts the centre of k-space in the window."""
    system = pp.Opts()
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_block_pulse(math.pi / 8, duration=1e-3, system=system, use="excitation")
    )
    read = pp.make_trapezoid("x", area=1000.0, duration=2e-3, system=system)
    sequence.add_block(
        pp.make_trapezoid("x", area=-500.0, duration=1e-3, system=system)
    )
    sequence.add_block(
        read,
        pp.make_adc(
            num_samples=32, duration=1.8e-3, delay=read.rise_time, system=system
        ),
    )

    k_traj_adc, *_ = sequence.calculate_kspace()

    along = k_traj_adc[0]
    assert np.all(np.diff(along) > 0), "a readout sweeps one way"
    assert along[0] < 0 < along[-1], "and through the middle"
    # It ends where the two areas leave it, give or take the ramps the window
    # does not cover.
    assert along[-1] == pytest.approx(500.0, rel=0.05)


def test_a_readout_without_a_prewinder_starts_at_the_origin():
    sequence = gradient_echo(lines=1)

    k_traj_adc, *_ = sequence.calculate_kspace()

    along = k_traj_adc[0]
    assert np.all(np.diff(along) > 0)
    assert along[0] == pytest.approx(0.0, abs=0.1 * along[-1])
    assert along[-1] == pytest.approx(1000.0, rel=1e-6)


def test_a_stepped_phase_encode_moves_the_line_it_reads():
    """Each shot reads a different line, which is what an encode is for."""
    sequence = gradient_echo(lines=4)

    k_traj_adc, *_ = sequence.calculate_kspace()
    per_shot = k_traj_adc[1].reshape(4, -1)

    lines = per_shot[:, 0]
    assert np.all(np.diff(lines) > 0), "the lines should march in one direction"
    assert len(set(np.round(lines, 6))) == 4


def test_a_refocusing_turns_the_trajectory_around():
    """A spin echo comes back to the origin, which is why it is an echo."""
    system = pp.Opts()
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system, use="excitation")
    )
    sequence.add_block(pp.make_trapezoid("x", area=500, duration=1e-3, system=system))
    sequence.add_block(
        pp.make_block_pulse(math.pi, duration=1e-3, system=system, use="refocusing")
    )
    sequence.add_block(
        pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
        pp.make_adc(num_samples=32, duration=2e-3, system=system),
    )

    k_traj_adc, *_ = sequence.calculate_kspace()

    # The dephasing before the pulse is undone by the readout after it, so
    # the trajectory crosses the origin partway through the window.
    along = k_traj_adc[0]
    assert along[0] < 0 < along[-1]


# -- where the samples are, without the trajectory between them ------------


def test_the_samples_are_the_same_without_the_trajectory(both):
    """A moment reported between two corners changes nothing about a sample.

    The trajectory is reported at every corner and through every ramp at the
    raster so that drawing it draws the gradient. None of that is part of
    where a sample sits: between two corners the gradient is a straight line
    and its integral a parabola, so the answer either side is exact.
    """
    _, ours = both

    whole = ours._kspace()
    quick = ours._kspace(samples_only=True)

    assert_same(whole["k_traj_adc"], quick["k_traj_adc"], "k_traj_adc")
    assert_same(whole["t_adc"], quick["t_adc"], "t_adc")
    assert_same(whole["pm_adc"], quick["pm_adc"], "pm_adc")


def test_asking_only_for_the_samples_leaves_the_trajectory_unbuilt(both):
    _, ours = both

    quick = ours._kspace(samples_only=True)

    assert quick["t_ktraj"].size == 0
    assert quick["k_traj"].shape[1] == 0


def test_the_samples_alone_still_start_over_at_each_excitation():
    """The pulses are what a sample's position is measured from."""
    sequence = gradient_echo(lines=3)

    whole = sequence._kspace()
    quick = sequence._kspace(samples_only=True)

    assert_same(whole["k_traj_adc"], quick["k_traj_adc"], "k_traj_adc")


def test_the_samples_alone_still_turn_around_at_a_refocusing():
    system = pp.Opts()
    sequence = pp.Sequence(system)
    for _ in range(2):
        sequence.add_block(
            pp.make_block_pulse(
                math.pi / 2, duration=1e-3, system=system, use="excitation"
            )
        )
        sequence.add_block(
            pp.make_trapezoid("x", area=500, duration=1e-3, system=system)
        )
        sequence.add_block(
            pp.make_block_pulse(math.pi, duration=1e-3, system=system, use="refocusing")
        )
        sequence.add_block(
            pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
            pp.make_adc(num_samples=16, duration=2e-3, system=system),
        )

    whole = sequence._kspace()
    quick = sequence._kspace(samples_only=True)

    assert_same(whole["k_traj_adc"], quick["k_traj_adc"], "k_traj_adc")
    # And it really is an echo: the readout crosses the origin.
    along = quick["k_traj_adc"][0]
    assert along.min() < 0 < along.max()


# -- the edges of an arbitrary gradient ------------------------------------


@pytest.mark.parametrize(
    ("label", "waveform"),
    [
        ("starts high", 8e4 * np.cos(np.linspace(0, np.pi / 2, 60))),
        ("ends high", 8e4 * np.sin(np.linspace(0, np.pi / 2, 60))),
        ("both ends high", 8e4 * (0.5 + 0.5 * np.cos(np.linspace(0, 4 * np.pi, 60)))),
        ("constant, edges at zero", np.full(40, 1e4)),
    ],
)
def test_the_trajectory_is_the_integral_of_the_waveform_that_is_drawn(label, waveform):
    """A shape's `first` and `last` are part of what it draws, and of what it
    encodes.

    An arbitrary gradient kept at raster centres says nothing about the
    interval boundaries; the recorded first and last values close it, and the
    waveform an interpreter draws runs between all of them. So the trajectory
    has to be the integral of *that*, not of the samples -- which is a
    difference for anything that does not start and end at zero, a corkscrew
    or a spiral arm among them.
    """
    system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    sequence = pp.Sequence(system)
    sequence.add_block(
        pp.make_arbitrary_grad(
            "x", waveform=waveform, first=waveform[0], last=waveform[-1], system=system
        )
    )

    drawn = sequence.waveforms()[0]
    enclosed = np.concatenate(
        [[0.0], np.cumsum(np.diff(drawn[0]) * 0.5 * (drawn[1][1:] + drawn[1][:-1]))]
    )
    trajectory = sequence.calculate_kspace()[1]

    assert trajectory[0, -1] == pytest.approx(enclosed[-1], rel=1e-12, abs=1e-12)


# -- what an axis encloses does not depend on when it starts ---------------


@pytest.mark.parametrize("delay", [0.0, 1e-5, 1e-3, 1e-2])
def test_a_gradient_encloses_its_area_wherever_it_starts(delay):
    """A trapezoid played late encloses what a trapezoid played early does.

    An axis that starts anywhere but the beginning has to be held at zero in
    front of what it plays, and holding it there must not lengthen the ramp it
    is held in front of. A picosecond of ramp at full amplitude is a real area,
    and it is the same area however long the wait before it was.
    """
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    amplitude = 372960.372960373
    lobe = pp.make_trapezoid(
        "y",
        amplitude=amplitude,
        rise_time=2.2e-4,
        flat_time=5.6e-4,
        fall_time=2.2e-4,
        system=system,
    )
    enclosed = amplitude * (2.2e-4 / 2 + 5.6e-4 + 2.2e-4 / 2)

    sequence = pp.Sequence(system)
    if delay:
        sequence.add_block(pp.make_delay(delay))
    sequence.add_block(lobe)

    assert float(sequence.calculate_kspace()[1][1, -1]) == pytest.approx(
        enclosed, abs=1e-9
    )


def test_a_balanced_pair_returns_k_to_where_it_found_it():
    """Which is what lets a phase encode be undone by its own rewinder."""
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    amplitude = 372960.372960373
    shape = {
        "rise_time": 2.2e-4,
        "flat_time": 5.6e-4,
        "fall_time": 2.2e-4,
        "system": system,
    }

    sequence = pp.Sequence(system)
    for _ in range(8):
        sequence.add_block(pp.make_delay(2e-4))
        sequence.add_block(pp.make_trapezoid("y", amplitude=amplitude, **shape))
        sequence.add_block(pp.make_delay(2e-4))
        sequence.add_block(pp.make_trapezoid("y", amplitude=-amplitude, **shape))

    assert float(sequence.calculate_kspace()[1][1, -1]) == pytest.approx(0.0, abs=1e-9)
