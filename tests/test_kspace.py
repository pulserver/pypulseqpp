"""Where the sequence goes in k-space, held against the toolbox.

The trajectory is the integral of the gradient waveforms, reset at every
excitation and turned around at every refocusing. It is what the report, the
FOV transform and any reconstruction read, so what matters is not that it
looks right but that it is the same trajectory the toolbox follows -- every
sample, every corner.
"""

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


def assert_same(expected, got, where):
    expected = np.asarray(expected, dtype=float)
    got = np.asarray(got, dtype=float)
    assert expected.shape == got.shape, f"{where}: shape"
    if expected.size:
        assert np.allclose(expected, got, rtol=1e-7, atol=1e-7, equal_nan=True), (
            f"{where}: values"
        )


def test_the_trajectory_is_the_toolboxs(both):
    theirs, ours = both

    reported = theirs.calculate_kspacePP()
    found = ours._kspace()

    for name, position in REPORTED:
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
