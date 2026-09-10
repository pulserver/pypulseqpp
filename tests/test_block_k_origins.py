"""Logical k-space origins, RF-centre resets and chunked integration."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext


def system():
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=10e-6,
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
    )


def block_starts(sequence):
    """When each block begins, in seconds from the start of the scan."""
    durations = [sequence.get_block(i + 1).block_duration for i in range(len(sequence))]
    return np.concatenate([[0.0], np.cumsum(durations)])[:-1]


def assert_agrees_with_the_trajectory(sequence):
    """Every origin is where `calculate_kspace` says k is at that moment."""
    origins = _ext.block_k_origins(sequence._native)["origins"]
    # The trajectory with its own time base, which the five-value tuple does
    # not carry.
    reported = sequence._kspace()
    k_traj = np.asarray(reported["k_traj"])
    t_ktraj = np.asarray(reported["t_ktraj"])

    compared = 0
    for index, when in enumerate(block_starts(sequence)):
        at = int(np.searchsorted(t_ktraj, when - 1e-12))
        if at >= t_ktraj.size:
            continue
        expected = k_traj[:, at]
        if not np.all(np.isfinite(expected)):
            continue  # the moment before an excitation belongs to what ended
        np.testing.assert_allclose(
            origins[index], expected, atol=1e-9, err_msg=f"block {index + 1}"
        )
        compared += 1
    assert compared, "nothing was compared"


def test_a_gradient_train_accumulates():
    opts = system()
    sequence = pp.Sequence(opts)
    lobe = pp.make_trapezoid("x", area=1000, duration=1e-3, system=opts)
    for _ in range(4):
        sequence.add_block(lobe)

    origins = _ext.block_k_origins(sequence._native)["origins"]

    np.testing.assert_allclose(origins[:, 0], [0, 1000, 2000, 3000], rtol=1e-9)
    assert_agrees_with_the_trajectory(sequence)


def test_an_excitation_starts_the_trajectory_again():
    opts = system()
    sequence = pp.Sequence(opts)
    lobe = pp.make_trapezoid("x", area=1000, duration=1e-3, system=opts)
    sequence.add_block(lobe)
    sequence.add_block(
        pp.make_block_pulse(0.5, duration=1e-3, system=opts, use="excitation")
    )
    sequence.add_block(lobe)

    origins = _ext.block_k_origins(sequence._native)["origins"]

    # The pulse block sweeps nothing, so k is zero from it and the lobe after
    # it counts from there rather than from 1000.
    np.testing.assert_allclose(origins[:, 0], [0, 1000, 0], rtol=1e-9, atol=1e-9)


def test_a_refocusing_turns_the_trajectory_around():
    opts = system()
    sequence = pp.Sequence(opts)
    lobe = pp.make_trapezoid("x", area=1000, duration=1e-3, system=opts)
    sequence.add_block(
        pp.make_block_pulse(0.5, duration=1e-3, system=opts, use="excitation")
    )
    sequence.add_block(lobe)
    sequence.add_block(
        pp.make_block_pulse(1.0, duration=1e-3, system=opts, use="refocusing")
    )
    sequence.add_block(lobe)

    origins = _ext.block_k_origins(sequence._native)["origins"]

    # 1000 swept, negated by the pulse, then swept back to zero: the echo.
    np.testing.assert_allclose(origins[:, 0], [0, 0, 1000, -1000], rtol=1e-9, atol=1e-9)
    assert_agrees_with_the_trajectory(sequence)


def test_a_pulse_acts_at_its_centre_not_at_its_block():
    """A refocusing between two crushers has each counted on its own side."""
    opts = system()
    sequence = pp.Sequence(opts)
    crusher = pp.make_trapezoid("x", area=500, duration=6e-4, system=opts)
    pulse = pp.make_block_pulse(1.0, duration=1e-3, system=opts, use="refocusing")
    sequence.add_block(
        pp.make_block_pulse(0.5, duration=1e-3, system=opts, use="excitation")
    )
    sequence.add_block(crusher)
    sequence.add_block(
        pp.make_extended_trapezoid(
            "x",
            amplitudes=np.array([0.0, 0.0]),
            times=np.array([0.0, pulse.delay + pulse.shape_dur]),
        ),
        pulse,
    )
    sequence.add_block(crusher)

    origins = _ext.block_k_origins(sequence._native)["origins"]

    # The crusher before is negated with everything else; the one after adds.
    np.testing.assert_allclose(origins[:, 0], [0, 0, 500, -500], rtol=1e-9, atol=1e-9)


def test_the_walk_can_be_taken_in_chunks():
    """Incoming k-space state is supplied by the preceding chunk, not inferred locally."""
    opts = system()
    sequence = pp.Sequence(opts)
    lobe = pp.make_trapezoid("x", area=1000, duration=1e-3, system=opts)
    for _ in range(6):
        sequence.add_block(lobe)

    whole = _ext.block_k_origins(sequence._native)["origins"]

    pieces, carry = [], (0.0, 0.0, 0.0)
    for first in (1, 3, 5):
        found = _ext.block_k_origins(
            sequence._native, first=first, last=first + 1, carry=carry
        )
        pieces.append(found["origins"])
        carry = found["carry"]

    np.testing.assert_allclose(np.concatenate(pieces), whole, rtol=1e-12)


def test_a_rotated_block_is_answered_in_the_logical_frame():
    """`dr . k` does not change when both are turned, so this need not turn."""
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    opts = system()
    lobe = pp.make_trapezoid("x", area=1000, duration=1e-3, system=opts)

    straight = pp.Sequence(opts)
    turned = pp.Sequence(opts)
    for _ in range(3):
        straight.add_block(lobe)
        turned.add_block(lobe, pp.make_rotation(Rotation.from_euler("z", 0.7)))

    np.testing.assert_allclose(
        _ext.block_k_origins(turned._native)["origins"],
        _ext.block_k_origins(straight._native)["origins"],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "name", ["spin_echo", "inversion_recovery_train", "gre_with_label_inc"]
)
def test_the_origins_are_the_trajectory_the_package_already_builds(name):
    """Held against every reference sequence the zoo carries, not a handful."""
    pytest.importorskip(
        "pypulseq_matlab_like",
        reason="the toolbox that builds the reference zoo; see reference.py",
    )
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import convert
    import reference

    if name not in reference.ZOO:
        pytest.skip(f"the zoo has no {name}")
    theirs = reference.ZOO[name]()
    sequence = pp.Sequence(theirs.system)
    sequence._native = convert.to_core(theirs)

    assert_agrees_with_the_trajectory(sequence)
