import numpy as np
import pytest

from pypulseqpp import sequences


def balanced_transient(flip_deg, tr, t1, t2, repetitions):
    """|Mxy| after each on-resonance bSSFP excitation following an alpha/2 catalyst."""

    def rotation(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def relaxed(m, t):
        e1, e2 = np.exp(-t / t1), np.exp(-t / t2)
        return np.array([m[0] * e2, m[1] * e2, 1 + (m[2] - 1) * e1])

    flip = np.radians(flip_deg)
    m = relaxed(rotation(flip / 2) @ np.array([0.0, 0.0, 1.0]), tr / 2)
    signal = []
    for n in range(repetitions):
        m = rotation(flip * (-1) ** (n + 1)) @ m
        signal.append(np.hypot(m[0], m[1]))
        m = relaxed(m, tr)
    return np.asarray(signal)


@pytest.mark.parametrize("flip_deg", [30.0, 60.0, 90.0])
def test_balanced_dummies_bring_the_bssfp_transient_within_the_tolerance(flip_deg):
    tr, t1, t2 = 3.5e-3, 1.5, 0.1
    n = sequences.steady_state_dummies(tr, flip_deg, t1_s=t1, t2_s=t2)
    signal = balanced_transient(flip_deg, tr, t1, t2, 20 * n)
    distance = np.abs(signal - signal[-1]) / np.abs(signal[0] - signal[-1])

    assert distance[n] <= 0.011
    assert distance[n // 2] > 0.011


def test_a_balanced_steady_state_takes_more_dummies_than_a_spoiled_one():
    spoiled = sequences.steady_state_dummies(3.5e-3, 60.0)
    balanced = sequences.steady_state_dummies(3.5e-3, 60.0, t2_s=0.1)

    assert balanced > 10 * spoiled
