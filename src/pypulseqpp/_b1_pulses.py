"""B1-selective pulses and Bloch-Siegert encoding pulses.

The B1-selective designs follow SigPy's ``sigpy.mri.rf.b1sel`` and the
Bloch-Siegert sweep its ``sigpy.mri.rf.adiabatic.bloch_siegert_fm``
(Copyright (c) 2016, Frank Ong and The Regents of the University of
California; BSD 3-Clause, see ``LICENSES/SigPy-BSD-3-Clause.txt``).
"""

from __future__ import annotations

__all__ = [
    "make_b1_gslider_pulse",
    "make_b1_hadamard_pulse",
    "make_b1_selective_pulse",
    "make_bloch_siegert_pulse",
]

import math

import numpy as np
from scipy.linalg import hadamard

from . import _events
from ._opts import default_system
from ._slr import (
    _calc_ripples,
    _check_slab,
    _gslider_beta,
    _hadamard_beta,
    _least_squares,
    design_b1_selective,
)

#: Filter samples per period of the highest frequency a B1-selective sweep
#: carries: enough for the hard-pulse steps of the tilted-frame design to
#: stay small, far fewer than the RF raster would take.
_SAMPLES_PER_PERIOD = 32


def _b1_selective(
    beta_of,
    flip_angle,
    amplitude,
    passband_center,
    passband_width,
    time_bw_product,
    pulse_type,
    passband_ripple,
    stopband_ripple,
    split_and_reflect,
    dwell,
    delay,
    system,
    use,
):
    """Play the B1-selective pulse whose filter ``beta_of(n, d1, d2)`` returns."""
    system = default_system(system)
    dwell = dwell or system.rf_raster_time
    if not 0 < passband_width < 2 * passband_center:
        raise ValueError(
            "passband_width must be positive and the passband must stay above "
            "zero B1: passband_center > passband_width / 2"
        )
    peak = amplitude * system.gamma
    centre_hz, width_hz = passband_center * peak, passband_width * peak
    step = dwell * max(
        1, math.floor(1.0 / (_SAMPLES_PER_PERIOD * (centre_hz + width_hz)) / dwell)
    )
    n = 2 * math.ceil(time_bw_product / width_hz / step / 2)
    _, d1, d2 = _calc_ripples(pulse_type, passband_ripple, stopband_ripple)
    signs, sweep = design_b1_selective(
        flip_angle * beta_of(n, d1, d2),
        step,
        centre_hz,
        split_and_reflect=split_and_reflect,
    )
    total = signs.size * step
    times = (np.arange(round(total / dwell)) + 0.5) * dwell
    sweep = np.interp(times, (np.arange(signs.size) + 0.5) * step, sweep)
    reversed_ = (times < total / 4) | (times >= 3 * total / 4)
    signal = (
        peak
        * np.where(reversed_, -1.0, 1.0)
        * np.exp(2j * np.pi * np.cumsum(sweep) * dwell)
    )
    return _events.make_arbitrary_rf(
        signal=signal,
        flip_angle=flip_angle,
        no_signal_scaling=True,
        dwell=dwell,
        delay=delay,
        system=system,
        use=use,
    )


def make_b1_selective_pulse(
    flip_angle: float,
    amplitude: float,
    passband_center: float = 1.0,
    passband_width: float = 0.2,
    *,
    time_bw_product: float = 4.0,
    pulse_type: str = "st",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    split_and_reflect: bool = True,
    dwell: float = 0.0,
    delay: float = 0.0,
    system=None,
    use: str = "excitation",
):
    """Design a pulse that excites only where B1 lies in a band.

    Constant-magnitude RF whose frequency is swept: in the frame tilted onto
    the RF field, the local B1 plays the part of a gradient and the sweep that
    of the RF, so the selection is along B1 rather than along space (Grissom,
    Cao and Does, J Magn Reson 242:189, 2014). No gradient is played.

    Parameters
    ----------
    amplitude : float
        RF amplitude the pulse plays at, in T: where B1 is nominal.
    passband_center, passband_width : float, optional
        The band of B1 selected, relative to ``amplitude``. The pulse lasts
        about twice ``time_bw_product`` over the width in Hz.
    split_and_reflect : bool, optional
        Split the sweep and reflect it about the pulse's centre, which keeps
        the selectivity at large tip.

    Other parameters are as in :func:`make_slr_pulse`.

    Returns
    -------
    SimpleNamespace
        The pulse.

    Raises
    ------
    ValueError
        If the passband reaches zero B1.
    """
    return _b1_selective(
        lambda n, d1, d2: _least_squares(n, time_bw_product, d1, d2),
        flip_angle,
        amplitude,
        passband_center,
        passband_width,
        time_bw_product,
        pulse_type,
        passband_ripple,
        stopband_ripple,
        split_and_reflect,
        dwell,
        delay,
        system,
        use,
    )


def make_b1_gslider_pulse(
    flip_angle: float,
    amplitude: float,
    num_subslices: int,
    subslice: int,
    *,
    subslice_phase: float = np.pi,
    passband_center: float = 1.0,
    passband_width: float = 0.5,
    time_bw_product: float = 12.0,
    pulse_type: str = "st",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    split_and_reflect: bool = True,
    dwell: float = 0.0,
    delay: float = 0.0,
    system=None,
    use: str = "excitation",
):
    """Design a B1-selective gSlider pulse: one sub-band of B1 at ``subslice_phase``.

    The B1 counterpart of :func:`make_gslider_pulse`: the passband is split
    into ``num_subslices`` sub-bands of B1, counted from the lowest, and
    ``subslice`` is excited at ``subslice_phase`` relative to the rest.

    Parameters
    ----------
    amplitude : float
        RF amplitude the pulse plays at, in T: where B1 is nominal.
    passband_center, passband_width : float, optional
        The band of B1 selected, relative to ``amplitude``. The pulse lasts
        about twice ``time_bw_product`` over the width in Hz.
    split_and_reflect : bool, optional
        Split the sweep and reflect it about the pulse's centre, which keeps
        the selectivity at large tip.

    Other parameters are as in :func:`make_slr_pulse`.

    Returns
    -------
    SimpleNamespace
        The pulse.

    Raises
    ------
    ValueError
        If the passband reaches zero B1, or the sub-bands are narrower than
        their transitions.
    """
    g = int(num_subslices)
    if g < 1 or not 0 <= subslice < g:
        raise ValueError(f"subslice must lie in [0, {g}), got {subslice}")

    def beta_of(n, d1, d2):
        ftw = _check_slab(n, time_bw_product, d1, d2, g)
        # Along B1 the design's band order runs from the lowest B1.
        return _gslider_beta(
            n, time_bw_product, g, subslice, d1, d2, ftw, subslice_phase
        )

    return _b1_selective(
        beta_of,
        flip_angle,
        amplitude,
        passband_center,
        passband_width,
        time_bw_product,
        pulse_type,
        passband_ripple,
        stopband_ripple,
        split_and_reflect,
        dwell,
        delay,
        system,
        use,
    )


def make_b1_hadamard_pulse(
    flip_angle: float,
    amplitude: float,
    order: int,
    row: int,
    *,
    passband_center: float = 1.0,
    passband_width: float = 1.0,
    time_bw_product: float = 16.0,
    pulse_type: str = "st",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    split_and_reflect: bool = True,
    dwell: float = 0.0,
    delay: float = 0.0,
    system=None,
    use: str = "excitation",
):
    """Design a B1-selective pulse whose sub-bands of B1 a Hadamard row signs.

    The B1 counterpart of :func:`make_hadamard_pulse`: sub-bands are counted
    from the lowest B1, and row 0 is the plain band.

    Parameters
    ----------
    amplitude : float
        RF amplitude the pulse plays at, in T: where B1 is nominal.
    passband_center, passband_width : float, optional
        The band of B1 selected, relative to ``amplitude``. The pulse lasts
        about twice ``time_bw_product`` over the width in Hz.
    split_and_reflect : bool, optional
        Split the sweep and reflect it about the pulse's centre, which keeps
        the selectivity at large tip.

    Other parameters are as in :func:`make_slr_pulse`.

    Returns
    -------
    SimpleNamespace
        The pulse.

    Raises
    ------
    ValueError
        If ``order`` is not a power of two, ``row`` is out of range, or the
        passband reaches zero B1.
    """
    if order < 1 or order & (order - 1):
        raise ValueError(f"order must be a power of two, got {order}")
    if not 0 <= row < order:
        raise ValueError(f"row must lie in [0, {order}), got {row}")

    def beta_of(n, d1, d2):
        ftw = _check_slab(n, time_bw_product, d1, d2, order)
        return _hadamard_beta(n, time_bw_product, hadamard(order)[row], d1, d2, ftw)

    return _b1_selective(
        beta_of,
        flip_angle,
        amplitude,
        passband_center,
        passband_width,
        time_bw_product,
        pulse_type,
        passband_ripple,
        stopband_ripple,
        split_and_reflect,
        dwell,
        delay,
        system,
        use,
    )


def make_bloch_siegert_pulse(
    amplitude: float,
    duration: float,
    *,
    k: float = 42.0,
    frequency_sign: int = 1,
    dwell: float = 0.0,
    delay: float = 0.0,
    system=None,
    use: str = "other",
):
    """Design an adiabatic Bloch-Siegert encoding pulse.

    Constant amplitude, swept in a U from far off resonance towards it and
    back, so on-resonant magnetisation is carried along adiabatically and
    left with a phase proportional to the square of its B1 (Khalighi, Rutt
    and Kerr, Magn Reson Med 70:829, 2013). Playing the sweep on either side
    of resonance and differencing the phases maps B1.

    Parameters
    ----------
    amplitude : float
        RF amplitude where B1 is nominal, in T.
    duration : float
        In s.
    k : float, optional
        Sweep shape: ``gamma B1 t / k`` must stay below one over half the
        pulse, and a larger ``k`` keeps the sweep further from resonance.
    frequency_sign : {1, -1}, optional
        Which side of resonance the sweep runs on; the phase changes sign
        with it.

    Returns
    -------
    SimpleNamespace
        The pulse.

    Raises
    ------
    ValueError
        If the sweep would reach resonance within the pulse.
    """
    system = default_system(system)
    dwell = dwell or system.rf_raster_time
    peak = amplitude * system.gamma
    rate = 2.0 * np.pi * peak
    n = 2 * round(duration / dwell / 2)
    t = (np.arange(n // 2) + 0.5) * dwell
    reach = rate / k * t
    if reach[-1] >= 1.0:
        raise ValueError(
            "the sweep reaches resonance within the pulse: raise k, or shorten "
            "the pulse or its amplitude"
        )
    sweep = rate / np.sqrt((1.0 - reach) ** -2 - 1.0)
    sweep = np.concatenate((sweep, sweep[::-1]))
    signal = peak * np.exp(1j * np.sign(frequency_sign) * np.cumsum(sweep) * dwell)
    return _events.make_arbitrary_rf(
        signal=signal,
        flip_angle=0.0,
        no_signal_scaling=True,
        dwell=dwell,
        delay=delay,
        system=system,
        use=use,
    )
