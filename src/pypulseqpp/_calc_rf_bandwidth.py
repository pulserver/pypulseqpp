"""The spectral width of an RF pulse, read off its envelope."""

from __future__ import annotations

__all__ = ["calc_rf_bandwidth"]

import warnings as _warnings
from types import SimpleNamespace as _SimpleNamespace

import numpy as _np
import pypulseq as _pp
from pypulseq import calc_rf_bandwidth as _upstream


def _full_freq_offset(rf) -> float:
    """Where the pulse is tuned, in Hz, with a ppm shift folded in."""
    offset = float(getattr(rf, "freq_offset", 0.0) or 0.0)
    ppm = float(getattr(rf, "freq_ppm", 0.0) or 0.0)
    if abs(ppm) > _np.finfo(float).eps:
        _warnings.warn(
            "calc_rf_bandwidth(): a ppm offset is read against the gamma and "
            "B0 of the default system",
            stacklevel=3,
        )
        system = _pp.Opts.default
        offset += ppm * 1e-6 * system.gamma * system.B0
    return offset


def _at_baseband(rf):
    at_rest = _SimpleNamespace(**vars(rf))
    at_rest.freq_offset = 0.0
    at_rest.freq_ppm = 0.0
    return at_rest


def _with_zero_edges(rf, dt: float):
    """Pad the envelope with zero samples one ``dt`` outside each end.

    This prevents interpolation from extending nonzero endpoint values
    across the FFT window.
    """
    edged = _SimpleNamespace(**vars(rf))
    edged.t = _np.concatenate(([rf.t[0] - dt], rf.t, [rf.t[-1] + dt]))
    edged.signal = _np.concatenate(([0.0], rf.signal, [0.0]))
    return edged


def _crossing(frequency, height, cutoff: float) -> float:
    """Interpolate the first threshold crossing from the supplied edge.

    Includes sidelobes above the cutoff; callers reverse the arrays for the
    opposite edge.
    """
    above = _np.flatnonzero(height >= cutoff * height.max())
    if not above.size:
        return float(frequency[0])
    first = int(above[0])
    if first == 0:
        return float(frequency[0])
    near, far = (
        height[first] - cutoff * height.max(),
        height[first - 1] - cutoff * height.max(),
    )
    return float((near * frequency[first - 1] - far * frequency[first]) / (near - far))


def _width(frequency, spectrum, cutoff: float) -> float:
    """Return the width between outermost threshold crossings, or zero for a zero spectrum."""
    height = _np.abs(_np.asarray(spectrum, dtype=complex))
    frequency = _np.asarray(frequency, dtype=float)
    if not height.size or not (height.max() > 0.0):
        return 0.0
    left = _crossing(frequency, height, cutoff)
    right = _crossing(frequency[::-1], height[::-1], cutoff)
    return float(right - left)


def calc_rf_bandwidth(
    rf,
    cutoff: float = 0.5,
    return_axis: bool = False,
    return_spectrum: bool = False,
    dw: float = 10,
    dt: float | None = None,
):
    """Estimate RF bandwidth from the envelope's Fourier magnitude.

    Uses the outermost crossings at ``cutoff`` times the peak, with linear
    interpolation between bins. This is a small-tip approximation, not a Bloch
    simulation. Frequency offsets shift the returned axis without changing
    the measured width; ppm offsets use the default system's gamma and B0.

    Parameters
    ----------
    rf : SimpleNamespace or RfEvent
        The RF event.
    cutoff : float, optional
        Fraction of the peak the flanks are measured at.
    return_axis : bool, optional
        Also return the frequency axis.
    return_spectrum : bool, optional
        Also return the spectrum.
    dw : float, optional
        Spectral resolution, in Hz.
    dt : float, optional
        Sampling step, in seconds. The default system's RF raster when
        omitted.

    Returns
    -------
    bw : float
        The bandwidth, in Hz. A scalar, where upstream hands back the
        one-element array its flank search indexes with.
    spectrum : numpy.ndarray, optional
        Present when ``return_spectrum``, second.
    w : numpy.ndarray, optional
        Present when ``return_axis``: second on its own, third alongside a
        spectrum. Centred on where the pulse is tuned.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> sinc = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=2e-3, time_bw_product=4)
    >>> abs(pp.calc_rf_bandwidth(sinc) - 4 / 2e-3) < 0.1 * 4 / 2e-3
    True

    A hard pulse is as wide as its duration is short:

    >>> hard = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3)
    >>> abs(pp.calc_rf_bandwidth(hard) - 1.207 / 1e-3) < 0.1 * 1.207 / 1e-3
    True

    And an SLR pulse is as wide as its time-bandwidth product says:

    >>> slr = pp.make_slr_pulse(np.pi / 9, duration=3e-3, time_bw_product=6)
    >>> abs(pp.calc_rf_bandwidth(slr) - 6 / 3e-3) < 0.1 * 6 / 3e-3
    True

    Retuning a pulse moves its band and leaves its width alone:

    >>> sinc.freq_offset = 1500.0
    >>> abs(pp.calc_rf_bandwidth(sinc) - 4 / 2e-3) < 0.1 * 4 / 2e-3
    True

    See Also
    --------
    sim_rf : the simulated profile, valid at any flip angle.
    """
    step = _pp.Opts.default.rf_raster_time if dt is None else dt
    offset = _full_freq_offset(rf)
    measured = _with_zero_edges(_at_baseband(rf) if offset else rf, step)

    # The spectrum is asked for whatever the caller wants, because the flanks
    # are read off it here rather than taken from upstream's bin walk.
    _, spectrum, frequency = _upstream(measured, cutoff, True, True, dw, dt)
    width = _width(frequency, spectrum, cutoff)

    if not (return_axis or return_spectrum):
        return width
    parts = [width]
    if return_spectrum:
        parts.append(spectrum)
    if return_axis:
        parts.append(frequency + offset if offset else frequency)
    return tuple(parts)
