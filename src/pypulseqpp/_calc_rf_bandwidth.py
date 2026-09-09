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
    """``rf`` tuned to zero, which is where its width is measured."""
    at_rest = _SimpleNamespace(**vars(rf))
    at_rest.freq_offset = 0.0
    at_rest.freq_ppm = 0.0
    return at_rest


def _with_zero_edges(rf, dt: float):
    """``rf`` with one zero sample placed a raster step outside each end.

    A pulse is resampled onto the spectrum's time grid by interpolation, and
    that grid spans ``1 / dw`` -- tens of milliseconds, against a pulse of a
    few. Interpolation holds the end samples across everything beyond them,
    so an envelope that does not itself reach zero is read as one that never
    stops, and the held tail is a rectangle tens of times longer than the
    pulse. Its transform is a spike at the centre frequency, and a spike
    above the passband moves the half height the flanks are found at.

    A hard pulse is the extreme: two samples, one at each end of a
    rectangle, held on both sides into a rectangle without end, whose
    transform is a delta and whose width is therefore zero. An SLR pulse is
    the ordinary case, its ends a percent of its peak and its passband read
    a third narrow. The zero samples give the interpolation somewhere to
    land, which is where the pulse actually ends.
    """
    edged = _SimpleNamespace(**vars(rf))
    edged.t = _np.concatenate(([rf.t[0] - dt], rf.t, [rf.t[-1] + dt]))
    edged.signal = _np.concatenate(([0.0], rf.signal, [0.0]))
    return edged


def _crossing(frequency, height, cutoff: float) -> float:
    """Where the flank crosses ``cutoff``, between the two bins that bracket it.

    Walked inward from the edge, so a side lobe over the threshold cannot
    narrow the answer, and interpolated rather than rounded to a bin: a flank
    is a smooth function, and taking the first bin above the cutoff makes the
    answer's precision the bin width. That is what would otherwise force a
    fine ``dw`` -- and the transform is the whole cost of asking.
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
    """Distance between the two flanks of the main lobe."""
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
    """Spectral width of an RF pulse, from an FFT of its envelope.

    A low-flip-angle approximation: the excitation profile is taken to be the
    Fourier transform of the pulse, and the bandwidth the width of that
    transform's main lobe at ``cutoff`` of its height.

    PyPulseq's function, called with a pulse it can answer. Two things are
    put right first. The pulse is given the zero it steps down to on either
    side of it, because the resampling before the transform otherwise holds
    its end samples out to the edge of the window -- which reads a hard pulse
    as a rectangle without end and answers zero for it, and reads an SLR
    pulse's one-percent ends as a spike above its own passband. And a pulse
    carrying a frequency offset is measured at baseband, the offset going
    back onto the frequency axis afterwards, because retuning a pulse moves
    its band without widening it.

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
