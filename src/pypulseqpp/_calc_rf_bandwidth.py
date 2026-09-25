"""RF pulse bandwidth, estimated from the Fourier magnitude of the envelope."""

from __future__ import annotations

__all__ = ["calc_rf_bandwidth"]

import warnings as _warnings
from itertools import pairwise as _pairwise
from types import SimpleNamespace as _SimpleNamespace

import numpy as _np
from pypulseq import calc_rf_bandwidth as _upstream

from ._calc_rf_power import _channels
from ._offsets import calc_absolute_offsets as _calc_absolute_offsets
from ._opts import Opts as _Opts
from ._results import RfBandwidth as _RfBandwidth


def _full_freq_offset(rf) -> float:
    """Return the pulse's total frequency offset in Hz, including the ppm term."""
    ppm = float(getattr(rf, "freq_ppm", 0.0) or 0.0)
    if abs(ppm) <= _np.finfo(float).eps:
        return float(getattr(rf, "freq_offset", 0.0) or 0.0)
    _warnings.warn(
        "calc_rf_bandwidth(): a ppm offset is read against the gamma and "
        "B0 of the default system",
        stacklevel=3,
    )
    return _calc_absolute_offsets(rf, system=_Opts.default)[0]


def _at_baseband(rf):
    at_rest = _SimpleNamespace(**vars(rf))
    at_rest.freq_offset = 0.0
    at_rest.freq_ppm = 0.0
    return at_rest


def _channels_summed(rf):
    """Return a dynamic pTx pulse as the sum of its channels over their shared time base."""
    t = _np.asarray(rf.t, dtype=float).ravel()
    channels = _channels(t)
    if channels == 1:
        return rf
    per_channel = t.size // channels
    summed = _SimpleNamespace(**vars(rf))
    summed.t = t[:per_channel]
    summed.signal = (
        _np.asarray(rf.signal).ravel().reshape(channels, per_channel).sum(axis=0)
    )
    return summed


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
    """Return the width between the outermost threshold crossings; zero for a zero spectrum."""
    height = _np.abs(_np.asarray(spectrum, dtype=complex))
    frequency = _np.asarray(frequency, dtype=float)
    if not height.size or not (height.max() > 0.0):
        return 0.0
    left = _crossing(frequency, height, cutoff)
    right = _crossing(frequency[::-1], height[::-1], cutoff)
    return float(right - left)


def _bands(frequency, spectrum, cutoff: float, band_cutoff: float):
    """Return the centre and width of each run of bins above ``band_cutoff`` of the peak.

    A band's centre is its magnitude-weighted mean frequency. Its width is
    measured at ``cutoff`` of its own peak, over the bins nearer to it than to
    any other band.
    """
    height = _np.abs(_np.asarray(spectrum, dtype=complex))
    frequency = _np.asarray(frequency, dtype=float)
    if not height.size or not (height.max() > 0.0):
        return _np.zeros(0), _np.zeros(0)
    above = _np.concatenate(([False], height >= band_cutoff * height.max(), [False]))
    edges = _np.flatnonzero(_np.diff(above.astype(int)))
    runs = list(zip(edges[::2], edges[1::2], strict=True))
    centres = _np.array(
        [_np.sum(frequency[a:b] * height[a:b]) / _np.sum(height[a:b]) for a, b in runs]
    )
    # Each band owns the bins up to halfway across the gap to its neighbours.
    bounds = [0, *((b + a) // 2 for (_, b), (a, _) in _pairwise(runs)), height.size]
    widths = _np.array(
        [
            _width(frequency[lo:hi], spectrum[lo:hi], cutoff)
            for lo, hi in _pairwise(bounds)
        ]
    )
    return centres, widths


def calc_rf_bandwidth(
    rf,
    cutoff: float = 0.5,
    return_axis: bool = False,
    return_spectrum: bool = False,
    dw: float = 10,
    dt: float | None = None,
    *,
    compat: bool = True,
    band_cutoff: float = 0.3,
):
    """Estimate RF bandwidth from the envelope's Fourier magnitude.

    Uses the outermost crossings at ``cutoff`` times the peak, with linear
    interpolation between bins. This is a small-tip approximation, not a Bloch
    simulation. Frequency offsets shift the returned axis without changing
    the measured width; ppm offsets use the default system's gamma and B0. A
    dynamic pTx pulse is measured as the sum of its channels over their shared
    time base: the pulse a location equally and in-phase sensitive to every
    channel sees, as its flip angle is counted.

    Parameters
    ----------
    rf : SimpleNamespace or RfEvent
        The RF event.
    cutoff : float, default=0.5
        Fraction of the peak the flanks are measured at.
    return_axis : bool, default=False
        Also return the frequency axis.
    return_spectrum : bool, default=False
        Also return the spectrum.
    dw : float, default=10
        Spectral resolution, in Hz.
    dt : float, default=None
        Sampling step, in seconds. The default system's RF raster when
        omitted.
    compat : bool, default=True
        Return upstream's values, which a drop-in caller unpacks. False
        returns an `RfBandwidth` carrying the bandwidth, the spectrum, its
        axis and the bands, whatever ``return_axis`` and ``return_spectrum``
        say.
    band_cutoff : float, default=0.3
        Fraction of the peak above which a run of spectral bins is a band.
        Read only when ``compat`` is False.

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
    RfBandwidth
        In place of the above when ``compat`` is False.

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

    Without ``compat``, the bands of a simultaneous multi-slice pulse are
    reported with their offsets from the carrier:

    >>> base = pp.make_sinc_pulse(flip_angle=np.pi / 6, duration=2e-3, time_bw_product=4)
    >>> sms, offsets, _ = pp.make_sms_pulse(base, 3, 5000.0)
    >>> result = pp.calc_rf_bandwidth(sms, compat=False)
    >>> result.num_bands, np.round(result.band_offsets).astype(int).tolist()
    (3, [-5000, 0, 5000])

    See Also
    --------
    sim_rf : the simulated profile, valid at any flip angle.
    """
    step = _Opts.default.rf_raster_time if dt is None else dt
    offset = _full_freq_offset(rf)
    measured = _with_zero_edges(
        _channels_summed(_at_baseband(rf) if offset else rf), step
    )

    # The spectrum is asked for whatever the caller wants, because the flanks
    # are read off it here rather than taken from upstream's bin walk.
    _, spectrum, frequency = _upstream(measured, cutoff, True, True, dw, dt)
    width = _width(frequency, spectrum, cutoff)

    if not compat:
        # Upstream's grid has an odd number of points, and it labels the bin
        # fftshift puts zero frequency in as -dw; the band centres need each
        # bin's own frequency.
        count = len(spectrum)
        bins = (_np.arange(count) - count // 2) / (count * step)
        offsets, widths = _bands(bins, spectrum, cutoff, band_cutoff)
        return _RfBandwidth(
            bandwidth=width,
            spectrum=spectrum,
            frequency=bins + offset,
            band_offsets=offsets,
            band_bandwidths=widths,
        )
    if not (return_axis or return_spectrum):
        return width
    parts = [width]
    if return_spectrum:
        parts.append(spectrum)
    if return_axis:
        parts.append(frequency + offset if offset else frequency)
    return tuple(parts)
