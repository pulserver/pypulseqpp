"""ADC dwell and readout duration compatible with the ADC and gradient rasters.

An ADC event's dwell time must be a multiple of the ADC raster, while the
readout gradient it is played under changes only on the gradient raster. These
helpers choose a dwell that satisfies both.

Receiver bandwidth here is the full sampling bandwidth, ``1 / dwell`` in Hz.
Bandwidth per pixel is ``1 / (num_samples * dwell)``.
"""

from __future__ import annotations

__all__ = [
    "calc_adc_timing",
    "ceil_to_raster",
    "quantize_readout_timing",
    "round_to_raster",
]

import math

import numpy as np

#: How far the dwell may be widened past the requested one while looking for a
#: readout duration that lands on the gradient raster. Large enough that no
#: realistic (samples, raster) pair runs out of room before finding one.
MAX_RASTER_SEARCH_STEPS = 50_000


def calc_adc_timing(
    num_samples: int,
    target_dwell: float,
    *,
    grad_raster_time: float,
    adc_raster_time: float,
    min_readout_duration: float = 0.0,
) -> tuple[float, float]:
    """Choose an ADC dwell whose acquisition duration is a multiple of the gradient raster.

    Search upward from the nearest ADC-raster dwell that meets the duration
    floor, and return the first whose acquisition duration is a whole number
    of gradient raster periods. The receiver bandwidth achieved is
    ``1 / dwell``.

    If the bounded search finds no common-raster solution, the returned
    duration may be off the gradient raster; check it before constructing
    events.

    Parameters
    ----------
    num_samples : int
        Number of ADC samples (>= 1).
    target_dwell : float
        Requested dwell time (s), the reciprocal of the requested receiver
        bandwidth.
    grad_raster_time : float
        Gradient raster (s), e.g. ``system.grad_raster_time``.
    adc_raster_time : float
        ADC raster (s), e.g. ``system.adc_raster_time``.
    min_readout_duration : float, optional
        Lower bound on the returned duration (s), e.g. to fit a flat top.

    Returns
    -------
    dwell : float
        Feasible dwell time (s), a multiple of ``adc_raster_time``.
    duration : float
        ``num_samples * dwell`` (s).

    Raises
    ------
    ValueError
        If the sample count, dwell or rasters are not positive, or the
        duration floor is negative.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> dwell, duration = pp.calc_adc_timing(
    ...     96, 3.7e-6, grad_raster_time=10e-6, adc_raster_time=100e-9
    ... )
    >>> round(dwell * 1e6, 4), round(duration * 1e6, 4)
    (5.0, 480.0)
    >>> round(duration / 10e-6, 6)
    48.0

    See Also
    --------
    make_adc : build the ADC event from the returned dwell.
    calc_adc_segments : split a long ADC into equal segments.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if target_dwell <= 0 or grad_raster_time <= 0 or adc_raster_time <= 0:
        raise ValueError("target_dwell and raster times must be > 0")
    if min_readout_duration < 0:
        raise ValueError("min_readout_duration must be >= 0")
    return quantize_readout_timing(
        num_samples,
        1.0 / target_dwell,
        grad_raster_time,
        adc_raster_time,
        min_readout_duration,
    )


def quantize_readout_timing(
    nx_ro: int,
    target_bw_hz_px: float,
    grad_raster_s: float,
    adc_raster_s: float,
    min_flat_time_s: float,
) -> tuple[float, float]:
    """Choose an ADC dwell from a requested receiver bandwidth.

    Receiver bandwidth is the full sampling bandwidth, ``1 / dwell`` in Hz,
    not bandwidth per pixel.

    Search upward from the nearest ADC-raster dwell satisfying the duration
    floor, for at most ``MAX_RASTER_SEARCH_STEPS`` candidates. If none aligns
    the readout to the gradient raster, return that starting dwell.
    Inputs must have positive sample count, bandwidth and rasters.

    Parameters
    ----------
    nx_ro : int
        Number of readout samples.
    target_bw_hz_px : float
        Requested receiver bandwidth in Hz, i.e. ``1 / dwell``. The parameter
        name is kept for compatibility; the value is not a per-pixel
        bandwidth.
    grad_raster_s, adc_raster_s : float
        Gradient and ADC rasters (s).
    min_flat_time_s : float
        Lower bound on the flat time (s), e.g. from the max-gradient limit.

    Returns
    -------
    dwell_s : float
        ADC dwell time (s), a multiple of ``adc_raster_s``.
    flat_time_s : float
        ``nx_ro * dwell_s`` (s); gradient-raster alignment is not guaranteed
        if the search is exhausted.

    Examples
    --------
    >>> from pypulseqpp._timing import quantize_readout_timing
    >>> dwell, flat = quantize_readout_timing(256, 62_500.0, 1e-5, 1e-7, 0.0)
    >>> abs(flat / 1e-5 - round(flat / 1e-5)) < 1e-9
    True
    """
    target_dwell = 1.0 / target_bw_hz_px
    m_target = max(1, round(target_dwell / adc_raster_s))

    m_min = max(m_target, int(np.ceil(min_flat_time_s / (nx_ro * adc_raster_s))))

    # The walk starts at or above the multiple nearest the requested dwell, so
    # every further step moves away from it: the first candidate that lands on
    # the gradient raster is the closest one there is, and nothing after it can
    # win.
    best_m = None

    for step in range(MAX_RASTER_SEARCH_STEPS):
        m = m_min + step
        flat = nx_ro * m * adc_raster_s
        ratio = flat / grad_raster_s
        if abs(ratio - round(ratio)) > 1e-9:
            continue
        best_m = m
        break

    if best_m is None:
        best_m = m_min

    dwell = best_m * adc_raster_s
    return dwell, nx_ro * dwell


def round_to_raster(value_s: float, raster_s: float) -> float:
    """Round seconds to the nearest multiple of ``raster_s``, with ties to even.

    ``raster_s`` is a raster period in seconds, such as
    :attr:`pypulseqpp.Opts.grad_raster_time` or
    :attr:`pypulseqpp.Opts.block_duration_raster`.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.round_to_raster(23e-6, 10e-6)
    2e-05

    A value halfway between two multiples rounds to the even one, so 1.5 and
    2.5 raster periods both round to 2:

    >>> pp.round_to_raster(15e-6, 10e-6), pp.round_to_raster(25e-6, 10e-6)
    (2e-05, 2e-05)
    """
    return round(value_s / raster_s) * raster_s


def ceil_to_raster(value_s: float, raster_s: float) -> float:
    """Round seconds up to the next raster multiple, within a 1e-10 relative tolerance.

    ``raster_s`` is a raster period in seconds. A value already within the
    tolerance of a raster multiple is not pushed to the next one.
    """
    return math.ceil(value_s / raster_s - 1e-10) * raster_s
