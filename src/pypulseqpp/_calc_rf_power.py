"""Energy, peak power and B1 rms of one RF pulse, in Pulseq's Hz units."""

from __future__ import annotations

import numpy as np

__all__ = ["calc_rf_power"]


def _channels(t: np.ndarray) -> int:
    """Return how many channels share the time base, as the reference interpreter counts them."""
    if t.size == 0:
        return 1
    count = int(np.count_nonzero(t == t[0]))
    if count < 2 or t.size % count:
        return 1
    per_channel = t.size // count
    if not np.array_equal(t[per_channel:], t[:-per_channel]):
        return 1
    return count


def calc_rf_power(rf, dt: float = 1e-6) -> tuple[float, float, float]:
    """Return a pulse's energy, peak power and RMS amplitude, as MATLAB Pulseq's ``calcRfPower``.

    Parameters
    ----------
    rf : RF event
        Anything with ``t`` (s), ``signal`` (Hz) and ``shape_dur`` (s).
    dt : float, default 1e-6
        Resampling step, in seconds.

    Returns
    -------
    total_energy : float
        Integral of ``|rf|^2``, in Hz^2 s.
    peak_pwr : float
        Largest ``|rf|^2``, in Hz^2.
    rf_rms : float
        ``sqrt(total_energy / shape_dur)``, in Hz.

    Notes
    -----
    The pulse is resampled at the midpoints of a ``dt`` grid over its shape
    duration, zero outside its samples. A dynamic pTx pulse is resampled channel
    by channel on its shared time base, and ``|rf|^2`` is the sum of ``|b_c|^2``
    over channels: its root-sum-square amplitude, which channels cannot cancel.
    The quantities are relative: divide ``rf_rms`` by gamma for tesla and
    energy by gamma squared for T^2 s.
    """
    t = np.asarray(rf.t, dtype=float).ravel()
    signal = np.asarray(rf.signal).ravel()
    shape_dur = float(rf.shape_dur)
    samples = int(np.round(shape_dur / dt))
    grid = (np.arange(samples) + 0.5) * dt

    channels = _channels(t)
    per_channel = t.size // channels
    power = np.zeros(samples)
    for channel in range(channels):
        part = slice(channel * per_channel, (channel + 1) * per_channel)
        resampled = np.interp(grid, t[part], signal[part], left=0.0, right=0.0)
        power += np.abs(resampled) ** 2

    total_energy = float(power.sum() * dt)
    peak_pwr = float(power.max()) if samples else 0.0
    rf_rms = float(np.sqrt(total_energy / shape_dur)) if shape_dur > 0.0 else 0.0
    return total_energy, peak_pwr, rf_rms
