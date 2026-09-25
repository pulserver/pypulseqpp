"""RF and ADC offsets with their ppm terms resolved at a field strength."""

from __future__ import annotations

__all__ = ["calc_absolute_offsets"]

from ._opts import Opts as _Opts


def _hz_per_ppm(system) -> float:
    """Return ``1e-6 * gamma * B0``: the Larmor frequency in MHz, and in Hz per ppm."""
    return 1e-6 * float(system.gamma) * float(system.B0)


def calc_absolute_offsets(event, *, system=None) -> tuple[float, float]:
    """Return an RF or ADC event's frequency and phase offsets with its ppm offsets resolved.

    Parameters
    ----------
    event : RF or ADC event
        Any object carrying ``freq_offset`` (Hz) and ``phase_offset`` (rad).
        ``freq_ppm`` (ppm) and ``phase_ppm`` (rad/MHz) count as zero where
        absent.
    system : Opts, default=None
        The gamma (Hz/T) and B0 (T) the ppm offsets are resolved at; the
        default system when None.

    Returns
    -------
    freq_offset : float
        ``freq_offset + freq_ppm * f`` in Hz, where ``f = 1e-6 * gamma * B0``
        is the Larmor frequency in MHz, which is also its value in Hz per ppm.
    phase_offset : float
        ``phase_offset + phase_ppm * f`` in rad.

    See Also
    --------
    pypulseqpp.io.SequenceLibraries.absolute_offsets : the same for every RF
        and ADC row of a sequence.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> fat = pp.make_block_pulse(1.0, duration=4e-3, freq_ppm=-3.45)
    >>> frequency, phase = pp.calc_absolute_offsets(fat, system=pp.Opts(B0=3.0))
    >>> round(frequency, 2), phase
    (-440.66, 0.0)
    """
    f = _hz_per_ppm(_Opts.default if system is None else system)
    freq_offset = float(getattr(event, "freq_offset", 0.0) or 0.0)
    phase_offset = float(getattr(event, "phase_offset", 0.0) or 0.0)
    freq_ppm = float(getattr(event, "freq_ppm", 0.0) or 0.0)
    phase_ppm = float(getattr(event, "phase_ppm", 0.0) or 0.0)
    return freq_offset + freq_ppm * f, phase_offset + phase_ppm * f
