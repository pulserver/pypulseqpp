"""Dynamic parallel-transmit pulses: one waveform per channel in one RF event."""

from __future__ import annotations

__all__ = ["make_ptx_pulse", "split_ptx_pulse"]

from types import SimpleNamespace

import numpy as np
import pypulseq as _pp

from . import _events
from ._opts import default_system


def make_ptx_pulse(
    signal,
    *,
    delay: float = 0.0,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    center: float | None = None,
    system=None,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
):
    """Make a dynamic pTx pulse from one waveform per transmit channel.

    The channels are stored as Roos et al. store them (Magn Reson Med 2025,
    doi:10.1002/mrm.30601): one after another in a single arbitrary RF event,
    each over the same time base, so the time shape restarts once per channel.
    An interpreter that knows the layout splits the pulse back into channels;
    one that does not still reads a pulse of the right duration.

    Parameters
    ----------
    signal : array_like
        Complex waveforms, ``(num_channels, num_samples)``, in Hz. Played as
        given: with several channels the flip depends on each channel's B1
        map, so nothing is scaled to a flip angle.
    center : float, optional
        Centre of the pulse from its start, in s. Defaults to the middle of
        the peak of the channel-summed magnitude.

    Other parameters are as in :func:`make_arbitrary_rf`.

    Returns
    -------
    SimpleNamespace
        The pulse, lasting ``num_samples * dwell``.

    Raises
    ------
    ValueError
        If ``signal`` is not two-dimensional or holds no samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> rf = pp.make_ptx_pulse(np.full((8, 100), 250.0 + 0j))
    >>> pp.split_ptx_pulse(rf).shape
    (8, 100)
    """
    waveforms = np.asarray(signal, dtype=np.complex128)
    if waveforms.ndim != 2 or waveforms.shape[1] == 0:
        raise ValueError(
            "signal must be (num_channels, num_samples) with at least one sample"
        )
    system = default_system(system)
    dwell = dwell or system.rf_raster_time
    channels, samples = waveforms.shape
    times = (np.arange(samples) + 0.5) * dwell
    if center is None:
        center, _ = _pp.calc_rf_center(
            SimpleNamespace(signal=np.abs(waveforms).sum(axis=0), t=times)
        )

    rf = _pp.make_arbitrary_rf(
        signal=waveforms.reshape(-1),
        flip_angle=0.0,
        no_signal_scaling=True,
        delay=delay,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        system=system,
        use=use,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
        center=float(center),
    )
    # The factory lays the samples on one time base end to end; each channel
    # plays over the same one.
    rf.t = np.tile(times, channels)
    rf.shape_dur = samples * dwell
    return _events.convert(rf)


def split_ptx_pulse(rf) -> np.ndarray:
    """Return a pulse's waveforms, one row per transmit channel, in Hz.

    The channel count is the number of samples at the pulse's first sample
    time, as the reference interpreter reads it; a single-channel pulse comes
    back as one row.

    Raises
    ------
    ValueError
        If the time shape is not one time base repeated once per channel.
    """
    times = np.asarray(rf.t, dtype=float)
    signal = np.asarray(rf.signal)
    channels = int(np.count_nonzero(times == times[0]))
    per_channel = times.size // channels
    if (
        per_channel * channels != times.size
        or signal.size != times.size
        or not np.array_equal(
            times.reshape(channels, per_channel),
            np.tile(times[:per_channel], (channels, 1)),
        )
    ):
        raise ValueError(
            "this pulse's time shape is not one time base repeated per channel"
        )
    return signal.reshape(channels, per_channel)
