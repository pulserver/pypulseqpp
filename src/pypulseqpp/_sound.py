"""Stereo audio of the gradient waveforms, as MATLAB Pulseq's ``sound`` computes it."""

from __future__ import annotations

import numpy as np

__all__ = ["gradient_sound"]

#: The audio sample rate MATLAB Pulseq's ``sound`` defaults to, in Hz.
SOUND_SAMPLE_RATE = 44100


def _gaussian_window(sample_rate: float) -> np.ndarray:
    """Return MATLAB's ``gausswin(2 * round(sample_rate / 6000) + 1)``, of unit sum.

    ``gausswin``'s default width parameter is 2.5 and MATLAB's ``round`` rounds
    half away from zero.
    """
    half = int(np.floor(sample_rate / 6000.0 + 0.5))
    if half == 0:
        return np.ones(1)
    n = np.arange(-half, half + 1, dtype=float)
    window = np.exp(-0.5 * (2.5 * n / half) ** 2)
    return window / window.sum()


def _axis(waveform) -> np.ndarray:
    corners = np.asarray(waveform, dtype=float)
    if corners.size == 0:
        return np.zeros((2, 0))
    if corners.ndim != 2 or corners.shape[0] != 2:
        raise ValueError(
            "each waveform must be a (2, n) array of time over amplitude, "
            f"got shape {corners.shape}"
        )
    return corners


def gradient_sound(
    waveforms,
    num_samples: int,
    *,
    first_sample: int = 0,
    channel_weights=(1.0, 1.0, 1.0),
    sample_rate: float = SOUND_SAMPLE_RATE,
    peak: float | None = None,
) -> np.ndarray:
    """Return the stereo audio of the gradient waveforms of the three axes.

    Parameters
    ----------
    waveforms : sequence of array_like
        Three ``(2, n)`` arrays of time (s) over gradient amplitude (Hz/m), for
        the x, y and z axes, as :meth:`Sequence.waveforms` returns them. The
        amplitude is linear between an axis's corners and zero outside them;
        an axis may be empty.
    num_samples : int
        Number of audio samples.
    first_sample : int, default=0
        Index of the first sample. Sample ``k`` is at ``k / sample_rate`` s on
        the waveforms' time axis.
    channel_weights : sequence of float, default=(1.0, 1.0, 1.0)
        Weights of the x, y and z axes.
    sample_rate : float, default=44100
        Audio sample rate in Hz.
    peak : float, default=None
        Magnitude (Hz/m) of the filtered signal that is scaled to 0.95. By
        default the largest magnitude among the samples returned.

    Returns
    -------
    NDArray[np.float64]
        ``(2, num_samples)``. The first channel carries the x axis and the
        second the y axis, each weighted, and both carry half the weighted z
        axis. Both are smoothed with MATLAB's ``gausswin`` window of
        ``2 * round(sample_rate / 6000) + 1`` samples, normalised to unit sum,
        and scaled by ``0.95 / peak``. Samples beyond ``peak`` are not clipped,
        and a signal that is zero throughout is returned as zeros.

    Raises
    ------
    ValueError
        If ``waveforms`` does not hold three axes, ``channel_weights`` three
        weights, ``num_samples`` is negative, or ``sample_rate`` or ``peak`` is
        not positive.

    Notes
    -----
    The window is applied to the waveforms beyond the samples returned, up to
    ``round(sample_rate / 6000)`` samples either side. Calls over consecutive
    sample ranges with one ``peak`` therefore return the samples of one call
    over their union, provided each call's waveforms cover its range widened
    by that margin.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> gx = np.array([[0.0, 1e-3, 2e-3, 3e-3], [0.0, 1e5, 1e5, 0.0]])
    >>> empty = np.zeros((2, 0))
    >>> audio = pp.gradient_sound([gx, empty, empty], 133)
    >>> audio.shape
    (2, 133)
    >>> float(np.abs(audio).max()), float(np.abs(audio[1]).max())
    (0.95, 0.0)
    """
    if len(waveforms) != 3:
        raise ValueError(
            f"waveforms must hold the three gradient axes, got {len(waveforms)}"
        )
    weights = np.asarray(channel_weights, dtype=float)
    if weights.shape != (3,):
        raise ValueError("channel_weights must hold three weights, for x, y and z")
    if not sample_rate > 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if peak is not None and not peak > 0:
        raise ValueError(f"peak must be positive, got {peak}")

    count = int(num_samples)
    if count < 0:
        raise ValueError(f"num_samples must not be negative, got {num_samples}")
    if count == 0:
        return np.zeros((2, 0))

    window = _gaussian_window(sample_rate)
    half = window.size // 2
    index = np.arange(int(first_sample) - half, int(first_sample) + count + half)
    times = index * (1.0 / sample_rate)

    def sampled(axis, weight):
        corners = _axis(waveforms[axis])
        if corners.shape[1] == 0:
            return None
        return np.interp(times, corners[0], corners[1] * weight, left=0.0, right=0.0)

    audio = np.zeros((2, times.size))
    for axis in (0, 1):
        values = sampled(axis, weights[axis])
        if values is not None:
            audio[axis] = values
    shared = sampled(2, 0.5 * weights[2])
    if shared is not None:
        audio += shared

    audio = np.stack(
        [np.convolve(audio[0], window, "valid"), np.convolve(audio[1], window, "valid")]
    )
    scale = float(np.abs(audio).max(initial=0.0)) if peak is None else float(peak)
    if scale > 0.0:
        audio = 0.95 * audio / scale
    return audio
