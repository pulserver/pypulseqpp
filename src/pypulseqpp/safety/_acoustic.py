"""Acoustic sound pressure level of a repetition played without end."""

from __future__ import annotations

import os
import warnings
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import scipy.fft

from ._physical import _gamma, _prescription

#: Reference sound pressure of 0 dB, in Pa.
REFERENCE_PRESSURE = 20e-6

#: Peak sound pressure level allowed by IEC 60601-2-33, in dB.
PEAK_LIMIT_DB = 140.0

#: A-weighted average sound pressure level allowed by IEC 60601-2-33, in dB(A).
AVERAGE_LIMIT_DBA = 99.0

_AXIS_DATASETS = (
    "X_AXIS_TRANSFER_FUNCTION",
    "Y_AXIS_TRANSFER_FUNCTION",
    "Z_AXIS_TRANSFER_FUNCTION",
)
_WEIGHTING_DATASET = "A_WEIGHTED_FILTER"

#: Pa per mT/m of a response tabulated per G/cm.
_PER_G_PER_CM = 0.1

#: Largest RMS distance, in dB over 20 Hz to 8 kHz, between a tabulated
#: A-weighting and the IEC 61672 curve on the frequency axis it is read with.
_WEIGHTING_TOLERANCE_DB = 1.0


class AcousticResponse(NamedTuple):
    """A gradient coil's acoustic transfer function, per physical axis.

    ``transfer`` is ``(3, bins)`` complex, the sound pressure in Pa per mT/m of
    gradient on x, y and z, at the frequencies ``k * frequency_step`` Hz for
    ``k = 0 .. bins - 1``.
    """

    transfer: np.ndarray
    frequency_step: float


def read_acoustic_response(
    path: str | os.PathLike, sampling_interval: float
) -> AcousticResponse:
    """Read a gradient coil's acoustic transfer function from an HDF5 file.

    The file holds the response of each physical axis to a gradient in G/cm,
    and an A-weighting filter, each as the ``N`` bins of the discrete Fourier
    transform of a waveform sampled every ``sampling_interval``: two rows,
    real and imaginary part. The bins below the Nyquist frequency are kept.

    Parameters
    ----------
    path : str | os.PathLike
        The HDF5 file to read.
    sampling_interval : float
        Sampling interval (s) the bins refer to; bin ``k`` lies at
        ``k / (N * sampling_interval)`` Hz.

    Returns
    -------
    AcousticResponse
        The response in Pa per mT/m, for
        :func:`~pypulseqpp.safety.check_spl` to take as its ``response``.

    Raises
    ------
    ValueError
        If ``sampling_interval`` is not positive, or the file lacks an axis or
        holds one of another shape than the others.
    OSError
        If the file cannot be read.

    Warns
    -----
    UserWarning
        If the file's A-weighting departs from the IEC 61672 curve on the
        frequency axis ``sampling_interval`` sets, which is then likely not
        the interval the file was tabulated at.
    """
    import h5py

    if not sampling_interval > 0.0:
        raise ValueError("sampling_interval must be positive")
    with h5py.File(path, "r") as held:
        missing = [name for name in _AXIS_DATASETS if name not in held]
        if missing:
            raise ValueError(f"{path} holds no {', '.join(missing)}")
        rows = [np.asarray(held[name], dtype=float) for name in _AXIS_DATASETS]
        weighting = (
            np.asarray(held[_WEIGHTING_DATASET], dtype=float)
            if _WEIGHTING_DATASET in held
            else None
        )
    shapes = {row.shape for row in rows}
    if len(shapes) != 1 or rows[0].ndim != 2 or rows[0].shape[0] != 2:
        raise ValueError(
            f"{path}: each axis must be two rows of one length, got {sorted(shapes)}"
        )
    count = rows[0].shape[1]
    kept = count // 2 + 1
    step = 1.0 / (count * sampling_interval)
    transfer = np.stack([row[0, :kept] + 1j * row[1, :kept] for row in rows])
    if weighting is not None and weighting.shape == rows[0].shape:
        _warn_on_frequency_axis(path, weighting[0] + 1j * weighting[1], step)
    return AcousticResponse(_PER_G_PER_CM * transfer, step)


def _warn_on_frequency_axis(path, weighting: np.ndarray, step: float) -> None:
    frequency = np.arange(1, weighting.size // 2) * step
    audible = (frequency >= 20.0) & (frequency <= 8000.0)
    if not audible.any():
        return
    tabulated = np.abs(weighting[1 : weighting.size // 2][audible])
    curve = a_weighting(frequency[audible])
    distance = np.sqrt(np.mean((20.0 * np.log10(tabulated / curve)) ** 2))
    if distance > _WEIGHTING_TOLERANCE_DB:
        warnings.warn(
            f"{path}: the A-weighting it holds is {distance:.1f} dB RMS from the "
            f"IEC 61672 curve at {step:.4g} Hz per bin; the sampling interval "
            "given is likely not the one the response was tabulated at",
            stacklevel=3,
        )


def a_weighting(frequency) -> np.ndarray:
    """Return the IEC 61672 A-weighting gain at ``frequency`` Hz, 1 at 1 kHz.

    Parameters
    ----------
    frequency : array_like
        Frequencies in Hz.

    Returns
    -------
    NDArray[np.float64]
        The amplitude gain, the shape of ``frequency``.

    Examples
    --------
    >>> from pypulseqpp.safety import a_weighting
    >>> round(float(a_weighting(1000.0)), 3)
    1.0
    """
    f2 = np.square(np.asarray(frequency, dtype=float))
    gain = (
        12194.0**2
        * f2**2
        / (
            (f2 + 20.6**2)
            * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
            * (f2 + 12194.0**2)
        )
    )
    return gain * 10.0 ** (2.0 / 20.0)


def _heaviest_repetition(seq) -> tuple[int, int]:
    """Return the 1-based first and last block of the repetition of most gradient energy."""
    size, _ = seq.repetition()
    events = np.asarray(seq._native.block_events())
    energy = np.concatenate([[0.0], seq.gradient_statistics().energy])
    per_block = energy[events[:, 1]] + energy[events[:, 2]] + energy[events[:, 3]]
    starts = np.arange(0, per_block.size, size)
    totals = np.add.reduceat(per_block, starts)
    first = int(starts[np.argmax(totals)])
    return first + 1, min(first + size, per_block.size)


def _mean_square(spectrum: np.ndarray, count: int) -> float:
    """Return the mean square of ``irfft(spectrum, count)`` from the spectrum itself."""
    power = np.abs(spectrum) ** 2
    total = spectrum[0].real ** 2 + 2.0 * power[1 : (count + 1) // 2].sum()
    if count % 2 == 0:
        total += spectrum[count // 2].real ** 2
    return float(total / count**2)


def check_spl(
    seq,
    response: AcousticResponse,
    *,
    rotation=None,
    peak_limit: float = PEAK_LIMIT_DB,
    average_limit: float = AVERAGE_LIMIT_DBA,
    system=None,
) -> tuple[bool, SimpleNamespace]:
    """Check the sound pressure level of the sequence's loudest repetition, played without end.

    The repetition is the one of :meth:`~pypulseqpp.Sequence.repetition`,
    counted from block 1, over which the squared gradient summed over the
    axes integrates to the most, the earliest on a tie. It is filtered by
    ``response`` as a periodic waveform, so the levels are those of the steady
    state it reaches when played back to back.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    response : AcousticResponse
        The gradient coil's acoustic transfer function, on the physical axes;
        see :func:`read_acoustic_response`.
    rotation : array_like, default=None
        3x3 prescription rotation from logical to physical axes, applied after
        each block's own rotation; identity (axial) by default.
    peak_limit : float, default=140.0
        Largest peak level allowed, in dB.
    average_limit : float, default=99.0
        Largest A-weighted average level allowed, in dB(A).
    system : pypulseqpp.Opts, default=None
        Source of the gyromagnetic ratio; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when neither level exceeds its limit.
    report : SimpleNamespace
        ``peak``, the largest sound pressure in dB; ``average``, the A-weighted
        RMS sound pressure in dB(A); ``average_unweighted``, the RMS sound
        pressure in dB; ``peak_limit`` and ``average_limit``; ``repetition``,
        the 1-based first and last block of the repetition evaluated; and its
        ``duration`` (s). A sequence without gradients reads ``-inf``.

    Raises
    ------
    ValueError
        If ``rotation`` is not a 3x3 orthonormal matrix, or ``response`` is
        not three axes of bins.

    Notes
    -----
    The repetition is sampled at the centres of ``n`` equal intervals, ``n``
    the smallest transform length with only small prime factors that reaches
    the highest frequency ``response`` tabulates. Its
    harmonics ``k / duration`` are weighted by ``response``, linearly
    interpolated between bins and zero above them, and summed over the axes;
    the averages are taken over the harmonics (Parseval), the A-weighted one
    with :func:`a_weighting`. Levels are referred to 20 uPa.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> from pypulseqpp import safety
    >>> seq = pp.Sequence(pp.Opts())
    >>> for _ in range(4):
    ...     _ = seq.add_block(pp.make_trapezoid("x", amplitude=4e5, flat_time=1e-3))
    >>> flat = safety.AcousticResponse(np.full((3, 2049), 0.5 + 0j), 5.0)
    >>> is_ok, report = safety.check_spl(seq, flat)
    >>> is_ok, report.repetition
    (True, (1, 1))
    """
    transfer = np.asarray(response.transfer)
    if transfer.ndim != 2 or transfer.shape[0] != 3 or transfer.shape[1] < 2:
        raise ValueError("response.transfer must be (3, bins) with at least 2 bins")
    turn = _prescription(rotation)
    first, last = _heaviest_repetition(seq) if seq.num_blocks else (0, 0)
    durations = np.asarray(seq._native.block_durations())
    duration = float(durations[first - 1 : last].sum()) if first else 0.0

    silent = SimpleNamespace(
        peak=-np.inf,
        average=-np.inf,
        average_unweighted=-np.inf,
        peak_limit=peak_limit,
        average_limit=average_limit,
        repetition=(first, last),
        duration=duration,
    )
    if duration <= 0.0:
        return True, silent

    top = (transfer.shape[1] - 1) * response.frequency_step
    count = scipy.fft.next_fast_len(
        max(2, int(np.ceil(2.0 * top * duration))), real=True
    )
    times = (np.arange(count) + 0.5) * (duration / count)
    waves = seq.waveforms(block_range=(first, last))
    gradient = np.zeros((3, count))
    for axis in range(3):
        corners = np.asarray(waves[axis])
        if corners.size:
            gradient[axis] = np.interp(
                times, corners[0], corners[1], left=0.0, right=0.0
            )
    gradient = turn @ gradient * (1e3 / _gamma(seq, system))
    if not gradient.any():
        return True, silent

    harmonics = scipy.fft.rfftfreq(count, duration / count)
    bins = np.arange(transfer.shape[1]) * response.frequency_step
    weights = np.stack(
        [
            np.interp(harmonics, bins, axis.real, right=0.0)
            + 1j * np.interp(harmonics, bins, axis.imag, right=0.0)
            for axis in transfer
        ]
    )
    pressure_spectrum = (weights * scipy.fft.rfft(gradient, axis=1)).sum(axis=0)
    pressure = scipy.fft.irfft(pressure_spectrum, count)
    weighted = pressure_spectrum * a_weighting(harmonics)

    def level(value: float) -> float:
        return (
            float(20.0 * np.log10(value / REFERENCE_PRESSURE))
            if value > 0.0
            else -np.inf
        )

    report = SimpleNamespace(
        peak=level(float(np.abs(pressure).max())),
        average=level(np.sqrt(_mean_square(weighted, count))),
        average_unweighted=level(np.sqrt(_mean_square(pressure_spectrum, count))),
        peak_limit=peak_limit,
        average_limit=average_limit,
        repetition=(first, last),
        duration=duration,
    )
    is_ok = report.peak <= peak_limit and report.average <= average_limit
    return bool(is_ok), report
