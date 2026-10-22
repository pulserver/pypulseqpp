"""Mechanical-resonance check against forbidden gradient bands."""

from __future__ import annotations

import fnmatch
import functools
import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from .. import _ext as _cxx
from ._physical import _gamma, _prescription


class ForbiddenBand(NamedTuple):
    """A forbidden gradient band, a frequency range on one physical axis.

    Attributes
    ----------
    axis : str or None
        ``'x'``, ``'y'``, ``'z'``, or ``None`` for every axis.
    f_min, f_max : float
        Inclusive band edges (Hz).
    tolerance : float
        Largest amplitude allowed inside the band (mT/m); 0 where the table
        states none.
    """

    axis: str | None
    f_min: float
    f_max: float
    tolerance: float


_AXES = {"x": 0, "y": 1, "z": 2, "gx": 0, "gy": 1, "gz": 2}


def read_forbidden_bands(path: str | os.PathLike) -> list[ForbiddenBand]:
    """Read forbidden bands from a Siemens ``.asc`` hardware description.

    The file is read with upstream PyPulseq's ``readasc``. Its acoustic
    resonances are a centre and a bandwidth, with no axis and no tolerance, so
    every band guards every axis with tolerance 0. Bands from any other table
    are passed to :func:`~pypulseqpp.safety.check_mech_resonance` as
    :class:`ForbiddenBand` values.

    Parameters
    ----------
    path : str or os.PathLike
        Siemens ``.asc`` hardware description.

    Returns
    -------
    list of ForbiddenBand
        The bands, for :func:`~pypulseqpp.safety.check_mech_resonance` to
        take as its ``bands``.

    Raises
    ------
    ValueError
        If the file is not a ``.asc`` file.
    """
    path = Path(path)
    if path.suffix.lower() != ".asc":
        raise ValueError(
            f"forbidden bands are read from a .asc hardware description, not from "
            f"{path.name!r}; pass the bands of any other table as ForbiddenBand values"
        )
    return _read_asc(path)


def _read_asc(path: Path) -> list[ForbiddenBand]:
    from pypulseq.utils.siemens.asc_to_hw import asc_to_acoustic_resonances
    from pypulseq.utils.siemens.readasc import readasc

    asc, _ = readasc(str(path))
    bands = []
    for resonance in asc_to_acoustic_resonances(asc):
        centre = float(resonance["frequency"])
        width = float(resonance["bandwidth"])
        if centre <= 0.0 or width <= 0.0:
            continue
        bands.append(
            ForbiddenBand(
                None, max(centre - 0.5 * width, 0.0), centre + 0.5 * width, 0.0
            )
        )
    return bands


@functools.lru_cache(maxsize=1)
def _mkl_runtime() -> str:
    """Path of an installed MKL runtime library, or "" when there is none."""
    patterns = ("libmkl_rt.so*", "libmkl_rt*.dylib", "mkl_rt*.dll")
    found: list[Path] = []
    try:
        dist = importlib.metadata.distribution("mkl")
        found += [
            Path(dist.locate_file(file))
            for file in dist.files or ()
            if any(fnmatch.fnmatch(Path(str(file)).name, p) for p in patterns)
        ]
    except importlib.metadata.PackageNotFoundError:
        pass
    for root in (Path(sys.prefix) / "lib", Path(sys.prefix) / "Library" / "bin"):
        for pattern in patterns:
            found += sorted(root.glob(pattern), reverse=True)
    return next((str(path) for path in found if path.is_file()), "")


def _axis_index(axis) -> int:
    if axis is None or axis == "":
        return -1
    try:
        return _AXES[str(axis).lower()]
    except KeyError:
        raise ValueError(
            f"unknown gradient axis {axis!r}; use 'x', 'y', 'z' or None"
        ) from None


def mech_resonance_spectrum(
    seq,
    *,
    window: int = 0,
    window_width: float = 40e-3,
    stride: float | None = None,
    frequency_oversampling: int = 3,
    rotation=None,
    system=None,
) -> SimpleNamespace:
    """Return the gradient amplitude spectrum of one window, per physical axis.

    The window is sampled, tapered and padded as :func:`check_mech_resonance`
    reads it, so a diagram of a band and the verdict on it read the same
    amplitudes.

    Parameters
    ----------
    seq : Sequence
        Sequence to read.
    window : int, default=0
        Which window, counted from zero. A report names the worst window of
        each band as ``window``.
    window_width : float, default=0.04
        Window length (s). Use the width of the report the window index
        refers to.
    stride : float, default=None
        Step between window starts (s); half the width by default.
    frequency_oversampling : int, default=3
        Zero-padded transform length, in window lengths.
    rotation : array_like, default=None
        3x3 prescription rotation from logical to physical axes, composed
        after each block's own rotation; identity (axial) when ``None``.
    system : pypulseqpp.Opts, default=None
        Source of the gyromagnetic ratio (Hz/T); ``seq.system`` when
        ``None``, then the default :class:`~pypulseqpp.Opts`.

    Returns
    -------
    SimpleNamespace
        ``frequency`` (Hz), one per bin; ``amplitude``, ``(3, bins)`` in mT/m
        for x, y and z; ``axes``, the names in that order; ``window``,
        ``window_start`` (s), ``window_width`` (s) and ``frequency_step`` (Hz).

    Raises
    ------
    ValueError
        If ``window_width`` or ``stride`` is not positive,
        ``frequency_oversampling`` is below 1, ``window`` is negative, or the
        sequence has no window of that index.
    """
    if window_width <= 0.0:
        raise ValueError("window_width must be positive")
    stride = 0.5 * window_width if stride is None else stride
    if stride <= 0.0:
        raise ValueError("stride must be positive")
    if int(frequency_oversampling) < 1:
        raise ValueError("frequency_oversampling must be at least 1")
    if int(window) < 0:
        raise ValueError("window must be at least 0")

    to_hz = 1e-3 * _gamma(seq, system)
    found = _cxx.mech_resonance(
        seq._native,
        [],
        window=float(window_width),
        stride=float(stride),
        oversampling=int(frequency_oversampling),
        rotation=_prescription(rotation).tolist(),
        mkl_runtime=_mkl_runtime(),
        keep_spectrum=int(window),
    )
    if "spectrum" not in found:
        raise ValueError(
            f"the sequence has {found['windows']} windows of "
            f"{found['window'] * 1e3:.1f} ms, so there is no window {int(window)}"
        )
    amplitude = np.asarray(found["spectrum"]) / to_hz
    return SimpleNamespace(
        frequency=np.arange(amplitude.shape[1]) * found["frequency_step"],
        amplitude=amplitude,
        axes=("x", "y", "z"),
        window=int(window),
        window_start=found["spectrum_start"],
        window_width=found["window"],
        frequency_step=found["frequency_step"],
    )


def check_mech_resonance(
    seq,
    bands,
    *,
    window_width: float = 40e-3,
    stride: float | None = None,
    frequency_oversampling: int = 3,
    min_threshold: float = 10.0,
    tolerance_limits: str = "line",
    min_line: float = 6.0,
    reread: bool = True,
    rotation=None,
    system=None,
) -> tuple[bool, SimpleNamespace]:
    """Check the gradient amplitude spectrum against forbidden gradient bands.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    bands : iterable of ForbiddenBand, or path
        Bands as ``(axis, f_min, f_max, tolerance)`` tuples, or the path of a
        ``.asc`` file read by :func:`read_forbidden_bands`.
    window_width : float, default=0.04
        Window length (s).
    stride : float, default=None
        Step between window starts (s); ``window_width / 2`` when ``None``.
    frequency_oversampling : int, default=3
        Zero-padded transform length, in window lengths.
    min_threshold : float, default=10.0
        Threshold (mT/m) for a band whose tolerance is 0.
    tolerance_limits : {"line", "gradient"}, default="line"
        What a positive tolerance limits. ``"line"``: the amplitude of any
        line inside the band. ``"gradient"``: the gradient amplitude on the
        guarded axis, as echo-spacing lockout tables state it; a window then
        violates the band when the band holds a line above ``min_line`` and
        the axis's gradient over the window exceeds the tolerance.
    min_line : float, default=6.0
        Line amplitude (mT/m) a band with ``tolerance_limits="gradient"``
        must hold before its tolerance is compared with the gradient.
    reread : bool, default=True
        Read each window that exceeds a band again over a window four times
        longer, centred on it. The excess is leakage, and no violation, when
        the strongest line there lies outside the band and keeps 80 % of the
        amplitude it read at.
    rotation : array_like, default=None
        3x3 prescription rotation from logical to physical axes, applied after
        each block's own rotation; identity (axial) by default.
    system : pypulseqpp.Opts, default=None
        Source of the gyromagnetic ratio; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when no window exceeds any band's threshold.
    report : SimpleNamespace
        ``window_width``, ``stride`` (as sampled, s), ``frequency_step`` (Hz),
        ``windows``, the FFT ``backend`` :func:`mech_resonance_spectrum`
        transforms with, and ``bands``: one entry per band, in
        the order given, with the band's ``axis`` (None for every axis),
        ``f_min``, ``f_max``, ``tolerance`` and applied ``threshold`` (mT/m);
        the band's worst window, its strongest violating one when any violates
        and its strongest reading otherwise: its ``peak`` amplitude (mT/m), on
        ``peak_axis`` at ``frequency``, in the
        0-based ``window`` starting at ``window_start`` (s), with that axis's
        largest ``gradient`` magnitude over the window (mT/m); ``violations``,
        the windows violating the band on any guarded axis; ``leakage``, the
        exceeding windows the longer read put down to a line outside the
        band, counted per axis; and ``axes``, the same reading for each
        guarded axis on its own.

    Raises
    ------
    ValueError
        If ``min_threshold``, ``min_line``, ``window_width`` or ``stride`` is
        not positive, ``frequency_oversampling`` is below 1,
        ``tolerance_limits`` is neither ``"line"`` nor ``"gradient"``, a
        band's frequency range is empty or negative, or ``rotation`` is not a
        3x3 orthonormal matrix.

    Notes
    -----
    The gradient is sampled at the centres of the sequence's own gradient
    raster, the one its definitions record. Each window is mean-subtracted,
    Hann-tapered and zero-padded before a real FFT, and a bin's amplitude is
    ``2 |X_k| / sum(w)``, so a sustained sinusoid of amplitude A at a bin
    frequency reads A. A band's threshold is its tolerance where that is
    positive and ``min_threshold`` otherwise.

    A wide band is read only up to the frequency the gradient can reach its
    threshold at: a window's reading at f is at most
    ``3 V / (sum(w) sin(pi f dt))``, with V the total variation of the axis's
    gradient over the window and dt the gradient raster. This leaves every
    verdict as it is and bounds the cost of a band open towards the raster's
    Nyquist frequency. A band's ``peak`` is the largest reading of the
    frequencies read.

    The FFT is MKL's when the ``mkl`` package is installed, and pocketfft
    otherwise.
    """
    if isinstance(bands, (str, os.PathLike)):
        bands = read_forbidden_bands(bands)
    bands = [ForbiddenBand(*band) for band in bands]
    if min_threshold <= 0.0:
        raise ValueError("min_threshold must be positive")
    if min_line <= 0.0:
        raise ValueError("min_line must be positive")
    if tolerance_limits not in ("line", "gradient"):
        raise ValueError("tolerance_limits must be 'line' or 'gradient'")
    if window_width <= 0.0:
        raise ValueError("window_width must be positive")
    stride = 0.5 * window_width if stride is None else stride
    if stride <= 0.0:
        raise ValueError("stride must be positive")
    if int(frequency_oversampling) < 1:
        raise ValueError("frequency_oversampling must be at least 1")

    turn = _prescription(rotation)
    to_hz = 1e-3 * _gamma(seq, system)
    on_gradient = tolerance_limits == "gradient"
    thresholds = [
        min_threshold
        if band.tolerance <= 0.0
        else min_line
        if on_gradient
        else band.tolerance
        for band in bands
    ]
    found = _cxx.mech_resonance(
        seq._native,
        [
            (
                _axis_index(band.axis),
                float(band.f_min),
                float(band.f_max),
                threshold * to_hz,
                band.tolerance * to_hz if on_gradient else 0.0,
            )
            for band, threshold in zip(bands, thresholds, strict=True)
        ],
        window=float(window_width),
        stride=float(stride),
        oversampling=int(frequency_oversampling),
        rotation=turn.tolist(),
        mkl_runtime=_mkl_runtime(),
        reread=4 if reread else 1,
    )

    per_axis = [[] for _ in bands]
    for reading in found["readings"]:
        per_axis[reading["band"]].append(
            SimpleNamespace(
                axis="xyz"[reading["axis"]],
                peak=reading["peak"] / to_hz,
                frequency=reading["frequency"],
                window=reading["window"],
                window_start=reading["window_start"],
                gradient=reading["gradient"] / to_hz,
                violations=reading["violations"],
                leakage=reading["leakage"],
            )
        )

    entries = []
    for band, threshold, axes, violations in zip(
        bands, thresholds, per_axis, found["band_violations"], strict=True
    ):
        violated = [reading for reading in axes if reading.violations]
        worst = max(violated or axes, key=lambda reading: reading.peak)
        index = _axis_index(band.axis)
        entries.append(
            SimpleNamespace(
                axis=None if index < 0 else "xyz"[index],
                f_min=band.f_min,
                f_max=band.f_max,
                tolerance=band.tolerance,
                threshold=threshold,
                peak=worst.peak,
                peak_axis=worst.axis,
                frequency=worst.frequency,
                window=worst.window,
                window_start=worst.window_start,
                gradient=worst.gradient,
                violations=violations,
                leakage=sum(reading.leakage for reading in axes),
                axes=axes,
            )
        )
    report = SimpleNamespace(
        window_width=found["window"],
        stride=found["stride"],
        frequency_step=found["frequency_step"],
        windows=found["windows"],
        backend=found["backend"],
        bands=entries,
    )
    return all(entry.violations == 0 for entry in entries), report
