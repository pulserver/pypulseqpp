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

from .. import _ext as _cxx
from ._physical import _gamma, _prescription


class ForbiddenBand(NamedTuple):
    """A frequency range a physical gradient axis must not be driven in.

    ``axis`` is ``"x"``, ``"y"``, ``"z"`` or None for every axis; ``f_min`` and
    ``f_max`` are inclusive edges in Hz; ``tolerance`` is the largest amplitude
    allowed inside the band, in mT/m, and 0 where the table states none.
    """

    axis: str | None
    f_min: float
    f_max: float
    tolerance: float


_AXES = {"x": 0, "y": 1, "z": 2, "gx": 0, "gy": 1, "gz": 2}

#: Largest band count one ESP axis may declare.
_MAX_ESP_PER_AXIS = 10

#: Factor from an ESP row's plateau tolerance to the amplitude the check reads.
#: An alternating trapezoid train reads between 8/pi^2 (triangular) and 4/pi
#: (square) of its plateau; the conversion awaits calibration.
_ESP_TOLERANCE_SCALE = 1.0


def read_forbidden_bands(path: str | os.PathLike) -> list[ForbiddenBand]:
    """Read forbidden bands from a vendor table.

    A ``.asc`` file is a Siemens hardware description, read with upstream
    PyPulseq's ``readasc``; its acoustic resonances are a centre and a
    bandwidth, with no axis and no tolerance, so every band guards every axis
    with tolerance 0.

    Any other file is a GE ``epiesp.dat`` table: for x, y and z in turn, a
    count line and that many ``esp_min_us esp_max_us amplitude_G_per_cm``
    rows; ``#`` starts a comment line. An echo-spacing range maps to the
    frequencies ``1 / (2 ESP)`` of an alternating readout train, and the
    amplitude, the train's plateau, is returned in mT/m times
    ``_ESP_TOLERANCE_SCALE``.

    Raises
    ------
    ValueError
        If an ESP table is malformed. A present but corrupt table is refused
        rather than read as having no bands.
    """
    path = Path(path)
    if path.suffix.lower() == ".asc":
        return _read_asc(path)
    return _read_esp(path)


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


def _read_esp(path: Path) -> list[ForbiddenBand]:
    rows = [
        line
        for line in (raw.strip() for raw in path.read_text().splitlines())
        if line and not line.startswith("#")
    ]
    bands = []
    cursor = 0
    for axis in "xyz":
        if cursor >= len(rows):
            raise ValueError(f"ESP table {path}: truncated before the {axis} axis")
        try:
            count = int(rows[cursor].split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"ESP table {path}: bad band count for {axis}") from exc
        cursor += 1
        if not 0 <= count <= _MAX_ESP_PER_AXIS:
            raise ValueError(
                f"ESP table {path}: implausible band count {count} for {axis}"
            )
        for _ in range(count):
            if cursor >= len(rows):
                raise ValueError(f"ESP table {path}: truncated inside the {axis} axis")
            fields = rows[cursor].split()
            cursor += 1
            try:
                esp_min, esp_max, amplitude = (float(value) for value in fields[:3])
            except ValueError as exc:
                raise ValueError(
                    f"ESP table {path}: expected 'min max amplitude', got {fields!r}"
                ) from exc
            if len(fields) < 3 or esp_min <= 0 or esp_max < esp_min or amplitude < 0:
                raise ValueError(f"ESP table {path}: invalid row {fields!r}")
            tolerance = 10.0 * amplitude * _ESP_TOLERANCE_SCALE
            bands.append(ForbiddenBand(axis, 5e5 / esp_max, 5e5 / esp_min, tolerance))
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


def check_mech_resonance(
    seq,
    bands,
    *,
    window_width: float = 40e-3,
    stride: float | None = None,
    frequency_oversampling: int = 3,
    min_threshold: float = 10.0,
    rotation=None,
    system=None,
) -> tuple[bool, SimpleNamespace]:
    """Return whether no window drives a forbidden band above its threshold.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    bands : iterable of ForbiddenBand, or path
        Bands as ``(axis, f_min, f_max, tolerance)``, or a table for
        :func:`read_forbidden_bands`.
    window_width : float
        Window length in seconds.
    stride : float, optional
        Step between window starts in seconds; ``window_width / 2`` by default.
    frequency_oversampling : int
        Zero-padded transform length, in window lengths.
    min_threshold : float
        Threshold in mT/m for a band whose tolerance is 0.
    rotation : array_like, optional
        3x3 prescription rotation from logical to physical axes, applied after
        each block's own rotation; identity (axial) by default.
    system : pypulseq.Opts, optional
        Source of the gyromagnetic ratio; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when no window exceeds any band's threshold.
    report : SimpleNamespace
        ``window_width``, ``stride`` (as sampled, s), ``frequency_step`` (Hz),
        ``windows``, the FFT ``backend``, and ``bands``: one entry per band, in
        the order given, with the band's ``axis`` (None for every axis),
        ``f_min``, ``f_max``, ``tolerance`` and applied ``threshold`` (mT/m);
        the band's worst window, kept whether or not it violates: its largest
        ``peak`` amplitude (mT/m), on ``peak_axis`` at ``frequency``, in the
        0-based ``window`` starting at ``window_start`` (s); ``violations``,
        the windows exceeding the threshold on any guarded axis; and ``axes``,
        the same reading for each guarded axis on its own.

    Notes
    -----
    The gradient is sampled at the centres of the sequence's own gradient
    raster, the one its definitions record. Each window is mean-subtracted,
    Hann-tapered and zero-padded before a real FFT, and a bin's amplitude is
    ``2 |X_k| / sum(w)``, so a sustained sinusoid of amplitude A at a bin
    frequency reads A. A band's threshold is its tolerance where that is
    positive and ``min_threshold`` otherwise. The FFT is MKL's when the
    ``mkl`` package is installed, and pocketfft otherwise.
    """
    if isinstance(bands, (str, os.PathLike)):
        bands = read_forbidden_bands(bands)
    bands = [ForbiddenBand(*band) for band in bands]
    if min_threshold <= 0.0:
        raise ValueError("min_threshold must be positive")
    if window_width <= 0.0:
        raise ValueError("window_width must be positive")
    stride = 0.5 * window_width if stride is None else stride
    if stride <= 0.0:
        raise ValueError("stride must be positive")
    if int(frequency_oversampling) < 1:
        raise ValueError("frequency_oversampling must be at least 1")

    turn = _prescription(rotation)
    to_hz = 1e-3 * _gamma(seq, system)
    thresholds = [
        band.tolerance if band.tolerance > 0.0 else min_threshold for band in bands
    ]
    found = _cxx.mech_resonance(
        seq._native,
        [
            (
                _axis_index(band.axis),
                float(band.f_min),
                float(band.f_max),
                threshold * to_hz,
            )
            for band, threshold in zip(bands, thresholds, strict=True)
        ],
        window=float(window_width),
        stride=float(stride),
        oversampling=int(frequency_oversampling),
        rotation=turn.tolist(),
        mkl_runtime=_mkl_runtime(),
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
                violations=reading["violations"],
            )
        )

    entries = []
    for band, threshold, axes, violations in zip(
        bands, thresholds, per_axis, found["band_violations"], strict=True
    ):
        worst = max(axes, key=lambda reading: reading.peak)
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
                violations=violations,
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
