"""Peripheral nerve stimulation check under the SAFE or the chronaxie nerve model."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import SimpleNamespace
from typing import NamedTuple

from .. import _ext as _cxx
from ._physical import _gamma, _prescription


class ChronaxieModel(NamedTuple):
    """Rheobase-chronaxie nerve model, one coefficient set for every physical axis.

    ``chronaxie`` is in s and ``rheobase`` in T/m/s. The response is
    normalised by ``rheobase / alpha``: a rectangular slew S held for tau
    reaches ``S alpha tau / (rheobase (chronaxie + tau))``.
    """

    chronaxie: float
    rheobase: float
    alpha: float = 1.0


_SAFE_FIELDS = ("a1", "a2", "a3", "tau1", "tau2", "tau3", "stim_limit", "g_scale")


def read_safe_model(path: str | os.PathLike) -> SimpleNamespace:
    """Read a SAFE nerve model from a Siemens ``.asc`` hardware description.

    Reading is upstream PyPulseq's ``readasc`` and ``asc_to_hw``; the result
    is shaped like ``pypulseq.utils.safe_pns_prediction.safe_example_hw()``:
    ``x``, ``y`` and ``z``, each with ``a1``-``a3``, ``tau1``-``tau3`` (ms),
    ``stim_limit`` (T/m/s) and ``g_scale``.
    """
    from pypulseq.utils.siemens.asc_to_hw import asc_to_hw
    from pypulseq.utils.siemens.readasc import readasc

    asc, _ = readasc(str(path))
    return asc_to_hw(asc)


def _coefficients(model):
    """Return (kind, per-axis SAFE tuples, chronaxie tuple) for the binding."""
    if isinstance(model, (str, os.PathLike)):
        model = read_safe_model(model)
    if isinstance(model, Mapping):
        model = ChronaxieModel(**model)
    if isinstance(model, ChronaxieModel):
        coefficients = tuple(float(value) for value in model)
        if min(coefficients) <= 0.0:
            raise ValueError("chronaxie, rheobase and alpha must be positive")
        return "chronaxie", [], coefficients

    axes = [getattr(model, axis, None) for axis in "xyz"]
    if any(axis is None for axis in axes):
        raise TypeError(
            "a PNS model is a ChronaxieModel, a SAFE description shaped like "
            "safe_example_hw(), or the path of a .asc file"
        )
    safe = []
    for name, axis in zip("xyz", axes, strict=True):
        missing = [
            field for field in _SAFE_FIELDS if getattr(axis, field, None) is None
        ]
        if missing:
            raise ValueError(f"SAFE axis {name} is missing {', '.join(missing)}")
        values = tuple(float(getattr(axis, field)) for field in _SAFE_FIELDS)
        if abs(sum(values[:3]) - 1.0) > 1e-3:
            raise ValueError(f"SAFE axis {name}: a1 + a2 + a3 must be 1")
        if values[6] <= 0.0:
            raise ValueError(f"SAFE axis {name}: stim_limit must be positive")
        safe.append(values)
    return "safe", safe, (0.0, 0.0, 1.0)


def _pns(seq, model, rotation, system, keep_trace):
    kind, safe, chronaxie = _coefficients(model)
    found = _cxx.pns(
        seq._native,
        kind=kind,
        safe=safe,
        chronaxie=chronaxie,
        rotation=_prescription(rotation).tolist(),
        gamma=_gamma(seq, system),
        keep_trace=keep_trace,
    )
    return kind, found


def check_pns(
    seq, model, *, rotation=None, system=None
) -> tuple[bool, SimpleNamespace]:
    """Return whether the nerve response stays below threshold throughout.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    model : ChronaxieModel, mapping, SAFE description, or path
        A :class:`ChronaxieModel` or a mapping of its fields; a SAFE
        description shaped like upstream's ``safe_example_hw()``; or a ``.asc``
        file for :func:`read_safe_model`.
    rotation : array_like, optional
        3x3 prescription rotation from logical to physical axes, applied after
        each block's own rotation; identity (axial) by default.
    system : pypulseq.Opts, optional
        Source of the gyromagnetic ratio; the sequence's own by default.

    Returns
    -------
    is_ok : bool
        True when the axis-combined response stays below 1 at every sample.
    report : SimpleNamespace
        ``model`` (``"safe"`` or ``"chronaxie"``), ``raster`` (s),
        ``samples``, ``peak``, the largest root-sum-square response, and
        ``axes``, the largest response of each of x, y and z. A peak carries
        ``value`` as a fraction of threshold, the sample ``time`` (s) and the
        1-based ``block`` that plays it.

    Notes
    -----
    The gradient is sampled at the centres of the sequence's own gradient
    raster, from rest, and its slew is the difference between neighbouring
    samples; this and the SAFE model are upstream PyPulseq's ``calc_pns``.
    The response is evaluated over the whole sequence in one pass, carrying
    each model's memory, in bounded storage.
    """
    kind, found = _pns(seq, model, rotation, system, keep_trace=False)
    report = SimpleNamespace(
        model=kind,
        raster=found["raster"],
        samples=found["samples"],
        peak=SimpleNamespace(**found["norm"]),
        axes=[
            SimpleNamespace(axis=name, **peak)
            for name, peak in zip("xyz", found["axes"], strict=True)
        ],
    )
    return report.peak.value < 1.0, report
