"""RF-module simulation and pulse-centre timing."""

from __future__ import annotations

__all__ = ["RfModule", "b1_fitted", "rf_reference", "selective_slr"]

from typing import Any

import numpy as np

import pypulseqpp as pp

from .._module import SequenceModule


def b1_fitted(
    system: Any, duration_s: float, make: Any, *, raster: float | None = None
) -> tuple[float, Any]:
    """Return a duration from ``duration_s`` up at which a pulse fits ``system.max_b1``, and the pulse.

    The flip angle and time-bandwidth product are ``make``'s, so the envelope
    keeps its shape and its peak falls as the duration grows. A pulse whose
    peak exceeds the limit is designed again at its duration times the ratio
    of its peak to the limit, rounded up to ``raster``, until it fits. A system
    without a ``max_b1``, a plain ``pypulseq.Opts`` among them, or with the
    pulse already within it, takes ``duration_s`` and the pulse as designed.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits; ``max_b1`` is the peak RF amplitude (Hz), ``None`` or
        absent for none.
    duration_s : float
        Duration the pulse is designed at when it fits (s).
    make : callable
        ``make(duration)`` designs the events at ``duration`` (s), the RF event
        alone or first in a tuple.
    raster : float, default=None
        Raster a lengthened duration is rounded up to (s);
        ``system.grad_raster_time`` when ``None``.

    Returns
    -------
    tuple
        ``(duration, events)``: the duration designed at (s) and what
        ``make`` returned for it.

    Raises
    ------
    ValueError
        If the pulse still exceeds the limit after four redesigns.

    Examples
    --------
    A 180 degree hard pulse of 1 ms peaks at 500 Hz, 11.7 uT; held to 5 uT it
    is lengthened by the ratio, to the next gradient raster:

    >>> import pypulseqpp as pp
    >>> from pypulseqpp.sequences.excitation._base import b1_fitted
    >>> system = pp.Opts(max_b1=5.0)
    >>> make = lambda d: pp.make_block_pulse(3.14159, duration=d, system=system)
    >>> duration, rf = b1_fitted(system, 1e-3, make)
    >>> round(duration * 1e3, 2), bool(abs(rf.signal).max() <= system.max_b1)
    (2.36, True)
    >>> b1_fitted(pp.Opts(), 1e-3, make)[0]
    0.001
    """
    limit = getattr(system, "max_b1", None)
    events = make(duration_s)
    if limit is None:
        return duration_s, events
    step = float(system.grad_raster_time if raster is None else raster)
    duration = duration_s
    for attempt in range(5):
        rf = events[0] if isinstance(events, tuple) else events
        peak = float(np.abs(np.asarray(rf.signal)).max())
        if peak <= limit * (1.0 + 1e-6):
            return duration, events
        if attempt == 4:
            break
        duration = float(np.ceil(duration * peak / limit / step - 1e-9) * step)
        events = make(duration)
    raise ValueError(
        f"the pulse peaks at {peak:.0f} Hz at {duration * 1e3:.2f} ms, above the "
        f"{limit:.0f} Hz the system allows"
    )


def rf_reference(rf: Any) -> float:
    """Return the RF pulse centre from the start of its block (s).

    This is ``delay + center``.

    Parameters
    ----------
    rf : object
        An RF event.

    Returns
    -------
    float
        Delay plus centre time of the pulse (s).
    """
    return float(rf.delay) + float(rf.center)


def selective_slr(flip_angle_deg: float, thickness_m: float, **design: Any) -> tuple:
    """Design an SLR pulse and the gradients that select ``thickness_m``.

    The thickness a pulse selects is its measured bandwidth,
    :func:`pypulseqpp.calc_rf_bandwidth`, over the selection plateau.
    :func:`pypulseqpp.make_slr_pulse` sizes the plateau for the nominal
    bandwidth, the time-bandwidth product over the duration, from which the
    designed envelope's spectrum departs; the plateau is sized again for the
    measured bandwidth.

    Parameters
    ----------
    flip_angle_deg : float
        Flip angle (degrees).
    thickness_m : float
        Thickness to select, from the pulse's measured bandwidth (m).
    **design
        The rest of :func:`pypulseqpp.make_slr_pulse`'s arguments, without
        ``slice_thickness`` and ``return_gz``.

    Returns
    -------
    tuple
        ``(rf, gz, gz_reph)``, as :func:`pypulseqpp.make_slr_pulse` returns
        them under ``return_gz``.
    """
    flip = np.deg2rad(flip_angle_deg)
    design.setdefault("delay", design["system"].rf_dead_time)
    rf, gz, _ = pp.make_slr_pulse(
        flip, slice_thickness=thickness_m, return_gz=True, **design
    )
    nominal = abs(gz.amplitude) * thickness_m
    return pp.make_slr_pulse(
        flip,
        slice_thickness=thickness_m * nominal / pp.calc_rf_bandwidth(rf),
        return_gz=True,
        **design,
    )


class RfModule(SequenceModule):
    """Sequence module whose RF pulse can be simulated across off-resonance.

    Base class of the excitation and preparation modules the package ships.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> module = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
    >>> isinstance(module, design.RfModule)
    True
    >>> mz_z, mz_xy, frequency = module.sim_rf()[:3]
    >>> mz_xy.shape == frequency.shape
    True
    """

    def sim_rf(self, pulse=None, **kwargs):
        """Simulate this module's pulse across off-resonance.

        Parameters
        ----------
        pulse : RfEvent, default=None
            Pulse to simulate; ``None`` takes the first RF event in block
            order.
        **kwargs
            Forwarded to :func:`pypulseqpp.sim_rf` (``rephase_factor``,
            ``df``, ``bandwidth_multiplier``, ``dt``).

        Returns
        -------
        tuple
            MATLAB's six answers: ``mz_z``, ``mz_xy``, the frequency axis,
            ``ref_eff``, ``mx_xy`` and ``my_xy``.

        Examples
        --------
        >>> import pypulseqpp.sequences as design
        >>> import pypulseqpp as pp
        >>> pulse = design.SpatialSelectiveExcitation(pp.Opts(), 15.0, 5e-3)
        >>> mz_z, mz_xy, frequency = pulse.sim_rf()[:3]
        >>> mz_z.shape == frequency.shape
        True
        """
        import pypulseqpp as pp

        if pulse is None:
            pulse = next(
                event
                for block in self.blocks
                for event in block
                if getattr(event, "type", None) == "rf"
            )
        return pp.sim_rf(pulse, **kwargs)
