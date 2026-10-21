"""RF-module simulation and pulse-centre timing."""

from __future__ import annotations

__all__ = [
    "ADIABATICITY_FLOOR",
    "ADIABATIC_MAX_STRETCH",
    "RfModule",
    "adiabatic_fitted",
    "b1_fitted",
    "held_sweep",
    "rf_reference",
    "selective_slr",
]

from typing import Any

import numpy as np

import pypulseqpp as pp

from ..._adiabatic import HYPSEC_BETA, HYPSEC_MU
from ..._opts import b1_design_limit
from .._module import SequenceModule


def b1_fitted(
    system: Any, duration_s: float, make: Any, *, raster: float | None = None
) -> tuple[float, Any]:
    """Return a duration from ``duration_s`` up at which a pulse fits the derated ``system.max_b1``, and the pulse.

    The pulse is held to :data:`~pypulseqpp.MAX_B1_DERATE` of ``max_b1``, or
    to ``max_b1`` itself on a system :func:`~pypulseqpp.apply_system_derates`
    has derated, so the margin is taken once. The flip angle and
    time-bandwidth product are ``make``'s, so the envelope keeps its shape and
    its peak falls as the duration grows. A pulse whose peak exceeds the limit
    is designed again at its duration times the ratio of its peak to the
    limit, rounded up to ``raster``, until it fits. A system without a
    ``max_b1``, a plain ``pypulseq.Opts`` among them, or with the pulse
    already within the limit, takes ``duration_s`` and the pulse as designed.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits; ``max_b1`` is the transmit chain's peak RF amplitude
        (Hz), ``None`` or absent for none.
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
    A 180 degree hard pulse of 1 ms peaks at 500 Hz, 11.7 uT; under a 5 uT
    limit it is held to 4.5 uT, lengthened by the ratio to the next gradient
    raster:

    >>> import pypulseqpp as pp
    >>> from pypulseqpp.sequences.excitation._base import b1_fitted
    >>> system = pp.Opts(max_b1=5.0)
    >>> make = lambda d: pp.make_block_pulse(3.14159, duration=d, system=system)
    >>> duration, rf = b1_fitted(system, 1e-3, make)
    >>> peak_ut = float(abs(rf.signal).max()) / system.gamma * 1e6
    >>> round(duration * 1e3, 2), round(peak_ut, 2)
    (2.62, 4.48)
    >>> b1_fitted(pp.Opts(), 1e-3, make)[0]
    0.001
    """
    limit = b1_design_limit(system)
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
        f"{limit:.0f} Hz a pulse is designed to"
    )


#: Most an adiabatic pulse is lengthened past its design duration, as a factor.
ADIABATIC_MAX_STRETCH = 2.0
#: Least fraction of its design adiabaticity an adiabatic pulse is lowered to.
ADIABATICITY_FLOOR = 0.75


def held_sweep(pulse_type: str, stretch: float) -> dict[str, float]:
    """Return the :func:`~pypulseqpp.make_adiabatic_pulse` keywords that keep a sweep's bandwidth over ``stretch`` times its duration.

    A hyperbolic secant sweeps ``mu * beta`` (rad/s) over an envelope
    ``sech(beta t)``: ``beta`` falls and ``mu`` rises by ``stretch``, so the
    sweep and the envelope's truncation are kept. The other sweeps are
    defined in time normalised to the duration and need nothing.

    Parameters
    ----------
    pulse_type : str
        Sweep family, as :func:`~pypulseqpp.make_adiabatic_pulse` names them.
    stretch : float
        Duration over the design duration.

    Returns
    -------
    dict
        ``beta`` and ``mu`` for ``"hypsec"``, empty otherwise.

    Examples
    --------
    >>> from pypulseqpp.sequences.excitation._base import held_sweep
    >>> held_sweep("hypsec", 2.0)
    {'beta': 400.0, 'mu': 9.8}
    >>> held_sweep("wurst", 2.0)
    {}
    """
    if pulse_type != "hypsec":
        return {}
    return {"beta": HYPSEC_BETA / stretch, "mu": HYPSEC_MU * stretch}


def adiabatic_fitted(
    system: Any,
    duration_s: float,
    adiabaticity: float,
    make: Any,
    *,
    raster: float | None = None,
) -> tuple[float, float, Any]:
    """Return a duration and adiabaticity at which an adiabatic pulse fits the derated ``system.max_b1``, and the pulse.

    The limit is :func:`b1_fitted`'s. At a fixed swept bandwidth the peak
    meeting ``adiabaticity`` at the sweep's centre falls as
    ``sqrt(adiabaticity / duration)``, so a pulse above the limit is first
    designed again at its duration times the square of its peak's ratio to the
    limit, rounded up to ``raster``, which holds its adiabaticity, its band
    and its response to B1. Past :data:`ADIABATIC_MAX_STRETCH` times
    ``duration_s`` the duration stays there and the adiabaticity is lowered
    by the square of the remaining ratio, to no less than
    :data:`ADIABATICITY_FLOOR` of ``adiabaticity``. A system without a
    ``max_b1``, or a pulse within the limit, takes ``duration_s``,
    ``adiabaticity`` and the pulse as designed.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits; ``max_b1`` is the transmit chain's peak RF amplitude
        (Hz), ``None`` or absent for none.
    duration_s : float
        Duration the pulse is designed at when it fits (s).
    adiabaticity : float
        Sweep-rate margin over the adiabatic condition the pulse is designed
        at when it fits.
    make : callable
        ``make(duration, adiabaticity)`` designs the events at ``duration``
        (s) with the bandwidth swept at ``duration_s`` (see
        :func:`held_sweep`), the RF event alone or first in a tuple.
    raster : float, default=None
        Raster a lengthened duration is rounded up to (s);
        ``system.grad_raster_time`` when ``None``.

    Returns
    -------
    tuple
        ``(duration, adiabaticity, events)``: what the pulse was designed at
        and what ``make`` returned for it.

    Raises
    ------
    ValueError
        If the pulse exceeds the limit at :data:`ADIABATIC_MAX_STRETCH` times
        its duration and :data:`ADIABATICITY_FLOOR` of its adiabaticity.

    Examples
    --------
    The 10 ms hyperbolic-secant inversion at adiabaticity 4 peaks at 13.2 uT;
    under an 11.2 uT limit, held to 10.08 uT, it lasts 17.26 ms at the same
    adiabaticity:

    >>> import pypulseqpp as pp
    >>> from pypulseqpp.sequences.excitation._base import adiabatic_fitted, held_sweep
    >>> system = pp.Opts(max_b1=11.2)
    >>> def make(duration, adiabaticity):
    ...     return pp.make_adiabatic_pulse(
    ...         "hypsec", adiabaticity=adiabaticity, duration=duration, system=system,
    ...         **held_sweep("hypsec", duration / 10e-3),
    ...     )
    >>> duration, adiabaticity, rf = adiabatic_fitted(system, 10e-3, 4, make)
    >>> round(duration * 1e3, 2), adiabaticity
    (17.26, 4)

    Under 9.5 uT, held to 8.55 uT, twice the duration is not enough and the
    adiabaticity falls:

    >>> system = pp.Opts(max_b1=9.5)
    >>> duration, adiabaticity, rf = adiabatic_fitted(system, 10e-3, 4, make)
    >>> round(duration * 1e3, 2), round(adiabaticity, 2)
    (20.0, 3.34)
    """
    limit = b1_design_limit(system)
    events = make(duration_s, adiabaticity)
    if limit is None:
        return duration_s, adiabaticity, events
    step = float(system.grad_raster_time if raster is None else raster)
    ceiling = float(np.floor(ADIABATIC_MAX_STRETCH * duration_s / step + 1e-9) * step)
    floor = ADIABATICITY_FLOOR * adiabaticity
    duration, margin = duration_s, adiabaticity
    for attempt in range(5):
        rf = events[0] if isinstance(events, tuple) else events
        peak = float(np.abs(np.asarray(rf.signal)).max())
        if peak <= limit:
            return duration, margin, events
        if attempt == 4:
            break
        wanted = float(np.ceil(duration * (peak / limit) ** 2 / step - 1e-9) * step)
        if wanted <= ceiling:
            duration = wanted
        else:
            # The adiabaticity that brings the peak at the ceiling to just
            # under the limit; the sampled peak follows the square root of
            # the duration to within a part in 1e6.
            margin *= (1.0 - 1e-6) * (limit / peak) ** 2 * ceiling / duration
            duration = ceiling
            if margin < floor * (1.0 - 1e-9):
                raise ValueError(
                    f"the adiabatic pulse fits the {limit:.0f} Hz a pulse is "
                    f"designed to only at adiabaticity {margin:.2f} over "
                    f"{ADIABATIC_MAX_STRETCH:g} times its {duration_s * 1e3:.2f} ms, "
                    f"below {ADIABATICITY_FLOOR:g} of its {adiabaticity:g}"
                )
        events = make(duration, margin)
    raise ValueError(
        f"the adiabatic pulse peaks at {peak:.0f} Hz at {duration * 1e3:.2f} ms, "
        f"above the {limit:.0f} Hz a pulse is designed to"
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
