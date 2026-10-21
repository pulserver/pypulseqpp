"""System limits and shared raster defaults."""

from __future__ import annotations

__all__ = [
    "MAX_B1_DERATE",
    "MAX_GRAD_DERATE",
    "MAX_SLEW_DERATE",
    "Opts",
    "apply_system_derates",
    "cap_system",
    "default_system",
]

import copy as _copy

import pypulseq as _pp
from pypulseq.convert import convert as _convert

_B1_PER_UNIT_T = {"T": 1.0, "mT": 1e-3, "uT": 1e-6, "G": 1e-4}


def _b1_to_hz(value: float, unit: str, gamma: float) -> float:
    """Return a positive B1 amplitude ``value`` in ``unit`` as gamma B1 in Hz."""
    if unit != "Hz" and unit not in _B1_PER_UNIT_T:
        raise ValueError(
            f"b1_unit must be Hz or one of {sorted(_B1_PER_UNIT_T)}, got {unit!r}"
        )
    if not float(value) > 0.0:
        raise ValueError(f"max_b1 must be positive, got {value}")
    if unit == "Hz":
        return float(value)
    return float(value) * _B1_PER_UNIT_T[unit] * float(gamma)


class Opts(_pp.Opts):
    """PyPulseq system limits with shared raster defaults.

    The constructor accepts PyPulseq's parameters. Raster defaults are 20 us
    for the gradient and block duration rasters, 2 us for the RF and ADC
    rasters, and zero for the RF and ADC dead times and the RF ringdown time.
    Set the actual hardware timings before checking a sequence.

    Gradient amplitude and slew-rate limits are stored in Hz/m and Hz/m/s after
    conversion from ``grad_unit`` and ``slew_unit``; ``gamma`` is in Hz/T and
    ``B0`` in T. ``max_b1``, the peak B1 the transmit chain allows, is stored in
    Hz (gamma B1) after conversion from ``b1_unit``, one of ``Hz``, ``T``,
    ``mT``, ``uT`` (the default) and ``G``; ``None`` sets no limit. The
    excitation and refocusing modules lengthen a pulse that would exceed
    :data:`MAX_B1_DERATE` of it.

    Importing pypulseqpp installs these defaults for PyPulseq factories.
    Constructing Opts alone does not change the shared default; use
    set_as_default or reset_default.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> system.max_grad
    1703040.0
    >>> system.grad_raster_time, system.rf_raster_time
    (2e-05, 2e-06)
    >>> round(pp.Opts(max_b1=11.2).max_b1, 1), pp.Opts().max_b1
    (476.9, None)
    """

    def __init__(
        self,
        adc_dead_time: float | None = 0.0,
        adc_raster_time: float | None = 2e-6,
        block_duration_raster: float | None = 20e-6,
        gamma: float | None = None,
        grad_raster_time: float | None = 20e-6,
        grad_unit: str = "Hz/m",
        max_grad: float | None = None,
        max_slew: float | None = None,
        rf_dead_time: float | None = 0.0,
        rf_raster_time: float | None = 2e-6,
        rf_ringdown_time: float | None = 0.0,
        adc_samples_limit: int | None = None,
        adc_samples_divisor: int | None = None,
        rise_time: float | None = None,
        slew_unit: str = "Hz/m/s",
        B0: float | None = None,
        max_b1: float | None = None,
        b1_unit: str = "uT",
    ) -> None:
        # Every raster is passed on explicitly rather than left None, which
        # upstream would fill from the shared default -- and the shared
        # default is what this class is being constructed to become.
        super().__init__(
            adc_dead_time=adc_dead_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            gamma=gamma,
            grad_raster_time=grad_raster_time,
            grad_unit=grad_unit,
            max_grad=max_grad,
            max_slew=max_slew,
            rf_dead_time=rf_dead_time,
            rf_raster_time=rf_raster_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_samples_limit=adc_samples_limit,
            adc_samples_divisor=adc_samples_divisor,
            rise_time=rise_time,
            slew_unit=slew_unit,
            B0=B0,
        )
        self.max_b1 = None if max_b1 is None else _b1_to_hz(max_b1, b1_unit, self.gamma)

    @classmethod
    def reset_default(cls) -> None:
        """Install a fresh vendor-neutral system as the shared default."""
        cls().set_as_default()


Opts.reset_default()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """Return ``system``, or the shared default system when it is ``None``.

    Parameters
    ----------
    system : Opts or None
        The system limits a caller prescribed, if any.

    Returns
    -------
    Opts
        ``system`` itself, or `Opts.default`.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> pp.default_system(None) is pp.Opts.default
    True
    >>> system = pp.Opts()
    >>> pp.default_system(system) is system
    True
    """
    return _pp.Opts.default if system is None else system


#: Fraction of the gradient amplitude limit a designed waveform may reach.
MAX_GRAD_DERATE = 0.9
#: Fraction of the slew-rate limit a designed waveform may reach.
MAX_SLEW_DERATE = 0.9
#: Fraction of the peak-B1 limit a designed pulse may reach.
MAX_B1_DERATE = 0.9


def apply_system_derates(
    opts: _pp.Opts,
    *,
    grad_derate: float = MAX_GRAD_DERATE,
    slew_derate: float = MAX_SLEW_DERATE,
    b1_derate: float = MAX_B1_DERATE,
) -> _pp.Opts:
    """Return a copy with gradient, slew and peak-B1 limits scaled from their base values.

    Base limits are retained on the copy, so repeated derating does not compound,
    and a ceiling :func:`cap_system` puts on a derated copy holds when it is
    derated again.

    Parameters
    ----------
    opts : Opts
        System limits. Not modified.
    grad_derate : float, default=MAX_GRAD_DERATE
        Fraction of the base ``max_grad`` to allow.
    slew_derate : float, default=MAX_SLEW_DERATE
        Fraction of the base ``max_slew`` to allow.
    b1_derate : float, default=MAX_B1_DERATE
        Fraction of the base ``max_b1`` to allow; a system without a
        ``max_b1`` keeps none.

    Returns
    -------
    Opts
        The derated copy.

    See Also
    --------
    cap_system : the same copy-not-mutate contract, for an absolute ceiling.

    Notes
    -----
    The default fractions, :data:`MAX_GRAD_DERATE` and
    :data:`MAX_SLEW_DERATE`, leave a margin below the amplitude and slew-rate
    limits for waveforms designed on one axis at a time and played while
    other axes are active. :data:`MAX_B1_DERATE` keeps a designed pulse's
    peak below the transmit chain's limit; the excitation and refocusing
    modules fit their pulses to it whether or not the system was derated.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> derated = pp.apply_system_derates(opts)
    >>> derated.max_grad == 0.9 * opts.max_grad
    True
    >>> derated is opts
    False

    The caller's own limits are untouched, and repeating does not compound:

    >>> pp.apply_system_derates(derated).max_grad == derated.max_grad
    True
    >>> b1 = pp.apply_system_derates(pp.Opts(max_b1=11.2))
    >>> round(b1.max_b1 / b1.gamma * 1e6, 2), pp.apply_system_derates(b1).max_b1 == b1.max_b1
    (10.08, True)

    A ceiling below the derated limit survives a second derate:

    >>> capped = pp.cap_system(derated, max_grad=30)
    >>> pp.apply_system_derates(capped).max_grad == capped.max_grad
    True
    """
    derated = _copy.copy(opts)
    fractions = {"max_grad": grad_derate, "max_slew": slew_derate}
    if getattr(opts, "max_b1", None) is not None:
        fractions["max_b1"] = b1_derate
    for name, fraction in fractions.items():
        base = f"_base_{name}"
        if not hasattr(derated, base):
            setattr(derated, base, float(getattr(opts, name)))
        value = getattr(derated, base) * float(fraction)
        setattr(derated, name, min(value, getattr(derated, f"_ceiling_{name}", value)))
    return derated


def b1_design_limit(opts: _pp.Opts) -> float | None:
    """Return the peak B1 a pulse is designed to (Hz), with the B1 derate applied once.

    A system :func:`apply_system_derates` has derated already gives its
    ``max_b1``; any other gives :data:`MAX_B1_DERATE` of it. ``None`` when
    the system has no ``max_b1``, a plain ``pypulseq.Opts`` among them.

    Parameters
    ----------
    opts : Opts
        System limits.

    Returns
    -------
    float or None
        The peak a designed pulse may reach (Hz), or ``None`` for no limit.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_b1=11.2)
    >>> from pypulseqpp._opts import b1_design_limit
    >>> b1_design_limit(system) == b1_design_limit(pp.apply_system_derates(system))
    True
    >>> round(b1_design_limit(system) / system.gamma * 1e6, 2)
    10.08
    """
    limit = getattr(opts, "max_b1", None)
    if limit is None:
        return None
    if hasattr(opts, "_base_max_b1"):
        return float(limit)
    return MAX_B1_DERATE * float(limit)


def cap_system(
    opts: _pp.Opts,
    *,
    max_grad: float | None = None,
    max_slew: float | None = None,
    grad_unit: str = "mT/m",
    slew_unit: str = "T/m/s",
    max_b1: float | None = None,
    b1_unit: str = "uT",
) -> _pp.Opts:
    """Return a copy with gradient, slew and RF limits lowered to the specified ceilings.

    Parameters
    ----------
    opts : Opts
        System limits. Not modified.
    max_grad : float, default=None
        Gradient amplitude ceiling, in ``grad_unit``; the limit is unchanged
        when None or already lower.
    max_slew : float, default=None
        Slew-rate ceiling, in ``slew_unit``; as ``max_grad``.
    grad_unit : str, default='mT/m'
        Unit of ``max_grad``.
    slew_unit : str, default='T/m/s'
        Unit of ``max_slew``.
    max_b1 : float, default=None
        Peak RF amplitude ceiling, in ``b1_unit``; as ``max_grad``. A system
        with no RF limit takes it.
    b1_unit : str, default='uT'
        Unit of ``max_b1``: ``Hz`` (gamma B1), ``T``, ``mT``, ``uT`` or ``G``.

    Returns
    -------
    Opts
        The capped copy.

    Notes
    -----
    On a system :func:`apply_system_derates` has derated, a limit becomes the
    lower of the derated limit and the ceiling, and stays so when the copy is
    derated again: the margin is taken from the system's limit, not from the
    ceiling.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> from pypulseq.convert import convert
    >>> system = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    >>> capped = pp.cap_system(system, max_grad=40, max_slew=150)
    >>> capped is system
    False
    >>> round(convert(from_value=capped.max_grad, from_unit="Hz/m", to_unit="mT/m", gamma=capped.gamma))
    40
    """
    capped = _copy.copy(opts)
    ceilings = {}
    if max_grad is not None:
        ceilings["max_grad"] = _convert(
            from_value=float(max_grad),
            from_unit=grad_unit,
            to_unit="Hz/m",
            gamma=opts.gamma,
        )
    if max_slew is not None:
        ceilings["max_slew"] = _convert(
            from_value=float(max_slew),
            from_unit=slew_unit,
            to_unit="Hz/m/s",
            gamma=opts.gamma,
        )
    if max_b1 is not None:
        ceilings["max_b1"] = _b1_to_hz(max_b1, b1_unit, opts.gamma)
    for name, ceiling in ceilings.items():
        held = getattr(opts, name, None)
        setattr(capped, name, ceiling if held is None else min(float(held), ceiling))
        # A derated limit keeps its base; the ceiling is kept beside it, so
        # derating again leaves the cap in place.
        if hasattr(capped, f"_base_{name}"):
            kept = getattr(capped, f"_ceiling_{name}", ceiling)
            setattr(capped, f"_ceiling_{name}", min(kept, ceiling))
    return capped
