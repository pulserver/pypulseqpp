"""System limits and shared raster defaults."""

from __future__ import annotations

__all__ = [
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


class Opts(_pp.Opts):
    """PyPulseq system limits with shared raster defaults.

    The constructor accepts PyPulseq's parameters. Defaults are 20 us for
    gradients and block durations, 2 us for RF and ADC, and zero for RF/ADC
    dead time and RF ringdown. Specify the actual hardware timings before
    checking a sequence.

    Gradient and slew limits are stored in Hz/m and Hz/m/s after conversion
    from grad_unit and slew_unit; gamma is in Hz/T and B0 in T.

    Importing pypulseqpp installs these defaults for PyPulseq factories.
    Constructing Opts alone does not change the shared default; use
    set_as_default or reset_default.
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

    @classmethod
    def reset_default(cls) -> None:
        """Make a fresh vendor-neutral system the shared default."""
        cls().set_as_default()


Opts.reset_default()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """``system``, or the shared default when it is ``None``."""
    return _pp.Opts.default if system is None else system


#: The fraction of the hardware limits a designed waveform may reach.
MAX_GRAD_DERATE = 0.9
MAX_SLEW_DERATE = 0.9


def apply_system_derates(
    opts: _pp.Opts,
    *,
    grad_derate: float = MAX_GRAD_DERATE,
    slew_derate: float = MAX_SLEW_DERATE,
) -> _pp.Opts:
    """Return a copy with gradient and slew limits scaled from their base values.

    Base limits are retained on the copy, so repeated derating does not compound.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits. Not modified.
    grad_derate, slew_derate : float, optional
        Fraction of the base ``max_grad`` / ``max_slew`` to allow.

    Returns
    -------
    pypulseq.Opts
        The derated copy.

    See Also
    --------
    cap_system : the same copy-not-mutate contract, for an absolute ceiling.

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
    """
    derated = _copy.copy(opts)
    if not hasattr(derated, "_base_max_grad"):
        derated._base_max_grad = float(opts.max_grad)
    if not hasattr(derated, "_base_max_slew"):
        derated._base_max_slew = float(opts.max_slew)

    derated.max_grad = derated._base_max_grad * float(grad_derate)
    derated.max_slew = derated._base_max_slew * float(slew_derate)
    return derated


def cap_system(
    opts: _pp.Opts,
    *,
    max_grad: float | None = None,
    max_slew: float | None = None,
    grad_unit: str = "mT/m",
    slew_unit: str = "T/m/s",
) -> _pp.Opts:
    """Return a copy with gradient and slew limits lowered to the specified ceilings.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits. Not modified.
    max_grad : float, optional
        Amplitude ceiling, in ``grad_unit``.
    max_slew : float, optional
        Slew ceiling, in ``slew_unit``.
    grad_unit, slew_unit : str, optional
        The units those two are given in.

    Returns
    -------
    pypulseq.Opts
        The capped copy.

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
    if max_grad is not None:
        ceiling = _convert(
            from_value=float(max_grad),
            from_unit=grad_unit,
            to_unit="Hz/m",
            gamma=opts.gamma,
        )
        capped.max_grad = min(float(opts.max_grad), ceiling)
    if max_slew is not None:
        ceiling = _convert(
            from_value=float(max_slew),
            from_unit=slew_unit,
            to_unit="Hz/m/s",
            gamma=opts.gamma,
        )
        capped.max_slew = min(float(opts.max_slew), ceiling)
    return capped
