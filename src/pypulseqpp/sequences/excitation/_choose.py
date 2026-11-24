"""Choosing between the excitations a volumetric sequence is prescribed with."""

from __future__ import annotations

__all__ = ["EXCITATIONS", "make_excitation"]

import math
from typing import Any

from .nonselective import NonSelectiveExcitation
from .selective import SpatialSelectiveExcitation
from .spectral import SpspExcitation

#: The excitations :func:`make_excitation` builds, in the order a prescription
#: usually offers them.
EXCITATIONS = ("nonselective", "slab", "spsp")


#: Subpulse counts and spectral time-bandwidth products the water-selective
#: slab pulse is tried with. The spectral bandwidth is the fat offset, so the
#: passband's first alias lies at the subpulse rate, ``n_subpulses /
#: time-bandwidth`` fat offsets from water; every design keeps it at two or
#: more, fat midway or nearer water than the alias. A spectral time-bandwidth
#: product of 4 narrows the transition band on the fat side, which keeps the
#: methylene peaks in the stopband for water up to about 300 Hz above
#: resonance; fewer subpulses, then a longer spectral envelope, give each
#: subpulse longer to select a thin slab.
SPSP_DESIGNS = ((10, 4.0), (10, 3.0), (8, 3.0), (8, 4.0))

#: Longest spectral envelope (s) a design is preferred with: at the fat offset
#: of 7 T, the sharpest design fits it; at lower fields, where none does, the
#: designs are tried shortest envelope first.
SPSP_MAX_ENVELOPE_S = 5e-3


def make_excitation(
    system,
    kind: str,
    flip_angle_deg: float,
    thickness_m: float,
    *,
    duration_s: float = 3e-3,
    time_bw_product: float = 4.0,
    hard_duration_s: float = 0.5e-3,
    fat_shift_ppm: float = -3.4,
) -> Any:
    """Build the excitation module named by ``kind`` for a slab prescription.

    The three kinds share one prescription, so a volumetric sequence selects
    among them by ``kind`` alone.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    kind : {'nonselective', 'slab', 'spsp'}
        A hard pulse, a slab-selective SLR pulse, or a slab- and
        water-selective spectral-spatial pulse, as :data:`EXCITATIONS` names
        them.
    flip_angle_deg : float
        Flip angle (degrees).
    thickness_m : float
        Slab thickness (m). A hard pulse excites everything and ignores it.
    duration_s : float, default=0.003
        Duration of the ``'slab'`` pulse (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product of the ``'slab'`` SLR pulse.
    hard_duration_s : float, default=0.0005
        Duration of the hard pulse (s), rounded up to an even number of block
        rasters so that its centre lands on the raster, which a spin echo
        needs in order to place its refocusing pulse midway.
    fat_shift_ppm : float, default=-3.4
        Methylene shift from water (ppm), converted with ``system.gamma`` and
        ``system.B0`` to the frequency band the ``'spsp'`` pulse rejects.
        The ``'spsp'`` pulse is the first design of ``SPSP_DESIGNS`` whose
        spectral envelope fits ``SPSP_MAX_ENVELOPE_S`` and whose
        slab-selection ramps the slew allows; where no envelope fits, the
        designs are tried shortest envelope first.

    Returns
    -------
    RfModule
        The excitation module, which publishes its pulse as ``rf`` and, when
        it selects, a ``gz`` and a ``gz_reph`` beside it.

    Raises
    ------
    ValueError
        If ``kind`` is not one of :data:`EXCITATIONS`, or no design of
        ``SPSP_DESIGNS`` selects the slab.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=40.0, grad_unit="mT/m")
    >>> slab = pp.sequences.make_excitation(system, "slab", 12.0, 0.128)
    >>> slab.gz is not None
    True
    >>> pp.sequences.make_excitation(system, "nonselective", 12.0, 0.128).rf.use
    'excitation'
    """
    if kind not in EXCITATIONS:
        raise ValueError(f"kind must be one of {EXCITATIONS}, got {kind!r}")
    if kind == "nonselective":
        raster = system.block_duration_raster
        duration = 2 * raster * math.ceil(hard_duration_s / (2 * raster) - 1e-9)
        return NonSelectiveExcitation(system, flip_angle_deg, duration_s=duration)
    if kind == "slab":
        return SpatialSelectiveExcitation(
            system,
            flip_angle_deg,
            thickness_m,
            duration_s=duration_s,
            time_bw_product=time_bw_product,
            is_slab=True,
        )
    fat_offset_hz = fat_shift_ppm * 1e-6 * system.gamma * system.B0
    for n_subpulses, spectral_time_bw_product in spsp_designs(abs(fat_offset_hz)):
        try:
            return SpspExcitation(
                system,
                flip_angle_deg,
                thickness_m=thickness_m,
                spectral_bandwidth_hz=abs(fat_offset_hz),
                n_subpulses=n_subpulses,
                spectral_time_bw_product=spectral_time_bw_product,
                is_slab=True,
            )
        except ValueError as error:
            refusal = error
    raise refusal


def spsp_designs(spectral_bandwidth_hz: float) -> list[tuple[int, float]]:
    """Return ``SPSP_DESIGNS`` in the order :func:`make_excitation` tries them.

    Designs whose spectral envelope, ``time-bandwidth / spectral_bandwidth_hz``
    (s), fits ``SPSP_MAX_ENVELOPE_S`` come first in their listed order, then
    the others, shortest envelope first.
    """
    fits = [
        d for d in SPSP_DESIGNS if d[1] / spectral_bandwidth_hz <= SPSP_MAX_ENVELOPE_S
    ]
    longer = sorted((d for d in SPSP_DESIGNS if d not in fits), key=lambda d: d[1])
    return fits + longer
