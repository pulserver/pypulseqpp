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
    """Build the excitation ``kind`` names, from one prescription of a slab.

    A volumetric sequence is prescribed with a slab and then excites it one of
    three ways, and which one it is does not change anything else about the
    sequence. This is that choice.

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
        Duration of the selective pulse (s).
    time_bw_product : float, default=4.0
        Time-bandwidth product of the SLR design.
    hard_duration_s : float, default=0.0005
        Duration of the hard pulse (s), rounded up to an even number of block
        rasters so that its centre lands on the raster, which a spin echo
        needs in order to place its refocusing pulse midway.
    fat_shift_ppm : float, default=-3.4
        Methylene shift from water (ppm), converted against ``system.gamma``
        and ``system.B0`` into the spectral band the ``'spsp'`` pulse rejects.

    Returns
    -------
    RfModule
        The excitation module, which publishes its pulse as ``rf`` and, when
        it selects, a ``gz`` and a ``gz_reph`` beside it.

    Raises
    ------
    ValueError
        If ``kind`` is not one of :data:`EXCITATIONS`.

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
    return SpspExcitation(
        system,
        flip_angle_deg,
        thickness_m=thickness_m,
        spectral_bandwidth_hz=abs(fat_offset_hz),
        is_slab=True,
    )
