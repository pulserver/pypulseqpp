"""Magnetisation-preparation modules."""

from __future__ import annotations

from .diffusion import DiffusionPreparation, diffusion_pair_b_value
from .fatsat import (
    FAT_T1_S,
    FatSaturation,
    fat_null_flip_deg,
    fat_null_flip_deg_cycle,
)
from .inversion import (
    InversionPreparation,
    InversionSlots,
    inversion_slots,
    shortest_inversion_slot,
)
from .saturation import (
    BlochSiegertPreparation,
    IhMtPreparation,
    MtPreparation,
    OffResonanceSaturation,
)
from .spatialsat import SpatialSaturation, spatial_saturations
from .t2prep import T1T2Preparation, T2Preparation

__all__ = [
    "FAT_T1_S",
    "BlochSiegertPreparation",
    "DiffusionPreparation",
    "FatSaturation",
    "IhMtPreparation",
    "InversionPreparation",
    "InversionSlots",
    "MtPreparation",
    "OffResonanceSaturation",
    "SpatialSaturation",
    "T1T2Preparation",
    "T2Preparation",
    "diffusion_pair_b_value",
    "fat_null_flip_deg",
    "fat_null_flip_deg_cycle",
    "inversion_slots",
    "shortest_inversion_slot",
    "spatial_saturations",
]
