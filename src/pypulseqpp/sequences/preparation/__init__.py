"""Magnetisation-preparation modules."""

from __future__ import annotations

from .diffusion import DiffusionPreparation, diffusion_pair_b_value
from .fatsat import (
    FAT_INVERSION_BANDWIDTH_PPM,
    FAT_SAT_BANDWIDTH_PPM,
    FAT_T1_S,
    FatInversion,
    FatSaturation,
    fat_null_flip_deg,
    fat_null_inversion_time,
)
from .inversion import InversionPreparation
from .saturation import (
    BlochSiegertPreparation,
    IhMtPreparation,
    MtPreparation,
    OffResonanceSaturation,
)
from .spatialsat import SpatialSaturation, spatial_saturations
from .t2prep import T1T2Preparation, T2Preparation

__all__ = [
    "FAT_INVERSION_BANDWIDTH_PPM",
    "FAT_SAT_BANDWIDTH_PPM",
    "FAT_T1_S",
    "BlochSiegertPreparation",
    "DiffusionPreparation",
    "FatInversion",
    "FatSaturation",
    "IhMtPreparation",
    "InversionPreparation",
    "MtPreparation",
    "OffResonanceSaturation",
    "SpatialSaturation",
    "T1T2Preparation",
    "T2Preparation",
    "diffusion_pair_b_value",
    "fat_null_flip_deg",
    "fat_null_inversion_time",
    "spatial_saturations",
]
