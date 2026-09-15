"""Readout modules: one whole repetition each."""

from __future__ import annotations

from ._trajectories import Arbitrary, NonCartesianGradient, Rosette, Spiral
from .bssfp import BssfpReadout2D, BssfpReadout3D
from .epi import EpiReadout2D, EpiReadout3D
from .fse import FseReadout2D, FseReadout3D
from .line import LineReadout2D, LineReadout3D
from .navigator import SpiralNavigator
from .noncartesian import (
    NonCartesianReadout,
    RadialProjectionReadout,
    RadialReadout2D,
    RadialStackReadout,
    RosetteProjectionReadout,
    RosetteReadout2D,
    RosetteStackReadout,
    SpiralProjectionReadout,
    SpiralReadout2D,
    SpiralStackReadout,
)
from .propeller import PropellerReadout2D, PropellerStackReadout
from .zte import ZteReadout

__all__ = [
    "Arbitrary",
    "BssfpReadout2D",
    "BssfpReadout3D",
    "EpiReadout2D",
    "EpiReadout3D",
    "FseReadout2D",
    "FseReadout3D",
    "LineReadout2D",
    "LineReadout3D",
    "NonCartesianGradient",
    "NonCartesianReadout",
    "PropellerReadout2D",
    "PropellerStackReadout",
    "RadialProjectionReadout",
    "RadialReadout2D",
    "RadialStackReadout",
    "Rosette",
    "RosetteProjectionReadout",
    "RosetteReadout2D",
    "RosetteStackReadout",
    "Spiral",
    "SpiralNavigator",
    "SpiralProjectionReadout",
    "SpiralReadout2D",
    "SpiralStackReadout",
    "ZteReadout",
]
