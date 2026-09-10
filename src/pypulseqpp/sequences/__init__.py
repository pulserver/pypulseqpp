"""Composable excitation, preparation and readout modules.

Modules expose reusable events and block layouts. The caller supplies the
acquisition loop, sampling order and per-shot event changes. Complete
sequences built from them are reached as ``sequences.<name>`` and are
callable as their ``main``.
"""

from __future__ import annotations

import importlib as _importlib
import inspect as _inspect
from types import ModuleType as _ModuleType
from typing import Any

from ._module import SequenceModule
from .excitation import (
    FrequencySelectiveExcitation,
    Inversion,
    MultibandExcitation,
    NonSelectiveExcitation,
    NonSelectiveRefocusing,
    RfModule,
    SmsExcitation,
    SpatialSelective2DExcitation,
    SpatialSelectiveExcitation,
    SpatialSelectiveRefocusing,
    SpspExcitation,
)
from .preparation import (
    BlochSiegertPreparation,
    DiffusionPreparation,
    FatSaturation,
    IhMtPreparation,
    InversionPreparation,
    MtPreparation,
    OffResonanceSaturation,
    T1T2Preparation,
    T2Preparation,
)
from .readout import (
    BssfpReadout2D,
    BssfpReadout3D,
    EpiReadout2D,
    EpiReadout3D,
    FseReadout2D,
    FseReadout3D,
    LineReadout2D,
    LineReadout3D,
    NonCartesianReadout,
    PropellerReadout2D,
    PropellerStackReadout,
    RadialProjectionReadout,
    RadialReadout2D,
    RadialStackReadout,
    RosetteProjectionReadout,
    RosetteReadout2D,
    RosetteStackReadout,
    SpiralNavigator,
    SpiralProjectionReadout,
    SpiralReadout2D,
    SpiralStackReadout,
    ZteReadout,
)

#: Modules whose point is an RF pulse: what tips the magnetisation, and where.
EXCITATION = (
    "FrequencySelectiveExcitation",
    "Inversion",
    "MultibandExcitation",
    "NonSelectiveExcitation",
    "NonSelectiveRefocusing",
    "SmsExcitation",
    "SpatialSelective2DExcitation",
    "SpatialSelectiveExcitation",
    "SpatialSelectiveRefocusing",
    "SpspExcitation",
)

#: Modules that prepare the magnetisation before a readout samples it.
PREPARATION = (
    "BlochSiegertPreparation",
    "DiffusionPreparation",
    "FatSaturation",
    "IhMtPreparation",
    "InversionPreparation",
    "MtPreparation",
    "T1T2Preparation",
    "T2Preparation",
)

#: Modules spanning a whole repetition, from the RF that opens it to the end
#: of the TR. One class per geometry, because the geometry is what changes
#: the blocks; direction, echo count and density are arguments.
READOUT = (
    "BssfpReadout2D",
    "BssfpReadout3D",
    "EpiReadout2D",
    "EpiReadout3D",
    "FseReadout2D",
    "FseReadout3D",
    "LineReadout2D",
    "LineReadout3D",
    "PropellerReadout2D",
    "PropellerStackReadout",
    "RadialProjectionReadout",
    "RadialReadout2D",
    "RadialStackReadout",
    "RosetteProjectionReadout",
    "RosetteReadout2D",
    "RosetteStackReadout",
    "SpiralNavigator",
    "SpiralProjectionReadout",
    "SpiralReadout2D",
    "SpiralStackReadout",
    "ZteReadout",
)

#: Base classes, for a family this package does not ship.
BASES = ("NonCartesianReadout", "OffResonanceSaturation", "RfModule")

#: Complete sequences, one per module, written in the repo's `examples/sequence/`
#: and mapped in beside this package. Each is reached here by name, imported on
#: first use, and callable as the sequence it builds.
ZOO = tuple(sorted(_importlib.import_module(f"{__name__}.sequence").__all__))

__all__ = sorted({*EXCITATION, *PREPARATION, *READOUT, *BASES, *ZOO, "SequenceModule"})


class SequenceScript(_ModuleType):
    """A zoo module, callable as the sequence it builds.

    A script *is* its ``main``, so the module carries ``main``'s docstring and
    signature rather than the file's: :func:`help` and
    :func:`inspect.signature` on it answer for the call about to be made.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.main(*args, **kwargs)


def _as_script(module: _ModuleType) -> _ModuleType:
    """Present ``module`` as the ``main`` it wraps."""
    main = getattr(module, "main", None)
    if main is not None and type(module) is not SequenceScript:
        module.__class__ = SequenceScript
        module.__doc__ = main.__doc__
        module.__signature__ = _inspect.signature(main)
    return module


def __getattr__(name: str):
    """Import one zoo sequence on first use."""
    if name in ZOO:
        return _as_script(_importlib.import_module(f"{__name__}.sequence.{name}"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
