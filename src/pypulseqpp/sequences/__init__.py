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

from ._app import SequenceApp
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

#: Modules built around one RF pulse and its selection gradients, if any.
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
#: of the TR. One class per geometry, which fixes the block layout;
#: direction, echo count and density are arguments.
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

#: Complete sequences, one per module of the repo's `examples/sequence/`, which
#: is installed as the subpackage `sequences.sequence`. Each is reached here by
#: name, imported on first use, and callable as its ``main``.
ZOO = tuple(sorted(_importlib.import_module(f"{__name__}.sequence").__all__))

__all__ = sorted(
    {*EXCITATION, *PREPARATION, *READOUT, *BASES, *ZOO, "SequenceApp", "SequenceModule"}
)


class SequenceScript(_ModuleType):
    """A zoo module whose call is its ``main``.

    The module's ``__doc__`` and ``__signature__`` are ``main``'s, not the
    file's, so :func:`help` and :func:`inspect.signature` describe the call.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.main(*args, **kwargs)


def _as_script(module: _ModuleType) -> _ModuleType:
    """Retype ``module`` in place as a :class:`SequenceScript`.

    Modules without ``main`` are returned unchanged.
    """
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
