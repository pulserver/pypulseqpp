"""Composable excitation, preparation and readout modules.

Modules expose reusable events and block layouts. The caller supplies the
acquisition loop, sampling order and per-shot event changes. Complete
sequences built from them are sequence functions, reached as
``sequences.<name>`` and callable as their ``main``. :class:`Labels` writes the
label events of a sequence function, :func:`parameters` reads its protocol, and
:func:`write` and :func:`duration` write and time what it returns.
"""

from __future__ import annotations

import importlib as _importlib
import inspect as _inspect
from types import ModuleType as _ModuleType
from typing import Any

from ._labels import Labels
from ._module import SequenceModule
from ._parameters import ProtocolParameter, parameters
from ._steady import STEADY_STATE_T1_S, steady_state_dummies
from ._write import duration, write
from .excitation import (
    EXCITATIONS,
    FrequencySelectiveExcitation,
    MultibandExcitation,
    NonSelectiveExcitation,
    NonSelectiveRefocusing,
    RfModule,
    SmsExcitation,
    SpatialSelective2DExcitation,
    SpatialSelectiveExcitation,
    SpatialSelectiveRefocusing,
    SpspExcitation,
    make_excitation,
)
from .preparation import (
    BlochSiegertPreparation,
    DiffusionPreparation,
    FatSaturation,
    IhMtPreparation,
    InversionPreparation,
    MtPreparation,
    OffResonanceSaturation,
    SpatialSaturation,
    T1T2Preparation,
    T2Preparation,
    spatial_saturations,
)
from .readout import (
    Arbitrary,
    BssfpReadout2D,
    BssfpReadout3D,
    EpiReadout2D,
    EpiReadout3D,
    FseReadout2D,
    FseReadout3D,
    LineReadout2D,
    LineReadout3D,
    NonCartesianGradient,
    NonCartesianReadout,
    PropellerReadout2D,
    PropellerStackReadout,
    RadialProjectionReadout,
    RadialReadout2D,
    RadialStackReadout,
    Rosette,
    RosetteProjectionReadout,
    RosetteReadout2D,
    RosetteStackReadout,
    Spiral,
    SpiralNavigator,
    SpiralProjectionReadout,
    SpiralReadout2D,
    SpiralStackReadout,
    ZteReadout,
)

#: Modules built around one RF pulse and its selection gradients, if any.
EXCITATION = (
    "FrequencySelectiveExcitation",
    "MultibandExcitation",
    "NonSelectiveExcitation",
    "NonSelectiveRefocusing",
    "SmsExcitation",
    "SpatialSelective2DExcitation",
    "SpatialSelectiveExcitation",
    "SpatialSelectiveRefocusing",
    "EXCITATIONS",
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
    "OffResonanceSaturation",
    "STEADY_STATE_T1_S",
    "SpatialSaturation",
    "spatial_saturations",
    "steady_state_dummies",
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
    "NonCartesianReadout",
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

#: Solved non-Cartesian interleaves, each with its ADC, prewinder and
#: rewinder, for a NonCartesianReadout to play.
TRAJECTORIES = ("Arbitrary", "NonCartesianGradient", "Rosette", "Spiral")

#: Base classes, for a family this package does not ship, and the chooser a
#: volumetric prescription selects its excitation with.
BASES = ("EXCITATIONS", "RfModule", "SequenceModule", "make_excitation")

#: Complete sequences, one per module of the repo's `examples/sequence/`, which
#: is installed as the subpackage `sequences.sequence`. Each is reached here by
#: name, imported on first use, and callable as its ``main``.
ZOO = tuple(sorted(_importlib.import_module(f"{__name__}.sequence").__all__))

__all__ = sorted(
    {
        *EXCITATION,
        *PREPARATION,
        *READOUT,
        *TRAJECTORIES,
        *BASES,
        *ZOO,
        "Labels",
        "ProtocolParameter",
        "duration",
        "parameters",
        "write",
    }
)


class SequenceScript(_ModuleType):
    """An example-sequence module whose call is its ``main``.

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
    """Import one example sequence on first use."""
    if name in ZOO:
        return _as_script(_importlib.import_module(f"{__name__}.sequence.{name}"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
