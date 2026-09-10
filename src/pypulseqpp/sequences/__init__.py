"""Reusable, composable sequence modules.

A :class:`SequenceModule` is a handful of Pulseq blocks that always travel
together -- an excitation with its slice-select and rephaser, a preparation
with its spoiler, one whole readout TR. It solves its gradients, budgets its
TE and TR and lands its ADC on both rasters once, at construction, and
publishes the events under the names its constructor gave them, so a scan
loop reaches them by name.

Modules are a convenience, never a requirement. Nothing here iterates the
sequence for you and nothing hides a scan loop: a script reads the events it
wants, scales them per shot, and writes plain Pulseq::

    import pypulseqpp as pp
    from pypulseqpp import sequences

    system = pp.Opts()
    excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3, is_slab=True)
    readout = sequences.LineReadout3D(
        system, excitation.rf, excitation.gz, fov=(0.22, 0.22, 0.12),
        matrix=(128, 128, 64), te=4e-3, tr=10e-3,
    )

    phases = pp.make_rf_spoiling_schedule(128 * 64)
    seq = pp.Sequence(system)
    for shot, (ky, kz) in enumerate(plan):
        readout.rf.phase_offset = readout.adc.phase_offset = phases[shot]
        seq.add_block(readout.rf, readout.gz)
        seq.add_block(
            pp.scale_grad(readout.gy_pre, ky),
            pp.scale_grad(readout.gz_pre, kz),
            readout.gx_pre,
        )
        seq.add_block(readout.gx, readout.adc)

The encoding plan is the script's own. The masks, orderings and angle
generators one is usually built from are a layer down, in the main namespace
-- ``make_uniform_mask``, ``make_poisson_disc_mask``, ``calc_traversal_order``,
``calc_golden_angles`` -- because they answer with plain arrays.

Beside the modules sit the complete sequences they are composed into -- the
zoo -- reached by name and callable as the sequence each builds::

    seq = sequences.gre2D_sequence(n_x=128, n_y=128, n_slices=5)

Each is also a script: ``python -m pypulseqpp.sequences.sequence.gre2D_sequence
-o gre2d.seq --n-y 64``, which is :func:`pypulseqpp.cli.run` reading the same
signature.
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
