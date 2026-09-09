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
"""

from __future__ import annotations

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

__all__ = sorted({*EXCITATION, *PREPARATION, *READOUT, *BASES, "SequenceModule"})
