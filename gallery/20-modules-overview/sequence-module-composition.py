"""
============================================
What a sequence module publishes
============================================

A sequence module solves the layout of one group of blocks at construction and
exposes the resulting events for a scan loop to place. The architecture is
described in :doc:`/explanations/design/sequence-module`; this page shows the
interface running.
"""

# %%
# Constructing a module
# ---------------------
#
# ``SpatialSelectiveExcitation`` designs an SLR pulse, its selection gradient
# and the rephaser that unwinds the second half of the selection.

import pypulseqpp as pp
import pypulseqpp.sequences as design

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=30e-6,
    adc_dead_time=10e-6,
)

excitation = design.SpatialSelectiveExcitation(
    system, flip_angle_deg=12.0, thickness_m=5e-3, duration_s=3e-3
)
print(f"{len(excitation.blocks)} blocks, {excitation.duration * 1e3:.2f} ms")
print("events:", ", ".join(sorted(vars(excitation.events))))

# %%
# Blocks, events and timing reference
# -----------------------------------
#
# ``blocks`` returns the block tuples in play order, so a module that needs no
# per-view modification is added to a sequence as it stands. The named events
# are the individual events a loop scales or offsets. ``center`` is the
# module's timing reference, in seconds from its start.

print(
    f"center at {excitation.center * 1e3:.3f} ms of {excitation.duration * 1e3:.3f} ms"
)
print(f"selection amplitude {excitation.selection_amplitude * 1e-3:.1f} kHz/m")

# %%
# Composing modules
# -----------------
#
# Intervals between modules are measured between their timing references. An
# inversion time runs from one pulse centre to the next, so the recovery delay
# subtracts the part of the inversion module after its pulse and the part of
# the excitation module before its own.

inversion = design.InversionPreparation(system, duration_s=10e-3, bandwidth_hz=40e3)

inversion_time = 300e-3
recovery = pp.make_delay(
    pp.round_to_raster(
        inversion_time - (inversion.duration - inversion.center) - excitation.center,
        system.block_duration_raster,
    )
)

readout = design.LineReadout2D(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=(220e-3, 220e-3),
    matrix=(128, 128),
    te=None,
)

seq = pp.Sequence(system=system)
for block in inversion.blocks:
    seq.add_block(*block)
seq.add_block(recovery)
for block in readout.blocks:
    seq.add_block(*block)

pulses = seq.rf_times(compat=False)
uses = list(pulses.use)
played = pulses.t[uses.index("excitation")] - pulses.t[uses.index("inversion")]
print(f"prescribed TI {inversion_time * 1e3:.1f} ms, played {played * 1e3:.1f} ms")

# %%
# Sequence diagram
# ----------------

seq.paper_plot(tr=1)
