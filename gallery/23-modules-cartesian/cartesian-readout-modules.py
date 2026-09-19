"""
==========================
Cartesian readout modules
==========================

A readout module solves the rest of the repetition around an excitation it is
given: the prewinders, the phase-encode template at its largest step, the
acquisition window and the spoiler. The Cartesian family differs in how many
views one excitation reads.
"""

# %%
# One view per excitation, and a whole train
# ------------------------------------------
#
# ``LineReadout2D`` reads one phase-encode line. ``FseReadout3D`` reads one
# ``(line, partition)`` view per refocused echo of a CPMG train.
# ``EpiReadout2D`` reads the whole phase-encode axis after one excitation.

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
    system, flip_angle_deg=70.0, thickness_m=4e-3, duration_s=3e-3
)

line = design.LineReadout2D(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=(220e-3, 220e-3),
    matrix=(128, 128),
    te=None,
)
epi = design.EpiReadout2D(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=(220e-3, 220e-3),
    matrix=(64, 64),
)

for name, module in (("LineReadout2D", line), ("EpiReadout2D", epi)):
    print(
        f"{name:16} {len(module.blocks):3d} blocks, {module.duration * 1e3:7.2f} ms, "
        f"TE {module.echo_time * 1e3:6.2f} ms"
    )
print(f"EPI echo spacing {epi.esp * 1e6:.0f} us over {epi.etl} echoes")

# %%
# The published events
# --------------------
#
# A scan loop scales the phase-encoding event for each view; the remaining
# events are unchanged.

print("LineReadout2D events:", ", ".join(sorted(vars(line.events))))

# %%
# One repetition of each
# ----------------------

seq = pp.Sequence(system=system)
for block in line.blocks:
    seq.add_block(*block)
seq.paper_plot(tr=1)

# %%
# The echo-planar readout, which covers the same axis in one shot.

train = pp.Sequence(system=system)
for block in epi.blocks:
    train.add_block(*block)
train.paper_plot(tr=1)
