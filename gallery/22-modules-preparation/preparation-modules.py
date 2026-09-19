"""
====================
Preparation modules
====================

Preparation modules put the magnetisation into a stated condition and acquire
nothing. None of them defines a recovery interval: the interval between a
preparation and the readout that follows belongs to the scan loop, so the same
module serves a single-shot and a segmented acquisition.
"""

# %%
# Inversion, fat saturation and T2 preparation
# --------------------------------------------

import pypulseqpp as pp
import pypulseqpp.sequences as design

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=30e-6,
)

inversion = design.InversionPreparation(system, duration_s=10e-3, bandwidth_hz=40e3)
fat_sat = design.FatSaturation(system, freq_offset_ppm=-3.45, flip_angle_deg=110.0)
t2_prep = design.T2Preparation(system, echo_time_s=50e-3)

for name, module in (
    ("InversionPreparation", inversion),
    ("FatSaturation", fat_sat),
    ("T2Preparation", t2_prep),
):
    print(f"{name:22} {len(module.blocks):2d} blocks, {module.duration * 1e3:6.2f} ms")

# %%
# Block layouts
# -------------
#
# A scan loop inserts a preparation's blocks before the readout and defines
# any subsequent recovery interval. Five-millisecond separators keep the three
# block layouts distinct in this comparison; they are not part of the modules.

seq = pp.Sequence(system=system)
for module in (fat_sat, t2_prep, inversion):
    for block in module.blocks:
        seq.add_block(*block)
    seq.add_block(pp.make_delay(5e-3))

print(f"{seq.num_blocks} blocks, {seq.duration()[0] * 1e3:.2f} ms")
seq.paper_plot(tr=1)
