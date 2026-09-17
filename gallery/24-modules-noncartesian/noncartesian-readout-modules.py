"""
==============================
Non-Cartesian readout modules
==============================

A non-Cartesian readout designs one base interleaf — its acquisition window and
the gradients that prewind to and rewind from the centre of k-space — and the
scan loop rotates it per shot with a ``ROTATIONS`` extension. One interleaf in
the gradient library therefore serves the whole scan, however many angles it is
played at.
"""

# %%
# Radial, spiral and PROPELLER interleaves
# ----------------------------------------

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
shared = (excitation.rf, excitation.gz, excitation.gz_reph)

radial = design.RadialReadout2D(system, *shared, fov=220e-3, matrix=192)
spiral = design.SpiralReadout2D(
    system, *shared, fov=220e-3, matrix=192, design_interleaves=16
)
blade = design.PropellerReadout2D(
    system, *shared, fov=220e-3, matrix=192, blade_width=16
)

for name, module in (
    ("RadialReadout2D", radial),
    ("SpiralReadout2D", spiral),
    ("PropellerReadout2D", blade),
):
    print(f"{name:20} {len(module.blocks):2d} blocks, {module.duration * 1e3:6.2f} ms")

# %%
# The solved interleaf
# --------------------
#
# A spiral arm is solved against the gradient system rather than taken from a
# fixed shape, so the module reports what it achieved rather than what was
# requested: the receiver bandwidth, the sample count and the duration of the
# acquisition window.

for name, module in (("radial", radial), ("spiral", spiral), ("blade", blade)):
    window = module.n_samples / module.bandwidth_hz
    print(
        f"{name:8} {module.n_samples:5d} samples at "
        f"{module.bandwidth_hz * 1e-3:6.1f} kHz over {window * 1e3:6.2f} ms"
    )

# %%
# The waveform of a solved spiral arm, and the limits that bound it, are the
# subject of :doc:`spiral-readout-limits` in this section. The trajectories the
# loop rotates these interleaves onto are shown on the sequence pages under
# :doc:`/examples/built-in-sequences/index`.
