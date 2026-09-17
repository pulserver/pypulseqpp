"""
======================
A basic Pulseq sequence
======================

The shortest complete workflow: system limits, events, blocks, a sequence, and
the file it is written to. Two repetitions of a slice-selective gradient echo
are enough to show every step.

The representation these objects belong to is described in
:doc:`/explanations/pulseq/events-and-blocks`.
"""

# %%
# System limits
# -------------
#
# Every factory solves its waveforms against a set of limits, and every event
# time is quantized to the rasters they declare.

import numpy as np

import pypulseqpp as pp

system = pp.Opts(
    max_grad=32.0,
    grad_unit="mT/m",
    max_slew=130.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=20e-6,
    adc_dead_time=10e-6,
)

# %%
# Events
# ------
#
# An RF pulse with its slice-selection gradient and rephaser, a readout
# gradient with its prewinder, a phase-encode gradient at its largest step, and
# an acquisition window.

fov, matrix, thickness = 220e-3, 64, 5e-3

rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(12.0),
    duration=2e-3,
    slice_thickness=thickness,
    apodization=0.5,
    time_bw_product=4.0,
    system=system,
    return_gz=True,
)
gx = pp.make_trapezoid(
    channel="x", flat_area=matrix / fov, flat_time=3.2e-3, system=system
)
gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)
gy_pre = pp.make_trapezoid(
    channel="y", area=matrix / (2 * fov), duration=1e-3, system=system
)
adc = pp.make_adc(
    num_samples=matrix, duration=gx.flat_time, delay=gx.rise_time, system=system
)

# %%
# Blocks
# ------
#
# A block holds at most one event per channel and the events in it are played
# together. Blocks are played back to back, so a block longer than its events
# is a delay. One repetition is four blocks; the phase encode is scaled per
# line.

seq = pp.Sequence(system=system)
for line in (-1.0, 1.0):
    seq.add_block(rf, gz)
    seq.add_block(gx_pre, pp.scale_grad(gy_pre, line), gz_reph)
    seq.add_block(gx, adc)
    seq.add_block(pp.make_delay(20e-3))

# %%
# Timing and inspection
# ---------------------
#
# ``check_timing`` establishes that every event time is addressable on the
# raster its event is played on and that the dead times are respected.

ok, errors = seq.check_timing()
print(f"timing: {ok}, {len(errors)} errors, {seq.num_blocks} blocks")

seq.paper_plot(tr=1)

# %%
# Writing the file
# ----------------
#
# The definitions a reconstruction reads are written beside the block table.

from pathlib import Path

seq.set_definition("FOV", [fov, fov, thickness])
seq.set_definition("Name", "basic_gre")
seq.write("basic_gre.seq")
print(Path("basic_gre.seq").read_text().splitlines()[0])
