"""
====================
Free induction decay
====================

The scope of this notebook is to build the smallest complete Pulseq sequence,
a pulse-acquire experiment, and to establish the vocabulary the rest of the
course adds to: the system limits a factory solves against, the events that
carry a pulse and an acquisition window, the blocks that play them, the timing
check, and the file that is written.

Outline:

#. **System limits.** The limits, rasters and dead times an event is built
   against.
#. **Two events.** A hard excitation pulse and an acquisition window, and the
   relation between samples, dwell time and receiver bandwidth.
#. **Two blocks.** Placing the events in time, and stating a repetition time.
#. **Sequence diagram.** Reading the result as a pulse-sequence diagram.
#. **Transverse magnetisation against flip angle.** Simulating the pulse the
   sequence holds, against the closed form for a hard pulse on resonance.
#. **Writing the file.** What a ``.seq`` file records beside the block table.

The representation these objects belong to is described in
:doc:`/explanations/pulseq/events-and-blocks`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# System limits
# -------------
#
# Every factory solves its waveforms against a set of limits, and every event
# time is quantized to the rasters those limits declare. The dead times bound
# what the transmit and receive chains can do rather than what the sequencer
# can address, and the timing check reports them separately.

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
# Two events
# ----------
#
# A rectangular pulse of constant amplitude, and an acquisition window of 8192
# samples. The dwell time is the sampling interval, and its reciprocal is the
# receiver bandwidth; :func:`~pypulseqpp.calc_adc_timing` moves a requested
# dwell onto the ADC raster and lands the acquisition duration on the gradient
# raster, which is what the timing check requires of both.

FLIP_ANGLES_DEG = (10.0, 30.0, 60.0, 90.0)

pulses = [
    pp.make_block_pulse(
        flip_angle=np.deg2rad(flip_angle),
        duration=200e-6,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    for flip_angle in FLIP_ANGLES_DEG
]
dwell, acquisition = pp.calc_adc_timing(
    8192,
    32e-6,
    grad_raster_time=system.grad_raster_time,
    adc_raster_time=system.adc_raster_time,
)
adc = pp.make_adc(
    num_samples=8192, dwell=dwell, delay=system.adc_dead_time, system=system
)

print(
    f"dwell {dwell * 1e6:.0f} us, receiver bandwidth {1e-3 / dwell:.1f} kHz, "
    f"acquisition {acquisition * 1e3:.1f} ms"
)

# %%
# Two blocks
# ----------
#
# A block holds at most one event per channel, and the events in it start
# together on the block's clock. Blocks are played back to back, so the
# acquisition begins when the pulse's block ends.
#
# One repetition per flip angle. The acquisition block carries a delay longer
# than the acquisition window, which is how a repetition time is stated: a
# block longer than the events in it is a delay.

seq = pp.Sequence(system=system)
for pulse in pulses:
    seq.add_block(pulse)
    seq.add_block(adc, pp.make_delay(500e-3))

ok, errors = seq.check_timing()
print(f"timing {ok}, {seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# The solid trace is one repetition; the shaded traces are the others, which
# differ only in the amplitude of the pulse.

seq.paper_plot()

# %%
# Transverse magnetisation against flip angle
# -------------------------------------------
#
# :func:`~pypulseqpp.sim_rf` integrates the Bloch equations over the pulse the
# sequence holds. On resonance a rectangular pulse rotates the magnetisation by
# its nominal flip angle, so the transverse component follows
# :math:`|M_{xy}| = \sin\alpha`.

on_resonance = []
for pulse in pulses:
    mz_xy, frequency = pp.sim_rf(pulse)[1:3]
    on_resonance.append(abs(mz_xy[int(np.argmin(abs(frequency)))]))

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.55, 3.0))
angles = np.linspace(0.0, 95.0, 200)
axis.plot(angles, np.sin(np.deg2rad(angles)), lw=1.2, label=r"$\sin\alpha$")
axis.plot(FLIP_ANGLES_DEG, on_resonance, "o", ms=6, label="simulated")
axis.set_xlabel("flip angle (degrees)")
axis.set_ylabel("$|M_{xy}|$")
axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.22))
figure.tight_layout(rect=(0, 0, 1, 0.9))
# sphinx_gallery_end_ignore

# %%
# Writing the file
# ----------------
#
# The definitions are written beside the block table and are what a
# reconstruction reads to interpret the acquisition.

from pathlib import Path
from tempfile import mkdtemp

seq.set_definition("Name", "fid")
path = Path(mkdtemp()) / "fid.seq"
seq.write(str(path))
print(path.read_text().splitlines()[0])
