"""
=========================
Events, blocks and a file
=========================

A two-dimensional spoiled gradient-echo sequence assembled from individual
Pulseq events, written as a ``.seq`` file and read back.

The sequence is the one the Pulseq tutorials build by hand: a slice-selective
excitation, a frequency-encoded readout, a phase encode stepped line by line,
and a spoiler. Everything here is the file format's own vocabulary; the
:doc:`sequence modules <../02-modules/01-modules>` cover the same sequence
assembled from reusable layouts.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.6),
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)
# sphinx_gallery_end_ignore
import tempfile
from pathlib import Path

import numpy as np

import pypulseqpp as pp

# %%
# System limits
# -------------
#
# :class:`~pypulseqpp.Opts` carries the gradient and RF limits every factory
# solves its waveforms against, together with the rasters every event time is
# quantized to. Amplitudes are given in the units named by ``grad_unit`` and
# ``slew_unit`` and stored in Hz/m and Hz/m/s.

system = pp.Opts(
    max_grad=32.0,
    grad_unit="mT/m",
    max_slew=130.0,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

FOV = 220e-3
N_X, N_Y = 128, 64
SLICE_THICKNESS = 5e-3
FLIP_ANGLE_DEG = 12.0
TE, TR = 8e-3, 20e-3

# %%
# Excitation
# ----------
#
# An SLR pulse designed for a small tip angle, with the selection gradient and
# the rephaser that unwinds its second half. The time-bandwidth product sets
# the sharpness of the slice profile at a given duration; ``return_gz`` asks
# the factory for the two gradients that go with the pulse.

rf, gz, gz_reph = pp.make_slr_pulse(
    np.deg2rad(FLIP_ANGLE_DEG),
    duration=3e-3,
    slice_thickness=SLICE_THICKNESS,
    time_bw_product=4.0,
    return_gz=True,
    system=system,
    use="excitation",
    delay=system.rf_dead_time,
)

# %%
# Readout and encoding gradients
# ------------------------------
#
# The readout gradient is fixed by the resolution and the receiver bandwidth:
# ``flat_area`` is the zeroth moment the ADC window has to traverse, which is
# ``N_X / FOV`` in 1/m. The prewinder covers the half of that moment before the
# echo, and is negative so that the echo falls at the centre of the window.
#
# The ADC dwell and the flat top of the readout gradient sit on different
# rasters, and the acquisition window has to be a whole number of gradient
# raster periods. :func:`~pypulseqpp.calc_adc_timing` searches upward from the
# requested dwell for the first that satisfies both, so the receiver bandwidth
# it returns, ``1 / dwell``, is at or below the one asked for.

dwell, readout_duration = pp.calc_adc_timing(
    N_X,
    1.0 / 250e3,
    grad_raster_time=system.grad_raster_time,
    adc_raster_time=system.adc_raster_time,
)
print(
    f"receiver bandwidth: {1e-3 / dwell:.0f} kHz over {readout_duration * 1e3:.2f} ms"
)

gx = pp.make_trapezoid(
    "x", flat_area=N_X / FOV, flat_time=readout_duration, system=system
)
adc = pp.make_adc(num_samples=N_X, dwell=dwell, delay=gx.rise_time, system=system)

# %%
# The phase-encode template is built at its largest step and scaled per line in
# the scan loop, so the whole scan references one gradient event.

gx_pre = pp.make_trapezoid("x", area=-gx.area / 2, system=system)
gy_pre = pp.make_trapezoid("y", area=N_Y / (2 * FOV), system=system)

# %%
# Spoiling
# --------
#
# Four cycles of dephasing across one readout voxel leave no coherent
# transverse magnetization for the next repetition to refocus. Combined with a
# quadratic RF phase increment, this is the standard spoiled gradient echo;
# the increment of 117 degrees is the one Zur et al. (Magn Reson Med 1991)
# report as robust against the residual coherences a linear increment leaves.

SPOILING_CYCLES = 4.0
RF_SPOILING_INCREMENT_DEG = 117.0

gx_spoil = pp.make_trapezoid("x", area=SPOILING_CYCLES * N_X / FOV, system=system)
gz_spoil = pp.make_trapezoid("z", area=SPOILING_CYCLES / SLICE_THICKNESS, system=system)

# %%
# Delays
# ------
#
# TE runs from the centre of the RF pulse to the echo at the centre of the ADC
# window, and TR from one excitation to the next. The two delays are what is
# left once the events between those points are accounted for, rounded up onto
# the block-duration raster.

te_fill = pp.ceil_to_raster(
    TE
    - (pp.calc_duration(gz) - rf.center - rf.delay)
    - pp.calc_duration(gz_reph, gx_pre, gy_pre)
    - gx.rise_time
    - gx.flat_time / 2,
    system.block_duration_raster,
)
tr_fill = pp.ceil_to_raster(
    TR
    - pp.calc_duration(gz)
    - pp.calc_duration(gz_reph, gx_pre, gy_pre)
    - pp.calc_duration(gx)
    - pp.calc_duration(gx_spoil, gz_spoil)
    - te_fill,
    system.block_duration_raster,
)

# %%
# The scan loop
# -------------
#
# One repetition is six blocks. Events inside a block play concurrently, each
# with its own delay from the block start, and a block lasts as long as its
# longest event unless a delay makes it longer; blocks play back to back with
# no gap, so the block order is the playout order.
#
# :func:`~pypulseqpp.scale_grad` returns the phase-encode template at this
# line's step, and :func:`~pypulseqpp.make_label` writes the ``LIN`` counter a
# reconstruction sorts the acquisition by. The RF phase offset is carried on
# the ADC as well, so the receiver demodulates in the frame the pulse
# transmitted in.

seq = pp.Sequence(system=system)
phase = 0.0
increment = 0.0

for line in range(N_Y):
    rf.phase_offset = phase
    adc.phase_offset = phase
    increment += np.deg2rad(RF_SPOILING_INCREMENT_DEG)
    phase = (phase + increment) % (2 * np.pi)

    step = (line - N_Y // 2) / (N_Y / 2)
    seq.add_block(rf, gz, pp.make_label(type="SET", label="LIN", value=line))
    seq.add_block(gz_reph, gx_pre, pp.scale_grad(gy_pre, step))
    seq.add_block(pp.make_delay(te_fill))
    seq.add_block(gx, adc)
    seq.add_block(gx_spoil, gz_spoil)
    seq.add_block(pp.make_delay(tr_fill))

# %%
# Timing
# ------
#
# :meth:`~pypulseqpp.Sequence.check_timing` reports every event time that is
# not addressable on the raster it is played on, together with the transmit and
# receive dead times the system limits declare. It also records the sequence
# duration as the ``TotalDuration`` definition.

is_ok, report = seq.check_timing()
print(f"timing: {'ok' if is_ok else report}")
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.3f} s")

# %%
# The timing diagram
# ------------------
#
# :meth:`~pypulseqpp.Sequence.paper_plot` draws one repetition solid and the
# others underneath in grey, so what varies from one repetition to the next —
# here the phase-encode step and the RF phase — is visible against what does
# not.

seq.paper_plot()

# %%
# The sampling pattern
# --------------------
#
# :func:`~pypulseqpp.plot.plot_kspace` expands the gradient waveforms, applies
# each block's rotation and integrates them, and draws the k-space location of
# every ADC sample in 1/m.

pp.plot.plot_kspace(seq, color_by="shot")

# %%
# Writing and reading
# -------------------
#
# :meth:`~pypulseqpp.Sequence.write` deduplicates the event libraries, appends
# an MD5 signature and writes Pulseq 1.5.1 text. The definitions written beside
# the block table are what a reconstruction reads the prescription from, so
# anything downstream needs goes there before the file is written.

seq.set_definition(key="FOV", value=[FOV, FOV, SLICE_THICKNESS])
seq.set_definition(key="Name", value="gre_2d")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory, "gre_2d.seq")
    seq.write(str(path))
    print(f"{path.stat().st_size / 1024:.1f} kB for {seq.num_blocks} blocks")

    read_back = pp.Sequence()
    read_back.read(str(path))

print(f"{read_back.num_blocks} blocks read back")
print(f"FOV: {read_back.definitions['FOV']}")
