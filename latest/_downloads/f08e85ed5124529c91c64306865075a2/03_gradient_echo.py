r"""
=============
Gradient echo
=============

The scope of this notebook is to turn the non-imaging experiments of the two
previous pages into a two-dimensional acquisition: the excitation becomes
slice-selective, the echo is formed by a gradient rather than by a refocusing
pulse, and a phase encode moves the acquired line from one repetition to the
next. This is the structure every Cartesian sequence in the course is a
variation on.

The observable is where the echo lands. The prewinder area decides which
sample of the readout window k-space crosses zero at, and the last section
measures the echo time that follows from it.

Outline:

#. **Prescription.** Field of view, matrix, slice and flip angle.
#. **Slice-selective excitation.** A sinc pulse with its selection gradient and
   the rephaser that unwinds the second half of the selection.
#. **Readout and phase encoding.** The readout gradient, its prewinder, the
   phase-encode table and the acquisition window.
#. **One repetition.** The four blocks that play them, and the repetition time.
#. **Sequence diagram and k-space.** The line the repetition acquires, and the
   raster the phase encodes build.
#. **Echo position against prewinder area.** Where k-space crosses zero, and
   what a partial prewinder costs and buys.

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
# Prescription
# ------------
#
# The field of view and the matrix fix the k-space extent and the sample
# spacing: a line spans :math:`N/\mathrm{FOV}` in 1/m and is sampled every
# :math:`1/\mathrm{FOV}`. Nothing below reads a length in metres except through
# those two numbers.

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

FOV = 220e-3
MATRIX = 128
THICKNESS = 5e-3
FLIP_ANGLE_DEG = 12.0
REPETITION_TIME = 20e-3

# %%
# Slice-selective excitation
# --------------------------
#
# A gradient played during the pulse makes the resonance frequency a function
# of position, so the pulse's bandwidth selects a slab of the prescribed
# thickness. Position within the slice maps to phase accumulated under the
# second half of the selection lobe, which the rephaser unwinds; without it the
# signal integrates to nothing across the slice.
#
# ``return_gz`` is what asks the factory for the two gradients beside the
# pulse.

rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.deg2rad(FLIP_ANGLE_DEG),
    duration=2e-3,
    slice_thickness=THICKNESS,
    apodization=0.5,
    time_bw_product=4.0,
    delay=system.rf_dead_time,
    system=system,
    use="excitation",
    return_gz=True,
)

print(
    f"selection {1e3 * gz.amplitude / 42.576e6:.2f} mT/m, "
    f"bandwidth {4.0 / 2e-3 * 1e-3:.1f} kHz, "
    f"rephaser {1e3 * pp.calc_duration(gz_reph):.2f} ms"
)

# %%
# Readout and phase encoding
# --------------------------
#
# The readout gradient is prescribed by the area its flat top has to cover, and
# the acquisition window is delayed into that flat top by the rise time, so
# that every sample is taken at constant amplitude. The dwell and the flat time
# come from :func:`~pypulseqpp.calc_adc_timing`, which puts the one on the ADC
# raster and the other on the gradient raster. The prewinder carries k-space to one end of the
# line before the readout traverses it, and the phase encode displaces the line
# perpendicular to it; one phase-encode step is :math:`1/\mathrm{FOV}`, so the
# largest of them is half the k-space extent.

dwell, readout_time = pp.calc_adc_timing(
    MATRIX,
    26e-6,
    grad_raster_time=system.grad_raster_time,
    adc_raster_time=system.adc_raster_time,
)

gx = pp.make_trapezoid(
    channel="x", flat_area=MATRIX / FOV, flat_time=readout_time, system=system
)
adc = pp.make_adc(num_samples=MATRIX, dwell=dwell, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)
gy_pre = pp.make_trapezoid(
    channel="y", area=MATRIX / (2 * FOV), duration=1e-3, system=system
)

# One line per phase encode, from one edge of k-space to the other.
phase_encodes = np.linspace(-1.0, 1.0, MATRIX, endpoint=False)

print(
    f"readout {1e3 * gx.amplitude / 42.576e6:.2f} mT/m, "
    f"dwell {1e6 * adc.dwell:.1f} us, "
    f"receiver bandwidth {1e-3 / adc.dwell:.1f} kHz, "
    f"{1 / (adc.dwell * MATRIX):.0f} Hz per pixel"
)

# %%
# One repetition
# --------------
#
# Four blocks: the pulse with its selection gradient, the prewinders and the
# rephaser together, the readout with the acquisition window, and a delay that
# brings the repetition up to the prescribed repetition time.
# :func:`~pypulseqpp.scale_grad` takes the largest phase encode to the step
# this repetition acquires.


def gradient_echo(prewinder_fraction=0.5):
    """The sequence, with the prewinder at the given fraction of the line."""
    prewinder = pp.scale_grad(gx_pre, 2 * prewinder_fraction)
    played = (
        pp.calc_duration(rf, gz)
        + pp.calc_duration(prewinder, gy_pre, gz_reph)
        + pp.calc_duration(gx, adc)
    )
    seq = pp.Sequence(system=system)
    for step in phase_encodes:
        seq.add_block(rf, gz)
        seq.add_block(prewinder, pp.scale_grad(gy_pre, step), gz_reph)
        seq.add_block(gx, adc)
        seq.add_block(
            pp.make_delay(
                pp.round_to_raster(
                    REPETITION_TIME - played, system.block_duration_raster
                )
            )
        )
    return seq


seq = gradient_echo()

ok, errors = seq.check_timing()
print(
    f"timing {ok}, {seq.num_blocks} blocks, "
    f"{seq.duration()[0]:.3f} s for {MATRIX} lines"
)

# %%
# Sequence diagram and k-space
# ----------------------------
#
# One repetition, drawn against the others it repeats.

seq.paper_plot(tr=1)

# %%
# The phase encodes make the raster; the prewinder and the ramps carry k-space
# between the lines and are not sampled.

pp.plot.plot_kspace(seq, plane="xy")

# %%
# Asymmetric echo
# ---------------
#
# The prewinder decides how much of k-space the readout covers before the echo.
# Carrying it only part of the way to the edge and shortening the readout by
# the same amount keeps the far edge of the line where it was, so the
# resolution is unchanged, and removes samples from the near side, which a
# partial-Fourier reconstruction then has to supply from conjugate symmetry.
# The echo time falls by the time those samples would have taken.
#
# A sample advances k-space by :math:`1/\mathrm{FOV}`, so the prewinder has to
# cancel the ramp of the readout gradient plus one such step per sample taken
# before the echo. The half step beside them is the offset from the start of
# the window to the centre of its first sample, and putting it in the prewinder
# is what lands the echo on a sample rather than between two.

HALF_LINE = MATRIX // 2
FRACTIONS = (1.0, 0.75, 0.5, 0.25)


def asymmetric_readout(fraction):
    """The readout, its window and its prewinder, with a partial near side."""
    before = round(fraction * HALF_LINE)
    samples = before + HALF_LINE
    readout = pp.make_trapezoid(
        channel="x",
        amplitude=gx.amplitude,
        flat_time=pp.round_to_raster(samples * dwell, system.grad_raster_time),
        system=system,
    )
    window = pp.make_adc(
        num_samples=samples, dwell=dwell, delay=readout.rise_time, system=system
    )
    ramp_area = readout.amplitude * readout.rise_time / 2
    prewinder = pp.make_trapezoid(
        channel="x",
        area=-(ramp_area + (before + 0.5) / FOV),
        duration=pp.calc_duration(gx_pre),
        system=system,
    )
    return readout, window, prewinder


def one_repetition(fraction):
    """A single repetition, at the largest phase encode."""
    readout, window, prewinder = asymmetric_readout(fraction)
    seq = pp.Sequence(system=system)
    seq.add_block(rf, gz)
    seq.add_block(prewinder, gy_pre, gz_reph)
    seq.add_block(readout, window)
    return seq


measured = []
for fraction in FRACTIONS:
    k_adc, _, t_excitation, _, t_adc = one_repetition(fraction).calculate_kspacePP()
    echo = int(np.argmin(np.abs(k_adc[0])))
    measured.append(
        {
            "fraction": fraction,
            "sample": echo,
            "samples": k_adc.shape[1],
            "echo_time": t_adc[echo] - t_excitation[0],
            "kx": 2 * k_adc[0] * FOV / MATRIX,
        }
    )

# sphinx_gallery_start_ignore
print(f"\n{'near side':>10}  {'samples':>9}  {'echo at':>9}  {'echo time':>12}")
for row in measured:
    print(
        f"{row['fraction']:10.2f}  {row['samples']:9d}  {row['sample']:9d}  "
        f"{1e3 * row['echo_time']:9.2f} ms"
    )

figure, (trajectory_axis, echo_axis) = plt.subplots(
    1, 2, figsize=(PAGE_WIDTH, 3.2), width_ratios=(1.4, 1.0)
)
for row in measured:
    trajectory_axis.plot(
        1e3 * np.arange(row["samples"]) * dwell,
        row["kx"],
        lw=1.2,
        label=f"{row['fraction']:.2f}",
    )
trajectory_axis.axhline(0.0, color="0.5", lw=0.8, zorder=0)
trajectory_axis.set_xlabel("time from the first sample (ms)")
trajectory_axis.set_ylabel(r"$k_x$ / $k_\mathrm{max}$")
trajectory_axis.legend(
    frameon=False,
    title="near side of the line, acquired",
    loc="upper left",
    bbox_to_anchor=(0.0, 1.30),
    ncols=4,
    fontsize=9,
    title_fontsize=9,
)
echo_axis.plot(
    [row["fraction"] for row in measured],
    [1e3 * row["echo_time"] for row in measured],
    "o-",
    lw=1.2,
    ms=5,
)
echo_axis.set_xlabel("near side of the line, acquired")
echo_axis.set_ylabel("echo time (ms)")
figure.tight_layout(rect=(0, 0, 1, 0.88))
# sphinx_gallery_end_ignore

# %%
# Every line reaches the same :math:`+k_\mathrm{max}`, so all four have the
# resolution the matrix prescribes; they differ in how far the near side is
# measured and in when the echo occurs. A quarter of the near side costs three
# quarters of the samples on that side and buys the echo time the figure
# reports.
#
# The repetition time is unchanged throughout, so what a shorter echo time
# buys here is less time for the signal to decay in before it is measured, not
# a shorter scan. Shortening the scan is the subject of
# :doc:`/generated/gallery/03-gre-to-epi/02_segmented`.
