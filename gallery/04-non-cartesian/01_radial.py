r"""
===============
Radial sampling
===============

The Cartesian gradient echo of
:doc:`/generated/gallery/01-pulseq-basics/03_gradient_echo` changes the
acquired line with a phase encode. This lesson replaces the phase encode with
a rotation of the readout gradient itself, so that every repetition acquires a
spoke through the centre of k-space. It establishes how many spokes such an
acquisition requires and what ordering them by the golden angle changes.

The measured quantity is the azimuthal gap between neighbouring spokes at the
edge of k-space, computed from the sampling locations of the sequence and
compared with the sample spacing along a spoke that the prescription sets.

Learning objectives
-------------------

After this lesson, you should be able to:

- rotate a readout gradient and its prewinder into the imaging plane;
- build one repetition per spoke from the Cartesian gradient echo;
- measure the azimuthal gap at the edge of k-space from the sampling
  locations, and relate it to the Nyquist spoke count
  :math:`P = \tfrac{\pi}{2} N`;
- compare uniform and golden-angle orderings of truncated acquisitions.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# A rotated readout
# -----------------
#
# A spoke runs from one edge of k-space through the centre to the other, so the
# prewinder has half the readout area as it does on a Cartesian line, and
# the echo is at the middle of the acquisition window. The pair is then rotated
# about the slice axis by the angle of the spoke, which
# :func:`~pypulseqpp.rotate` does by resolving each gradient onto the two
# in-plane axes.

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
REPETITION_TIME = 10e-3
DWELL = 10e-6

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

acquisition = MATRIX * DWELL
raster = system.grad_raster_time
gx = pp.make_trapezoid(
    channel="x",
    amplitude=MATRIX / FOV / acquisition,
    flat_time=raster * np.ceil(acquisition / raster),
    system=system,
)
adc = pp.make_adc(num_samples=MATRIX, dwell=DWELL, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(
    channel="x",
    area=-(gx.amplitude * gx.rise_time / 2 + (MATRIX / 2 + 0.5) / FOV),
    duration=5e-4,
    system=system,
)
spoiler = pp.make_crusher(4.0, FOV / MATRIX, channel="z", system=system)[0]

# Both in-plane axes have a gradient for every spoke, but the vector amplitude
# of a spoke is the readout amplitude at every angle, and each axis plays its
# projection of it, so the per-axis limit binds where a spoke lies along an
# axis and nowhere else.
print(
    f"readout {1e3 * gx.amplitude / 42.576e6:.2f} mT/m, "
    f"per-axis limit {1e3 * system.max_grad / 42.576e6:.1f} mT/m"
)

# %%
# One repetition per spoke
# ------------------------
#
# The repetition is the Cartesian one with the phase encode removed and the
# readout rotated. Both in-plane axes have a gradient for every spoke.

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0)) / 2.0


def radial(spokes, golden=False):
    """A radial acquisition of the given number of spokes."""
    if golden:
        angles = (np.arange(spokes) * GOLDEN_ANGLE) % np.pi
    else:
        angles = np.arange(spokes) * np.pi / spokes
    played = (
        pp.calc_duration(rf, gz)
        + pp.calc_duration(gx_pre, gz_reph)
        + pp.calc_duration(gx)
        + pp.calc_duration(spoiler)
    )
    recovery = pp.make_delay(
        pp.round_to_raster(REPETITION_TIME - played, system.block_duration_raster)
    )
    seq = pp.Sequence(system=system)
    for angle in angles:
        seq.add_block(rf, gz)
        seq.add_block(*pp.rotate(gx_pre, angle=angle, axis="z"), gz_reph)
        seq.add_block(*pp.rotate(gx, angle=angle, axis="z"), adc)
        seq.add_block(spoiler)
        seq.add_block(recovery)
    return seq, angles


SPOKES = 128

seq, angles = radial(SPOKES)

ok, errors = seq.check_timing()
print(
    f"timing {ok}, {seq.num_blocks} blocks, {SPOKES} spokes, {seq.duration()[0]:.3f} s"
)

seq.paper_plot(tr=1)

# %%
# The trajectory
# --------------
#
# Every spoke passes through the centre of k-space, so the centre is sampled
# once per repetition and the periphery only where a spoke reaches it.

pp.plot.plot_kspace(seq, color_by="shot", plane="xy", show_trajectory=False)

# %%
# Spokes against azimuthal gap
# ----------------------------
#
# Along a spoke the samples are :math:`1/\mathrm{FOV}` apart, as the field of
# view requires. Between spokes the spacing grows with the distance
# from the centre, and at the edge it is the azimuthal arc between neighbouring
# spokes,
#
# .. math::
#
#     \Delta k_\varphi = \frac{\pi}{P} \, k_\mathrm{max} ,
#
# for :math:`P` spokes over half a turn. Requiring it to be no larger than the
# radial spacing gives the familiar :math:`P = \tfrac{\pi}{2} N`: a radial
# acquisition needs more spokes than a Cartesian acquisition of the same matrix
# needs lines.
#
# The gap is measured from the sampling locations the sequence produces.

COUNTS = (32, 64, 128, 201, 256)

radial_spacing = 1.0 / FOV


def spoke_angles(sequence, spokes):
    """The angle of every spoke, from the samples the analysis reports."""
    sampled = sequence.calculate_kspacePP()[0][:2].reshape(2, spokes, MATRIX)
    outermost = sampled[:, :, -1]
    return np.arctan2(outermost[1], outermost[0])


def largest_gap(sampled_angles):
    """The arc between neighbouring spokes at the edge, in 1/m."""
    sorted_angles = np.sort(sampled_angles % np.pi)
    steps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + np.pi]]))
    return steps.max() * MATRIX / (2 * FOV)


uniform = []
for count in COUNTS:
    sampled, _ = radial(count)
    uniform.append(
        {
            "spokes": count,
            "gap": largest_gap(spoke_angles(sampled, count)),
            "scan": count * REPETITION_TIME,
        }
    )

nyquist = int(np.ceil(np.pi / 2 * MATRIX))

# sphinx_gallery_start_ignore
print(f"\n{'spokes':>7}  {'gap at the edge':>18}  {'of the radial spacing':>23}")
for row in uniform:
    print(
        f"{row['spokes']:7d}  {row['gap']:15.2f} 1/m  "
        f"{row['gap'] / radial_spacing:20.2f}"
    )
print(f"radial sample spacing {radial_spacing:.2f} 1/m, Nyquist at {nyquist} spokes")

figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.2))
axis.plot(
    [row["spokes"] for row in uniform],
    [row["gap"] / radial_spacing for row in uniform],
    "o-",
    lw=1.2,
    ms=5,
    label="uniform",
)
axis.axhline(1.0, color="0.5", ls="--", lw=1.0, label="radial sample spacing")
axis.axvline(nyquist, color="0.5", ls=":", lw=1.0, label=f"{nyquist} spokes")
axis.set_xscale("log")
axis.set_yscale("log")
axis.set_xticks(COUNTS)
axis.set_xticklabels([str(count) for count in COUNTS])
axis.set_xlabel("spokes")
axis.set_ylabel("edge gap / radial spacing")
axis.legend(
    frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.32), ncols=3, fontsize=9
)
figure.tight_layout(rect=(0, 0, 1, 0.86))
# sphinx_gallery_end_ignore

# %%
# The measured gap falls as the reciprocal of the spoke count and crosses the
# radial sample spacing at the predicted count. An acquisition below it is
# undersampled at the periphery and not at the centre, which is why the
# artefact it produces is a streak from the edge of the object rather than the
# fold-over a Cartesian acquisition produces.

# %%
# Golden-angle ordering
# ---------------------
#
# Advancing the angle by :math:`\pi` times the golden ratio conjugate instead
# of by :math:`\pi/P` gives an ordering whose every prefix is nearly uniform,
# so the acquisition can be stopped, or divided into frames, at any length. The
# gap of a prefix is, however, never exactly that of the uniform ordering.

PREFIXES = np.arange(8, 257, 8)

golden, _ = radial(PREFIXES.max(), golden=True)
golden_angles = spoke_angles(golden, PREFIXES.max())
golden_gap = np.array([largest_gap(golden_angles[:count]) for count in PREFIXES])
uniform_gap = np.array(
    [largest_gap(np.arange(count) * np.pi / count) for count in PREFIXES]
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.2))
axis.plot(PREFIXES, golden_gap / uniform_gap, lw=1.4, label="golden angle")
axis.axhline(1.0, color="0.5", ls="--", lw=1.0, label="uniform")
axis.set_xlabel("spokes acquired")
axis.set_ylabel("edge gap, relative to uniform")
axis.set_ylim(bottom=0.9)
axis.legend(
    frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.26), ncols=2, fontsize=9
)
figure.tight_layout(rect=(0, 0, 1, 0.88))

print(
    f"\ngolden-angle gap, over prefixes of 8 to {PREFIXES.max()} spokes: "
    f"{(golden_gap / uniform_gap).min():.2f} to "
    f"{(golden_gap / uniform_gap).max():.2f} times the uniform gap"
)
# sphinx_gallery_end_ignore

# %%
# Every prefix is within a small factor of the uniform ordering of the same
# length, and no prefix leaves a gap of the kind a truncated uniform ordering
# would: stopping a uniform acquisition after half its spokes leaves half the
# angular range unsampled, while stopping a golden-angle one leaves the same
# range covered at half the density. The complete golden-angle set is not
# exactly uniform.
