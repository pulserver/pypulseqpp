r"""
========================
Segmented echo planar
========================

The scope of this notebook is to put a phase-encode blip between the echoes of
the train of the previous page, so that one excitation acquires several k-space
lines instead of the same line several times. The number of excitations the
matrix is divided over is then a free parameter, and it decides both the scan
time and how far off-resonance displaces the image.

The observable is that trade-off: the bandwidth per pixel along the
phase-encode direction, which segmentation raises in proportion to the number
of shots, against the number of excitations the scan then costs.

Outline:

#. **Blips between the echoes.** One blip per echo, of the area the
   segmentation calls for.
#. **One shot.** The blocks of a shot, and the lines it acquires.
#. **The raster the shots build.** Which lines each excitation contributes.
#. **Distortion against shot count.** Phase-encode bandwidth and displacement,
   against the excitations they cost.

The single-shot end of this trade-off is
:doc:`/generated/gallery/03-gre-to-epi/03_epi`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# Blips between the echoes
# ------------------------
#
# Within a shot, consecutive echoes are separated by as many k-space lines as
# there are shots, so that the shots interleave and together cover every line.
# :func:`~pypulseqpp.make_phase_blip` takes that number of lines and the field
# of view rather than an area, and solves the shortest gradient that delivers
# it.

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
MATRIX = 64
THICKNESS = 5e-3
FLIP_ANGLE_DEG = 90.0
REPETITION_TIME = 100e-3
DWELL = 4e-6
SHOTS = 4

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

# The readout of the previous page, at a fixed dwell time.
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

blip = pp.make_phase_blip(channel="y", fov=FOV, steps=SHOTS, system=system)

print(
    f"echo spacing {1e3 * pp.calc_duration(gx):.3f} ms, "
    f"blip {1e6 * pp.calc_duration(blip):.0f} us for {SHOTS} lines, "
    f"readout {1e3 * gx.amplitude / 42.576e6:.1f} mT/m "
    f"rising in {1e6 * gx.rise_time:.0f} us"
)

# %%
# One shot
# --------
#
# A shot is the excitation, the prewinders, and then one block per echo. The
# blip is played in the same block as the readout gradient it follows, on the
# other axis, which is what keeps the echo spacing equal to the duration of one
# readout gradient. The shot's phase-encode prewinder carries k-space to the
# line that shot begins on.


def segmented(shots, lines=MATRIX):
    """A segmented echo planar acquisition covering ``lines`` k-space lines."""
    step = pp.make_phase_blip(channel="y", fov=FOV, steps=shots, system=system)
    train = lines // shots
    seq = pp.Sequence(system=system)
    for shot in range(shots):
        start = pp.make_trapezoid(
            channel="y", area=(-lines / 2 + shot) / FOV, duration=5e-4, system=system
        )
        played = (
            pp.calc_duration(rf, gz)
            + pp.calc_duration(gx_pre, start, gz_reph)
            + train * pp.calc_duration(gx)
        )
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, start, gz_reph)
        for echo in range(train):
            readout = pp.scale_grad(gx, (-1.0) ** echo)
            # The blip advances k-space before every echo but the first, and
            # plays inside the rise of the readout gradient it shares a block
            # with, so it is complete before the acquisition window opens.
            if echo == 0:
                seq.add_block(readout, adc)
            else:
                seq.add_block(readout, step, adc)
        seq.add_block(
            pp.make_delay(
                pp.round_to_raster(
                    REPETITION_TIME - played, system.block_duration_raster
                )
            )
        )
    return seq


seq = segmented(SHOTS)

ok, errors = seq.check_timing()
print(
    f"timing {ok}, {seq.num_blocks} blocks, {MATRIX // SHOTS} echoes per shot, "
    f"{seq.duration()[0]:.3f} s"
)

seq.paper_plot(tr=1)

# %%
# The raster the shots build
# --------------------------
#
# Each excitation contributes every fourth line, and the four together cover
# the matrix, each line once.

k_adc = seq.calculate_kspacePP()[0]
# Every sample of one echo shares that echo's phase-encode line.
lines = np.round(k_adc[1] * FOV).astype(int)
covered, samples = np.unique(lines, return_counts=True)
print(
    f"{covered.size} lines from {covered.min()} to {covered.max()}, "
    f"each acquired {(samples // MATRIX).min()} to {(samples // MATRIX).max()} times"
)

pp.plot.plot_kspace(seq, color_by="shot", plane="xy")

# %%
# Distortion against shot count
# -----------------------------
#
# Off-resonance displaces a voxel along the phase-encode direction by the ratio
# of its offset to the bandwidth per pixel in that direction. That bandwidth is
# not the receiver bandwidth: the phase-encode direction is traversed one line
# per echo, so one line takes an echo spacing divided by the number of shots,
# and the bandwidth per pixel is
#
# .. math::
#
#     \mathrm{BW}_\mathrm{pe} = \frac{S}{N\,\mathrm{ESP}} ,
#
# with :math:`S` shots, :math:`N` lines and an echo spacing
# :math:`\mathrm{ESP}`. It is smaller than the receiver bandwidth per pixel by
# the number of echoes in a shot, which is why an echo planar image is
# distorted along the phase-encode direction and not along the readout.

OFF_RESONANCE_HZ = 100.0
COUNTS = (1, 2, 4, 8, 16)

echo_spacing = pp.calc_duration(gx)

trade_off = []
for shots in COUNTS:
    phase_bandwidth = shots / (MATRIX * echo_spacing)
    trade_off.append(
        {
            "shots": shots,
            "echoes": MATRIX // shots,
            "train": (MATRIX // shots) * echo_spacing,
            "phase_bandwidth": phase_bandwidth,
            "displacement": OFF_RESONANCE_HZ / phase_bandwidth,
            "scan": shots * REPETITION_TIME,
        }
    )

# sphinx_gallery_start_ignore
print(
    f"\n{'shots':>6}  {'echoes':>7}  {'train':>10}  {'BW per pixel':>15}  "
    f"{'displacement':>13}  {'per slice':>10}"
)
for row in trade_off:
    print(
        f"{row['shots']:6d}  {row['echoes']:7d}  {1e3 * row['train']:7.2f} ms  "
        f"{row['phase_bandwidth']:12.1f} Hz  {row['displacement']:10.2f} px  "
        f"{1e3 * row['scan']:7.0f} ms"
    )

figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.66, 3.2))
axis.plot(
    [row["shots"] for row in trade_off],
    [row["displacement"] for row in trade_off],
    "o-",
    lw=1.2,
    ms=5,
    label=f"displacement at {OFF_RESONANCE_HZ:.0f} Hz (pixels)",
)
axis.plot(
    [row["shots"] for row in trade_off],
    [1e3 * row["train"] for row in trade_off],
    "s--",
    lw=1.2,
    ms=5,
    label="echo train (ms)",
)
axis.set_xscale("log", base=2)
axis.set_yscale("log")
axis.set_xticks(COUNTS)
axis.set_xticklabels([str(count) for count in COUNTS])
axis.set_xlabel("shots")
secondary = axis.twinx()
secondary.plot(
    [row["shots"] for row in trade_off],
    [1e3 * row["scan"] for row in trade_off],
    "^:",
    lw=1.2,
    ms=5,
    color="0.45",
    label="time per slice (ms)",
)
secondary.set_yscale("log")
secondary.set_ylabel("time per slice (ms)")
handles = axis.get_legend_handles_labels()
extra = secondary.get_legend_handles_labels()
axis.legend(
    handles[0] + extra[0],
    handles[1] + extra[1],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.0, 1.40),
    fontsize=9,
)
figure.tight_layout(rect=(0, 0, 1, 0.80))
# sphinx_gallery_end_ignore

# %%
# Displacement and echo train fall as the reciprocal of the shot count, and the
# scan time rises in proportion to it, so the segmentation is a straight
# exchange of time for geometric fidelity. The other terms of it are the echo
# spacing, which the previous page shortened with the receiver bandwidth and
# which enters the displacement in the same way, and the number of lines, which
# the prescription fixes.
#
# Two things segmentation does not fix. Each shot is excited separately, so any
# motion or phase change between them appears as an inconsistency between
# interleaved lines rather than as blurring within one; and the displacement it
# reduces is a property of the trajectory, not of the reconstruction, so an
# image acquired in one shot is distorted whatever is done to it afterwards.
