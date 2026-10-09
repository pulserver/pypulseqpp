r"""
=========================
Multi-echo gradient echo
=========================

A multi-echo gradient echo follows the first readout gradient with further
readout gradients of alternating polarity, each with its own ADC event. Each
gradient traverses the same k-space line, in the opposite direction to the
one before it, so the train forms one echo per gradient and needs no
rewinder between them. The echoes sample the line at increasing echo times,
which is what lets you map :math:`T_2^*` from a single repetition.

In this Tour you build the repetition and then ask how many echoes fit in it.
The echo spacing is the duration of one readout gradient, and it depends on
the receiver bandwidth through the ramps. You sweep the bandwidth, read the
echo spacing and the number of echoes that fit in the repetition time, and
see where the gradient amplitude limit stops the sweep.

**Prerequisites:** lessons 1 and 2 of the :doc:`course </examples/course>`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# A train of readouts
# -------------------
#
# A readout gradient traverses one k-space line from one end to the other. A
# second gradient of the opposite polarity traverses it back, so a train of
# them needs no rewinder between the echoes and forms one echo per gradient.
# The even echoes are therefore acquired in reverse order along the line.
# The readout is designed as in lesson 1, with the amplitude set so that one
# sample advances k-space by :math:`1/\mathrm{FOV}`.
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
FLIP_ANGLE_DEG = 20.0
REPETITION_TIME = 40e-3
BANDWIDTH_HZ = 250e3
ECHOES = 6

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


def readout(bandwidth_hz):
    """The readout gradient, its acquisition window and its prewinder.

    The dwell time is put on the ADC raster and the flat top on the gradient
    raster, which are different rasters, so the flat top is the acquisition
    window rounded up rather than equal to it. The amplitude is set so that a
    sample advances k-space by ``1 / FOV`` however long the window is.
    """
    dwell = pp.round_to_raster(1.0 / bandwidth_hz, system.adc_raster_time)
    acquisition = MATRIX * dwell
    raster = system.grad_raster_time
    gx = pp.make_trapezoid(
        channel="x",
        amplitude=MATRIX / FOV / acquisition,
        flat_time=raster * np.ceil(acquisition / raster),
        system=system,
    )
    adc = pp.make_adc(
        num_samples=MATRIX, dwell=dwell, delay=gx.rise_time, system=system
    )
    # The prewinder cancels the ramp, one step per sample before the echo and
    # the half step to the centre of the first sample.
    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-(gx.amplitude * gx.rise_time / 2 + (MATRIX / 2 + 0.5) / FOV),
        duration=1e-3,
        system=system,
    )
    return gx, adc, gx_pre


gx, adc, gx_pre = readout(BANDWIDTH_HZ)
gy_pre = pp.make_trapezoid(
    channel="y", area=MATRIX / (2 * FOV), duration=1e-3, system=system
)

print(
    f"dwell {1e6 * adc.dwell:.1f} us, "
    f"readout {1e3 * pp.calc_duration(gx):.3f} ms, "
    f"amplitude {1e3 * gx.amplitude / 42.576e6:.2f} mT/m, "
    f"slew {gx.amplitude / gx.rise_time / 42.576e6:.0f} T/m/s"
)

# %%
# One repetition
# --------------
#
# The train replaces the single readout block. The gradients of the train are
# played back to back, so the echo spacing is the duration of one readout
# gradient, ramps included.


def multi_echo(echoes, bandwidth_hz=BANDWIDTH_HZ, lines=MATRIX):
    """A multi-echo gradient echo with the given train length."""
    gx, adc, gx_pre = readout(bandwidth_hz)
    spoiler = pp.make_crusher(4.0, FOV / MATRIX, channel="z", system=system)[0]
    played = (
        pp.calc_duration(rf, gz)
        + pp.calc_duration(gx_pre, gy_pre, gz_reph)
        + echoes * pp.calc_duration(gx)
        + pp.calc_duration(spoiler)
    )
    seq = pp.Sequence(system=system)
    for step in np.linspace(-1.0, 1.0, lines, endpoint=False):
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, pp.scale_grad(gy_pre, step), gz_reph)
        for echo in range(echoes):
            seq.add_block(pp.scale_grad(gx, (-1.0) ** echo), adc)
        seq.add_block(spoiler)
        seq.add_block(
            pp.make_delay(
                pp.round_to_raster(
                    REPETITION_TIME - played, system.block_duration_raster
                )
            )
        )
    return seq


seq = multi_echo(ECHOES)

ok, errors = seq.check_timing()
print(
    f"timing {ok}, {seq.num_blocks} blocks, "
    f"{seq.duration()[0]:.3f} s for {ECHOES} echoes on each of {MATRIX} lines"
)

seq.paper_plot(tr=1)

# %%
# Echo spacing against receiver bandwidth
# ---------------------------------------
#
# The readout gradient has to cover the same area whatever the bandwidth, so a
# shorter flat top is a proportionally stronger gradient, until the amplitude
# limit is reached and the flat top can shorten no further. The echo spacing
# follows the flat top, and the train that fits in a repetition follows the
# echo spacing.

# The dwell time is the quantity the ADC raster quantizes, so the sweep is over
# dwell times and the bandwidth is read from them.
DWELLS = np.array([16e-6, 12e-6, 10e-6, 8e-6, 6e-6, 4e-6, 2e-6])
BANDWIDTHS = 1.0 / DWELLS

overhead = (
    pp.calc_duration(rf, gz)
    + pp.calc_duration(gx_pre, gy_pre, gz_reph)
    + pp.calc_duration(
        pp.make_crusher(4.0, FOV / MATRIX, channel="z", system=system)[0]
    )
)

spacing = []
for bandwidth in BANDWIDTHS:
    try:
        gradient = readout(bandwidth)[0]
    except ValueError:
        spacing.append({"bandwidth": bandwidth, "realizable": False})
        continue
    spacing.append(
        {
            "bandwidth": bandwidth,
            "realizable": True,
            "spacing": pp.calc_duration(gradient),
            "ramps": gradient.rise_time + gradient.fall_time,
            "amplitude": gradient.amplitude / 42.576e6,
            "echoes": int((REPETITION_TIME - overhead) // pp.calc_duration(gradient)),
        }
    )

realizable = [row for row in spacing if row["realizable"]]

# sphinx_gallery_start_ignore
print(
    f"\n{'bandwidth':>12}  {'amplitude':>12}  {'spacing':>10}  {'ramps':>7}  "
    f"{'echoes':>7}"
)
for row in spacing:
    if not row["realizable"]:
        print(f"{row['bandwidth'] / 1e3:9.0f} kHz  {'beyond the amplitude limit':>40}")
        continue
    print(
        f"{row['bandwidth'] / 1e3:9.0f} kHz  {1e3 * row['amplitude']:9.2f} mT/m  "
        f"{1e3 * row['spacing']:7.3f} ms  "
        f"{100 * row['ramps'] / row['spacing']:6.0f}%  {row['echoes']:7d}"
    )

figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4), layout="constrained")
axis.plot(
    [row["bandwidth"] / 1e3 for row in realizable],
    [1e3 * row["spacing"] for row in realizable],
    "o-",
    lw=1.2,
    ms=5,
    label="echo spacing (ms)",
)
axis.plot(
    [row["bandwidth"] / 1e3 for row in realizable],
    [1e3 * row["amplitude"] for row in realizable],
    "s--",
    lw=1.2,
    ms=5,
    label="readout amplitude (mT/m)",
)
axis.axhline(
    1e3 * system.max_grad / 42.576e6,
    color="0.5",
    ls=":",
    lw=1.0,
    label="amplitude limit (mT/m)",
)
axis.set_xlabel("receiver bandwidth (kHz)")
axis.set_xscale("log")
axis.minorticks_off()
axis.set_xticks(BANDWIDTHS / 1e3)
axis.set_xticklabels([f"{b / 1e3:.0f}" for b in BANDWIDTHS])
figure.legend(loc="outside right upper")
# sphinx_gallery_end_ignore

# %%
# The acquisition window falls as the reciprocal of the bandwidth; the echo
# spacing does not. A shorter window at the same k-space extent is a stronger
# gradient, and a stronger gradient takes longer to ramp, at both ends of every
# echo. Over the sweep the window shortens fourfold and the spacing by a factor
# of little more than two, with the ramps growing from a sixteenth of the echo
# spacing to nearly half of it. Beyond the last point the amplitude the readout
# would need is above the limit, and the factory raises an error rather than
# lengthening the flat top.
#
# The shorter spacing reduces the signal-to-noise ratio. The noise in a sample
# grows as the square root of the bandwidth, so the fourfold bandwidth of the
# sweep halves the signal-to-noise ratio of each echo, in exchange for the
# number of echoes and a shorter shortest echo time.
#
# As a spec
# ---------
#
# What this Tour built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Write a 2D multi-echo gradient echo with pypulseqpp: 220 mm FOV,
#    128 matrix, 5 mm slice, 20 degree sinc excitation, TR 40 ms. After
#    the excitation and the readout and phase prewinders, play a train of
#    6 readout gradients of alternating polarity at 250 kHz receiver
#    bandwidth, each with its own ADC event and no rewinder between them,
#    then a z spoiler. Check the timing. Sweep the dwell time from 16 us
#    to 2 us and report the echo spacing, the ramp fraction and the
#    number of echoes that fit in TR, stopping where the readout
#    amplitude exceeds the system limit.
