r"""
===============
Readout modules
===============

The scope of this notebook is to replace the hand-built readout of the first
sections with the module that designs one, and to use the two prescriptions the
earlier pages solved by hand — a partial echo and a multi-echo train — as the
check that the module reaches the same answers and states them.

A readout module is given the system, the pulse that opens the repetition and
a prescription, and solves the gradients, the acquisition window and the timing
from them. What it does not do is decide the order the lines are acquired in;
that belongs to the loop, which is the subject of
:doc:`/generated/gallery/05-sequence-modules/03_sequence_app`.

Outline:

#. **What a module holds.** The prescription in, and the events, the timing and
   the sampling that come out.
#. **One repetition.** The blocks the module publishes, played into a sequence.
#. **Shortest echo time against partial echo.** The same measurement as
   :doc:`/generated/gallery/01-pulseq-basics/03_gradient_echo`, from the
   module's own solution.
#. **Monopolar against bipolar trains.** The two ways to read several echoes,
   and what each costs.

What a module is, and why the design is split this way, is described in
:doc:`/explanations/design/sequence-module`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# What a module holds
# -------------------
#
# The excitation module designs the pulse, its selection gradient and the
# rephaser; the readout module takes those and the prescription. The echo time
# is left unset, which asks for the shortest the prescription admits, and the
# requested receiver bandwidth is what the rasters allow rather than what was
# asked for — the module reports what it achieved.

import pypulseqpp as pp
import pypulseqpp.sequences as design

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=20e-6,
    adc_dead_time=10e-6,
)

FOV = 220e-3
MATRIX = 128
THICKNESS = 5e-3
FLIP_ANGLE_DEG = 12.0

excitation = design.SpatialSelectiveExcitation(
    system, FLIP_ANGLE_DEG, THICKNESS, duration_s=3e-3, time_bw_product=4.0
)
readout = design.LineReadout2D(
    system,
    excitation.rf,
    excitation.gz,
    excitation.gz_reph,
    fov=(FOV, FOV),
    matrix=(MATRIX, MATRIX),
    te=None,
    readout_bandwidth_hz=250e3,
    spoiling_cycles=4.0,
)

print(
    f"echo time {1e3 * readout.echo_time:.3f} ms, "
    f"module {1e3 * readout.duration:.3f} ms\n"
    f"{readout.n_samples} samples at {readout.bandwidth_hz / 1e3:.1f} kHz, "
    f"echo on sample {readout.center_sample}, "
    f"line spacing {readout.delta_kx:.2f} 1/m"
)

# %%
# One repetition
# --------------
#
# ``blocks`` is the module's playout in order, as tuples of events. A loop adds
# them to a sequence, scaling the phase-encode template to the line it is
# acquiring; here the largest step is played, and the pulse's block is added
# first because the module is the readout half of the repetition.

seq = pp.Sequence(system=system)
seq.add_block(excitation.rf, excitation.gz)
for block in readout.blocks:
    seq.add_block(*block)

ok, errors = seq.check_timing()
print(f"timing {ok}, {seq.num_blocks} blocks, {1e3 * seq.duration()[0]:.3f} ms")

readout.paper_plot()

# %%
# Shortest echo time against partial echo
# ---------------------------------------
#
# ``partial_echo`` is the fraction of the full echo acquired, and truncates the
# samples before it. The shortest echo time follows, as it did when the same
# readout was built by hand: the samples that are no longer taken are the ones
# that stood between the excitation and the echo.

FRACTIONS = (1.0, 0.875, 0.75, 0.625, 0.5625)


def solved(partial_echo=1.0, bandwidth_hz=250e3, **prescription):
    """One readout module, solved for the shortest echo time."""
    return design.LineReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        excitation.gz_reph,
        fov=(FOV, FOV),
        matrix=(MATRIX, MATRIX),
        te=None,
        partial_echo=partial_echo,
        readout_bandwidth_hz=bandwidth_hz,
        spoiling_cycles=4.0,
        **prescription,
    )


partial = [
    {
        "fraction": fraction,
        "module": solved(partial_echo=fraction),
    }
    for fraction in FRACTIONS
]

# sphinx_gallery_start_ignore
print(
    f"\n{'partial echo':>13}  {'samples':>8}  {'echo on':>8}  {'achieved':>12}  "
    f"{'TE':>10}  {'module':>10}"
)
for row in partial:
    module = row["module"]
    print(
        f"{row['fraction']:13.4f}  {module.n_samples:8d}  "
        f"{module.center_sample:8d}  {module.bandwidth_hz / 1e3:9.1f} kHz  "
        f"{1e3 * module.echo_time:7.3f} ms  {1e3 * module.duration:7.3f} ms"
    )

figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.2))
axis.plot(
    [row["fraction"] for row in partial],
    [1e3 * row["module"].echo_time for row in partial],
    "o-",
    lw=1.2,
    ms=5,
    label="shortest echo time",
)
axis.plot(
    [row["fraction"] for row in partial],
    [1e3 * row["module"].duration for row in partial],
    "s--",
    lw=1.2,
    ms=5,
    label="module duration",
)
axis.set_xlabel("fraction of the echo acquired")
axis.set_ylabel("time (ms)")
axis.legend(
    frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.28), ncols=2, fontsize=9
)
figure.tight_layout(rect=(0, 0, 1, 0.86))
# sphinx_gallery_end_ignore

# %%
# The sample the echo lands on moves with the fraction, and the echo time falls
# with it.
#
# The achieved bandwidth is not the requested one at every fraction, and not
# the same one at every fraction either. A dwell time lies on the ADC raster
# and an acquisition window on the gradient raster, and whether a given dwell
# satisfies both depends on how many samples are taken: at 80 samples the
# requested 250 kHz lands on both rasters and is used, and at the neighbouring
# counts the fastest rate that does is 100 kHz. That is why the module duration
# does not fall monotonically while the echo time does, and why the module
# reports the rate it achieved rather than the one it was asked for.

# %%
# Monopolar against bipolar trains
# --------------------------------
#
# ``n_echoes`` asks for a train. ``flyback`` decides how it is played: a
# monopolar train rewinds between the echoes so that every one is read in the
# same direction, and a bipolar train alternates the readout sign, as the
# hand-built train of
# :doc:`/generated/gallery/03-gre-to-epi/01_multi_echo` did. The bipolar train
# is shorter by the rewinders it does not play, and reads its even echoes
# backwards.

ECHOES = 4

trains = {
    "monopolar": solved(n_echoes=ECHOES, flyback=True),
    "bipolar": solved(n_echoes=ECHOES, flyback=False),
}

# sphinx_gallery_start_ignore
print(f"\n{'train':>12}  {'echo spacing':>14}  {'first echo':>12}  {'module':>11}")
for name, module in trains.items():
    print(
        f"{name:>12}  {1e3 * module.echo_spacing:11.3f} ms  "
        f"{1e3 * module.echo_time:9.3f} ms  {1e3 * module.duration:8.3f} ms"
    )

figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0), sharey=True)
for axis, (name, module) in zip(axes, trains.items(), strict=True):
    k_adc, _, _, _, t_adc = module.calculate_kspace()
    samples = k_adc[0] * 2 * FOV / MATRIX
    per_echo = samples.size // ECHOES
    for echo in range(ECHOES):
        window = slice(echo * per_echo, (echo + 1) * per_echo)
        axis.plot(1e3 * t_adc[window], samples[window], lw=1.4)
    axis.set_title(name, fontsize=10)
    axis.set_xlabel("time within the module (ms)")
axes[0].set_ylabel(r"$k_x$ / $k_\mathrm{max}$")
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Every echo of the monopolar train is traversed in the same direction and the
# gaps between them are the rewinders; the bipolar train has no gaps and every
# second echo runs backwards. Which to prefer is the trade the earlier page
# measured: the bipolar train is shorter, and any delay between the gradient
# and the acquisition enters it as a difference between the odd and the even
# echoes rather than as a shift common to all of them.
