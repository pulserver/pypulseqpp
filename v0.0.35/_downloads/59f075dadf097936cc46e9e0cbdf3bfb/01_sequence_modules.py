r"""
================
Sequence modules
================

The earlier lessons built the excitation and the readout of a repetition by
hand, with event factories. This lesson designs them with sequence modules: a
module takes the system limits and a prescription, solves the events and the
timing of one part of the repetition, and publishes them, with its timing
measured from its ``center``. The module concept, and the reason the design is
divided in this way, are described in :doc:`/explanations/sequence-design`.

The first half designs slice-selective excitations with the excitation module,
and measures how the three numbers that specify a selective pulse — flip
angle, slice thickness and time-bandwidth product — affect the slice profile,
the selection gradient and the peak :math:`B_1`, and which combinations of
them the gradient system permits. A slice-selective pulse and its selection
gradient are not independent: the gradient has to place the pulse's bandwidth
across the slice,

.. math::

    G = \frac{\mathrm{TBW}}{\gamma\, T\, \Delta z} ,

so a shorter pulse at the same thickness and the same time-bandwidth product
needs a proportionally stronger selection gradient and a proportionally larger
:math:`B_1`. The gradient amplitude limit therefore bounds the two together.

The second half replaces the hand-built readout with the readout module, and
uses two prescriptions that earlier lessons solved by hand — a partial echo
and a train of echoes — to check that the module reaches the same results and
reports them. The order in which lines are acquired is not part of a module;
it belongs to the loop of the sequence function of the next lesson,
:doc:`/generated/gallery/05-sequence-modules/03_sequence_function`. The
previous lesson, :doc:`/generated/gallery/04-non-cartesian/01_radial`, played
its readout by hand.

Learning objectives
-------------------

After this lesson, you should be able to:

- design a slice-selective excitation with the excitation module, simulate
  its slice profile and measure its transition width and ripple;
- relate the time-bandwidth product and the pulse duration to the profile,
  the selection gradient amplitude and the peak :math:`B_1`, and identify the
  designs the gradient amplitude limit permits;
- design a readout with the readout module, read its events, timing, sampling
  and achieved receiver bandwidth, and play its blocks for one repetition;
- relate the partial-echo fraction to the shortest echo time, and explain why
  the achieved receiver bandwidth depends on the number of samples;
- compare monopolar and bipolar multi-echo trains.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 7.8  # inches, the width of the documentation column


def profile_figure(designs, title, legend):
    """Pulse envelopes beside the slice profiles they produce."""
    figure, (envelope_axis, profile_axis) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.4), layout="constrained"
    )
    for label, design_ in designs.items():
        envelope_axis.plot(
            1e3 * design_["time"], design_["envelope"], lw=1.2, label=label
        )
        profile_axis.plot(1e3 * design_["position"], design_["profile"], lw=1.2)
    envelope_axis.set_xlabel("time (ms)")
    envelope_axis.set_ylabel("$|B_1|$ (Hz)")
    figure.legend(
        *envelope_axis.get_legend_handles_labels(),
        title=legend,
        loc="outside lower center",
    )
    profile_axis.axvspan(
        -0.5e3 * THICKNESS, 0.5e3 * THICKNESS, color="0.5", alpha=0.15, lw=0, zorder=0
    )
    profile_axis.set_xlim(-12, 12)
    profile_axis.set_xlabel("position (mm)")
    profile_axis.set_ylabel(r"$|M_{xy}|$, normalised")
    profile_axis.set_title("nominal slice in grey")
    figure.suptitle(title, y=1.02)
    return figure


def feasibility_figure(grid, boundary_tbw, boundary_duration):
    """The designs the gradient system admits, against the amplitude bound."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.8), layout="constrained")
    for entry in grid:
        axis.plot(
            entry["tbw"],
            1e3 * entry["duration"],
            "o" if entry["feasible"] else "x",
            ms=7,
            color="C0" if entry["feasible"] else "C7",
        )
    axis.plot(
        boundary_tbw,
        1e3 * boundary_duration,
        "-",
        color="0.5",
        label=r"$T = \mathrm{TBW} / (\gamma\, \Delta z\, G_\mathrm{max})$",
    )
    axis.plot([], [], "o", color="C0", label="designed")
    axis.plot([], [], "x", color="C7", label="rejected")
    axis.set_xlabel("time-bandwidth product")
    axis.set_ylabel("pulse duration (ms)")
    figure.legend(loc="outside right upper")
    return figure


def table(rows, first):
    """Print one line per design."""
    print(
        f"{first:>10}  {'gradient':>12}  {'transition':>12}  {'passband':>10}  "
        f"{'stopband':>10}  {'peak B1':>10}"
    )
    for row in rows:
        print(
            f"{row[first]:10.1f}  {1e3 * row['gradient']:9.2f} mT/m  "
            f"{1e3 * row['transition']:9.3f} mm  {row['passband']:10.4f}  "
            f"{row['stopband']:10.4f}  {row['peak_b1']:7.1f} Hz"
        )


# sphinx_gallery_end_ignore
import numpy as np

import pypulseqpp as pp
import pypulseqpp.sequences as design

#: Gyromagnetic ratio of the proton (Hz/T).
GAMMA = 42.576e6

THICKNESS = 5e-3
FLIP_ANGLE_DEG = 8.0

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    adc_dead_time=10e-6,
)

# %%
# Simulating an excitation
# ------------------------
#
# The excitation module designs the pulse, its selection gradient and the
# rephaser that unwinds the second half of the selection. Its ``sim_rf``
# simulates the Bloch response of its pulse across off-resonance,
# which under a selection gradient of amplitude ``selection_amplitude`` is the
# slice profile, because a spin at position ``z`` is off-resonance by
# ``selection_amplitude * z``.


def simulate(time_bw_product, duration_s):
    """Design one excitation and simulate the profile its pulse produces."""
    module = design.SpatialSelectiveExcitation(
        system,
        flip_angle_deg=FLIP_ANGLE_DEG,
        thickness_m=THICKNESS,
        duration_s=duration_s,
        time_bw_product=time_bw_product,
    )
    magnetisation, frequency = module.sim_rf()[1:3]
    profile = np.abs(magnetisation)
    return {
        "tbw": time_bw_product,
        "duration": duration_s,
        "gradient": module.selection_amplitude / GAMMA,
        "position": frequency / module.selection_amplitude,
        "profile": profile / profile.max(),
        "time": np.arange(module.rf.signal.size) * system.rf_raster_time,
        "envelope": np.abs(module.rf.signal),
        "peak_b1": float(np.abs(module.rf.signal).max()),
    }


# %%
# Two numbers describe a profile: how far it takes to fall from the passband to
# the stopband, and how flat it is on either side of that transition.


def describe(simulated):
    """Transition width, passband ripple and stopband level of a profile."""
    position, profile = simulated["position"], simulated["profile"]
    edge = position > 0
    outward, falling = position[edge], profile[edge]

    def crosses(level):
        index = int(np.argmax(falling < level))
        return np.interp(
            level,
            [falling[index], falling[index - 1]],
            [outward[index], outward[index - 1]],
        )

    passband = profile[np.abs(position) < 0.35 * THICKNESS]
    stopband = profile[np.abs(position) > 1.5 * THICKNESS]
    return {
        **simulated,
        "transition": crosses(0.1) - crosses(0.9),
        "passband": float(passband.max() - passband.min()),
        "stopband": float(stopband.max()),
    }


# %%
# The time-bandwidth product at a fixed duration
# ----------------------------------------------
#
# Every design below is 3 ms long and selects the same 5 mm. The pulse has more
# zero crossings as the time-bandwidth product rises, and the selection gradient
# rises with it so that the wider bandwidth still lands on the same slice.

PRODUCTS = (2.0, 4.0, 6.0, 8.0, 12.0)

by_product = [describe(simulate(product, 3e-3)) for product in PRODUCTS]

# sphinx_gallery_start_ignore
table(by_product, "tbw")
profile_figure(
    {f"TBW {row['tbw']:.0f}": row for row in by_product},
    "3 ms pulse, 5 mm slice",
    "time-bandwidth",
)
# sphinx_gallery_end_ignore

# %%
# Above the smallest product the transition width falls close to inversely
# with it, so their product settles towards a figure set by the slice thickness
# rather than by the design. The passband ripple falls over the same range and
# the stopband stays below a percent throughout. The last column gives the
# corresponding increase in transmit amplitude: the peak :math:`B_1` rises in
# proportion to the time-bandwidth product, because the same flip angle is
# delivered by an envelope with more structure in the same time.

# %%
# The duration at a fixed time-bandwidth product
# ----------------------------------------------
#
# Varying duration at fixed time-bandwidth product separates slice-profile
# properties from gradient amplitude and peak :math:`B_1` requirements.

DURATIONS = (1e-3, 2e-3, 3e-3, 5e-3, 8e-3)

by_duration = [describe(simulate(4.0, duration)) for duration in DURATIONS]

# sphinx_gallery_start_ignore
table(
    [{**row, "duration_ms": 1e3 * row["duration"]} for row in by_duration],
    "duration_ms",
)
profile_figure(
    {f"{1e3 * row['duration']:.0f} ms": row for row in by_duration},
    "time-bandwidth product 4, 5 mm slice",
    "duration",
)
# sphinx_gallery_end_ignore

# %%
# The five profiles lie on top of each other. The transition width is the same
# to three decimal places across an eightfold change of duration, and the small
# residual differences in the ripple follow the number of samples the pulse is
# written with: on a fixed RF raster a 1 ms envelope has an eighth of the
# samples of an 8 ms one. The duration sets the selection gradient and the peak
# :math:`B_1`, both of which scale as its reciprocal, and the time the
# repetition spends on the excitation.

# %%
# Designs admitted by the gradient amplitude limit
# ------------------------------------------------
#
# The two sweeps are two lines through one plane, and the amplitude limit cuts
# it along :math:`T = \mathrm{TBW} / (\gamma\, \Delta z\, G_\mathrm{max})`. A
# design above that line is realizable; one below it requires a selection
# gradient above the amplitude limit, and the module raises an error rather
# than widening the slice.

grid = []
for product in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
    for duration in (0.3e-3, 0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3):
        try:
            design.SpatialSelectiveExcitation(
                system,
                flip_angle_deg=FLIP_ANGLE_DEG,
                thickness_m=THICKNESS,
                duration_s=duration,
                time_bw_product=product,
            )
        except ValueError:
            feasible = False
        else:
            feasible = True
        grid.append({"tbw": product, "duration": duration, "feasible": feasible})

# sphinx_gallery_start_ignore
boundary_tbw = np.linspace(1.0, 17.0, 64)
feasibility_figure(grid, boundary_tbw, boundary_tbw / (THICKNESS * system.max_grad))
rejected = sum(not entry["feasible"] for entry in grid)
print(f"{len(grid) - rejected} of {len(grid)} designs realizable, {rejected} rejected")
# sphinx_gallery_end_ignore

# %%
# The designs the module accepted are exactly those above the line. The bound
# is on the amplitude alone: changing the slew limit over the range a gradient
# system covers moves none of the points across it, because a lower slew rate
# lengthens the ramps on either side of the selection plateau, and hence the
# duration of the module, without changing the plateau amplitude.
#
# The same plane read along its other axis gives the complementary statement: a
# sharper profile at a fixed slice thickness is available at any duration the
# gradient amplitude supports, and choosing between a long pulse and a strong
# gradient determines the echo time and the peak :math:`B_1` rather than the
# profile.

# %%
# What a readout module holds
# ---------------------------
#
# The excitation module designs the pulse, its selection gradient and the
# rephaser; the readout module takes those and the prescription. With the echo
# time unset, the module uses the shortest echo time the prescription allows.
# The achieved receiver bandwidth is constrained by the rasters and can differ
# from the requested one; the module reports the achieved value.

FOV = 220e-3
MATRIX = 128

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

figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 3.2), layout="constrained")
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
figure.legend(ncols=2, loc="outside upper left")
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
# reports the achieved rate rather than the requested one.

# %%
# Monopolar against bipolar trains
# --------------------------------
#
# ``n_echoes`` sets the train length, and ``flyback`` selects how it is played: a
# monopolar train rewinds between the echoes so that every one is read in the
# same direction, and a bipolar train alternates the readout sign, as the
# hand-built echo train of
# :doc:`/generated/gallery/03-gre-to-epi/03_epi` does. The bipolar train
# is shorter by the duration of the rewinders, and its even echoes are read
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
    axis.set_title(name)
    axis.set_xlabel("time within the module (ms)")
axes[0].set_ylabel(r"$k_x$ / $k_\mathrm{max}$")
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Every echo of the monopolar train is traversed in the same direction and the
# gaps between them are the rewinders; the bipolar train has no gaps and every
# second echo runs backwards. The choice between them follows from the
# relationship measured in the echo planar lesson: the bipolar train is
# shorter, and any delay between the gradient and the acquisition enters it as
# a difference between the odd and the even echoes rather than as a shift
# common to all of them.
