r"""
========
Spoiling
========

The gradient echo of the previous lesson,
:doc:`/generated/gallery/01-pulseq-basics/03_gradient_echo`, leaves transverse
magnetisation at the end of each repetition, and the following pulses refocus
part of it. This lesson first adds a spoiler gradient after the acquisition.
The spoiler dephases the remaining transverse magnetisation through several
cycles across a voxel, so that it integrates to nearly zero there; because
every repetition applies the same dephasing, a coherent pathway remains and
contributes to the steady state. The lesson then suppresses that pathway by
advancing the phase of the RF pulse and of the receiver by a quadratically
increasing amount from one repetition to the next, and measures which phase
increments bring the steady-state signal close to that of ideal spoiling.

The steady-state signal is computed by summing isochromats across a voxel over
several hundred repetitions of the sequence built here, and compared with the
signal of an ideally spoiled repetition. The next lesson,
:doc:`/generated/gallery/03-gre-to-epi/03_epi`, acquires many k-space lines
after one excitation.

Learning objectives
-------------------

After this lesson, you should be able to:

- prescribe a spoiler gradient by its dephasing across a voxel and place it
  in the repetition;
- compute a steady-state signal by an isochromat summation that uses the
  echo time, repetition time and spoiler area of the built sequence;
- explain why no spoiler area reproduces the ideally spoiled signal;
- write the quadratic phase cycle of RF spoiling, set the phase offset of the
  RF and ADC events per repetition, and read it back from the blocks;
- identify phase increments that leave the residual pathway coherent;
- compare the gradient-spoiled and RF-spoiled steady states with the ideally
  spoiled signal over flip angle.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# The spoiler
# -----------
#
# The prescription is the same gradient echo, with one gradient added after the
# acquisition. A spoiler is stated as the phase it winds across a voxel, and
# the voxel here is the in-plane sample spacing of the prescription.

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
SPOILER_CYCLES = 4.0
PHASE_INCREMENT_DEG = 117.0

VOXEL = FOV / MATRIX

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

# %%
# One repetition
# --------------
#
# The spoiler is played on the slice axis after the acquisition, where it
# dephases the transverse magnetisation the repetition leaves behind without
# adding to the k-space the readout traverses. It lengthens the repetition, so
# the delay that brings the repetition up to the repetition time absorbs less.
#
# ``phase_offset`` is the phase of an RF event or an ADC event, in radians. The
# function below sets both to the phase of the cycle introduced later in this
# lesson; with an increment of zero every phase is zero, and the sequence is
# gradient spoiled only.


def transmit_phase(repetition, increment_deg):
    """The phase of one repetition's pulse and receiver, in radians."""
    return np.deg2rad(increment_deg) * repetition * (repetition + 1) / 2


def spoiled(cycles, increment_deg=0.0, flip_angle_deg=FLIP_ANGLE_DEG):
    """The sequence, with a spoiler and an RF phase cycle of the given increment."""
    pulse, selection, rephaser = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=2e-3,
        slice_thickness=THICKNESS,
        apodization=0.5,
        time_bw_product=4.0,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
        return_gz=True,
    )
    spoiler = pp.make_crusher(cycles, VOXEL, channel="z", system=system)[0]
    played = (
        pp.calc_duration(pulse, selection)
        + pp.calc_duration(gx_pre, gy_pre, rephaser)
        + pp.calc_duration(gx, adc)
        + pp.calc_duration(spoiler)
    )
    recovery = pp.make_delay(
        pp.round_to_raster(REPETITION_TIME - played, system.block_duration_raster)
    )
    seq = pp.Sequence(system=system)
    steps = np.linspace(-1.0, 1.0, MATRIX, endpoint=False)
    for repetition, step in enumerate(steps):
        phase = transmit_phase(repetition, increment_deg) % (2 * np.pi)
        pulse.phase_offset = phase
        adc.phase_offset = phase
        seq.add_block(pulse, selection)
        seq.add_block(gx_pre, pp.scale_grad(gy_pre, step), rephaser)
        seq.add_block(gx, adc)
        seq.add_block(spoiler)
        seq.add_block(recovery)
    return seq


seq = spoiled(SPOILER_CYCLES)

ok, errors = seq.check_timing()
spoiler_duration = pp.calc_duration(
    pp.make_crusher(SPOILER_CYCLES, VOXEL, channel="z", system=system)[0]
)
print(
    f"timing {ok}, {seq.num_blocks} blocks, "
    f"spoiler {1e3 * spoiler_duration:.2f} ms for {SPOILER_CYCLES:.0f} cycles "
    f"across a {1e3 * VOXEL:.2f} mm voxel"
)

seq.paper_plot(tr=1)

# %%
# Isochromats across a voxel
# --------------------------
#
# The steady state is reached by playing the repetition several hundred times
# on a set of isochromats spread across one voxel. Each one has a complex
# transverse component and a longitudinal one, and each repetition applies the
# pulse at its transmit phase, the interval to the echo, the interval from the
# echo to the end of the repetition, and the phase the spoiler winds at that
# isochromat's position. The receiver is advanced with the transmitter, so the
# transmit phase of the current repetition is removed from the recorded signal.
#
# The echo time and the repetition time come from the sequence that was built,
# and the spoiler phase from the dephasing it was prescribed with, so the
# summation follows the design rather than a restatement of it.

ISOCHROMATS = 201

_, _, t_excitation, _, t_adc = seq.calculate_kspace(block_range=[1, 5])
ECHO_TIME = float(t_adc[MATRIX // 2] - t_excitation[0])

print(
    f"echo time {1e3 * ECHO_TIME:.2f} ms, repetition time {1e3 * REPETITION_TIME:.2f} ms"
)


def steady_state(
    cycles, increment_deg, flip_angle_deg, t1=1000e-3, t2=80e-3, repetitions=600
):
    """The signal of each repetition, summed across a voxel."""
    flip = np.deg2rad(flip_angle_deg)
    position = (np.arange(ISOCHROMATS) + 0.5) / ISOCHROMATS
    spoiler_phase = np.exp(2j * np.pi * cycles * position)
    rest = REPETITION_TIME - ECHO_TIME

    transverse = np.zeros(ISOCHROMATS, dtype=complex)
    longitudinal = np.ones(ISOCHROMATS)
    signal = np.zeros(repetitions, dtype=complex)

    for repetition in range(repetitions):
        # The pulse, about an axis in the transverse plane at the transmit phase.
        turn = np.exp(1j * transmit_phase(repetition, increment_deg))
        rotated = (
            np.cos(flip / 2) ** 2 * transverse
            + np.sin(flip / 2) ** 2 * turn**2 * np.conj(transverse)
            - 1j * np.sin(flip) * turn * longitudinal
        )
        longitudinal = np.cos(flip) * longitudinal + np.sin(flip) * np.imag(
            np.conj(turn) * transverse
        )
        transverse = rotated

        # Excitation to echo, where the signal is read at the receiver phase.
        transverse *= np.exp(-ECHO_TIME / t2)
        longitudinal = 1.0 + (longitudinal - 1.0) * np.exp(-ECHO_TIME / t1)
        signal[repetition] = (transverse * np.conj(turn)).mean()

        # Echo to the end of the repetition, and the spoiler.
        transverse *= np.exp(-rest / t2) * spoiler_phase
        longitudinal = 1.0 + (longitudinal - 1.0) * np.exp(-rest / t1)

    return signal


def ideally_spoiled(flip_angle_deg, t1=1000e-3):
    """The signal of a repetition that begins with no transverse component."""
    flip = np.deg2rad(flip_angle_deg)
    recovery = np.exp(-REPETITION_TIME / t1)
    return np.sin(flip) * (1 - recovery) / (1 - recovery * np.cos(flip))


# %%
# Steady state against spoiler area
# ---------------------------------
#
# The sweep is run without a phase cycle, at the prescribed flip angle and at
# one large enough for the residual pathway to be substantial.

CYCLES = np.linspace(0.0, 8.0, 41)
SWEPT_FLIPS = (FLIP_ANGLE_DEG, 30.0)

against_area = {
    flip: np.array([abs(steady_state(cycles, 0.0, flip)[-1]) for cycles in CYCLES])
    for flip in SWEPT_FLIPS
}

# sphinx_gallery_start_ignore
figure, (against_cycles, approach) = plt.subplots(
    1, 2, figsize=(PAGE_WIDTH, 3.2), layout="constrained"
)
for flip, curve in against_area.items():
    line = against_cycles.plot(CYCLES, curve, lw=1.4, label=f"{flip:.0f} deg")[0]
    against_cycles.axhline(
        ideally_spoiled(flip), color=line.get_color(), ls="--", lw=1.2
    )
against_cycles.set_xlabel(f"spoiler, cycles across {1e3 * VOXEL:.2f} mm")
against_cycles.set_ylabel("steady-state signal")
against_cycles.set_ylim(bottom=0.0)
against_cycles.set_title("dashed: ideally spoiled")
figure.legend(
    *against_cycles.get_legend_handles_labels(), ncols=2, loc="outside upper left"
)
evolution = steady_state(SPOILER_CYCLES, 0.0, 30.0)
approach.plot(np.abs(evolution), lw=1.2, label="gradient spoiling")
approach.axhline(
    ideally_spoiled(30.0), color="0.5", ls="--", lw=1.2, label="ideally spoiled"
)
approach.set_xlabel("repetition")
approach.set_ylabel("signal")
approach.set_xscale("log")
approach.set_title("30 deg, 4 cycles")
figure.legend(*approach.get_legend_handles_labels(), loc="outside upper right")

beyond = against_area[30.0][CYCLES >= 3.0]
print(
    f"\nat 30 degrees: ideally spoiled {ideally_spoiled(30.0):.4f}, "
    f"gradient spoiled {beyond.mean():.4f}, rippling by "
    f"{100 * (beyond.max() - beyond.min()) / beyond.mean():.1f} percent "
    f"above three cycles"
)
# sphinx_gallery_end_ignore

# %%
# Below one cycle across a voxel the isochromats are not spread over the whole
# circle, and the steady state depends on how far the spoiler winds them. Above
# it the dependence is a ripple that decays as the reciprocal of the cycle
# count, because a uniform voxel wound through a non-integral number of cycles
# does not quite average to zero, and past about three cycles the curve is flat
# to the last digit.
#
# The plateau consists of a pathway that the pulse refocuses from one
# repetition to the next: the spoiler winds every repetition through the same
# phase, so it leaves that pathway unchanged. At 30 degrees that residual is
# more than one and a half times the ideally spoiled signal it is being
# compared with, and no spoiler area removes it.
#
# The right-hand panel shows the approach to the steady state at the prescribed
# spoiler, which takes a few hundred repetitions — long enough that the first
# lines of a scan are acquired before it.

# %%
# The RF phase cycle
# ------------------
#
# The transmit phase of repetition :math:`n` is advanced by :math:`n` times a
# fixed increment, so that the phase itself grows quadratically:
#
# .. math::
#
#     \varphi_n = \varphi_{n-1} + n \, \Delta\varphi
#             = \tfrac{1}{2} n (n + 1) \, \Delta\varphi .
#
# The pathway a pulse refocuses from an earlier repetition arrives with a phase
# that depends on how many repetitions ago it was produced, so a phase that is
# not linear in :math:`n` puts successive repetitions' residuals at different
# phases and their sum towards zero. The signal of the current repetition is
# unaffected, because the receiver is advanced by the same amount.
#
# The phase the events were written with is read back from the blocks, which
# is where an interpreter and a reconstruction find it.

rf_spoiled = spoiled(SPOILER_CYCLES, PHASE_INCREMENT_DEG)

ok, errors = rf_spoiled.check_timing()
print(f"timing {ok}, {rf_spoiled.num_blocks} blocks, {rf_spoiled.duration()[0]:.3f} s")

written = np.array(
    [np.rad2deg(rf_spoiled.get_block(1 + 5 * n).rf.phase_offset) for n in range(6)]
)
print("transmit phase of the first repetitions (degrees): ", np.round(written, 1))

# %%
# Steady state against phase increment
# ------------------------------------
#
# The summation above, with the transmit phase of each repetition applied to the
# pulse and removed from the signal, swept over the increment.

INCREMENTS = np.arange(0.0, 181.0, 1.0)
INCREMENT_FLIPS = (FLIP_ANGLE_DEG, 30.0, 60.0)

against_increment = {
    flip: np.array(
        [
            abs(steady_state(SPOILER_CYCLES, step, flip, repetitions=500)[-1])
            for step in INCREMENTS
        ]
    )
    for flip in INCREMENT_FLIPS
}

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6), layout="constrained")
for flip, curve in against_increment.items():
    line = axis.plot(
        INCREMENTS, curve / ideally_spoiled(flip), lw=1.2, label=f"{flip:.0f} deg"
    )[0]
axis.axhline(1.0, color="0.5", ls="--", lw=1.0)
axis.axvline(PHASE_INCREMENT_DEG, color="0.5", lw=0.8)
axis.annotate(
    f"{PHASE_INCREMENT_DEG:.0f} deg",
    (PHASE_INCREMENT_DEG, 1.0),
    xycoords=("data", "axes fraction"),
    textcoords="offset points",
    xytext=(4, -12),
    fontsize="small",
)
axis.set_xlabel("phase increment (degrees)")
axis.set_ylabel("signal / ideally spoiled")
axis.set_xlim(0.0, 180.0)
figure.legend(
    *axis.get_legend_handles_labels(),
    title="flip angle, dashed: ideally spoiled",
    ncols=3,
    loc="outside upper left",
)

print(f"\n{'flip':>5}  {'at 117 deg':>11}  {'lowest':>18}  {'highest':>18}")
for flip, curve in against_increment.items():
    ratio = curve / ideally_spoiled(flip)
    low, high = int(np.argmin(ratio)), int(np.argmax(ratio))
    print(
        f"{flip:5.0f}  {ratio[int(PHASE_INCREMENT_DEG)]:11.3f}  "
        f"{ratio[low]:9.3f} at {INCREMENTS[low]:3.0f}  "
        f"{ratio[high]:9.3f} at {INCREMENTS[high]:3.0f}"
    )
# sphinx_gallery_end_ignore

# %%
# The curve is spiky rather than smooth. An increment that returns to a small
# set of phases leaves the residual as coherent as no phase cycle at all: zero
# and 180 degrees are the clearest, and each of the flip angles swept here has
# others of its own. An increment of 117 degrees is within a few percent of the
# ideally spoiled value at every one of them, which is the reason for its
# common use, and its neighbourhood is narrow — at 60 degrees of flip, an
# increment of 82 degrees gives a quarter less signal than the ideal and 117
# gives it to two percent.

# %%
# Steady state against flip angle
# -------------------------------
#
# The residual pathway contains magnetisation that the pulse returns from the
# longitudinal axis, so it grows with the flip angle, and it adds to or
# subtracts from the ideally spoiled signal depending on where the flip angle
# sits. Both spoiling schemes are compared with the ideally spoiled signal.

FLIP_ANGLES = np.arange(2.0, 61.0, 2.0)

gradient_only = np.array(
    [abs(steady_state(SPOILER_CYCLES, 0.0, flip)[-1]) for flip in FLIP_ANGLES]
)
phase_cycled = np.array(
    [
        abs(
            steady_state(SPOILER_CYCLES, PHASE_INCREMENT_DEG, flip, repetitions=500)[-1]
        )
        for flip in FLIP_ANGLES
    ]
)
ideal = ideally_spoiled(FLIP_ANGLES)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.7, 3.4), layout="constrained")
axis.plot(FLIP_ANGLES, gradient_only, lw=1.4, label="gradient spoiling")
axis.plot(
    FLIP_ANGLES,
    phase_cycled,
    lw=1.4,
    label=f"RF spoiling, {PHASE_INCREMENT_DEG:.0f} deg",
)
axis.plot(FLIP_ANGLES, ideal, lw=1.4, ls="--", color="0.5", label="ideally spoiled")
axis.set_xlabel("flip angle (degrees)")
axis.set_ylabel("steady-state signal")
figure.legend(ncols=2, loc="outside upper left")

print(
    f"\n{'flip':>5}  {'gradient spoiled':>17}  {'RF spoiled':>11}  "
    f"{'ideally spoiled':>16}"
)
for flip, gradient, cycled, reference in zip(
    FLIP_ANGLES, gradient_only, phase_cycled, ideal, strict=True
):
    if flip % 10 == 0:
        print(f"{flip:5.0f}  {gradient:17.4f}  {cycled:11.4f}  {reference:16.4f}")
print(
    f"peak: ideally spoiled at {FLIP_ANGLES[int(np.argmax(ideal))]:.0f} "
    f"degrees, gradient spoiled at {FLIP_ANGLES[int(np.argmax(gradient_only))]:.0f} "
    f"degrees, RF spoiled at {FLIP_ANGLES[int(np.argmax(phase_cycled))]:.0f} degrees"
)
print(
    f"largest departure of RF spoiling from the ideally spoiled signal: "
    f"{100 * np.abs(phase_cycled / ideal - 1.0).max():.1f} percent, at "
    f"{FLIP_ANGLES[int(np.argmax(np.abs(phase_cycled / ideal - 1.0)))]:.0f} degrees"
)
# sphinx_gallery_end_ignore

# %%
# The ideally spoiled curve peaks at the Ernst angle for this repetition time
# and T1, and its shape is what a signal model inverted for T1 assumes. The
# gradient-spoiled curve peaks well beyond it, falls below it at small flip
# angles and rises to twice it at large ones, and the whole departure depends
# on T2, which that model does not include.
#
# The RF-spoiled curve lies with the ideally spoiled one over the whole range,
# and its peak is back at the Ernst angle. A departure of a few percent
# remains, which depends on T2 and on the flip angle: the phase cycle cancels
# the residual pathway approximately rather than removing it, and a T1
# estimated from this signal under an ideally spoiled model is biased by the
# remainder.
