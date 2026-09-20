r"""
=======================================================
Gradient, slew and receiver limits on a spiral readout
=======================================================

Three limits bound the traversal of a spiral arm. Two are properties of the
gradient system: the maximum amplitude and the maximum slew rate. The third
follows from the receiver: with a dwell time :math:`\Delta t` the trajectory
may not advance further than :math:`1/\mathrm{FOV}` between samples, which
caps the gradient amplitude at

.. math::

    G_\mathrm{bw} = \frac{1}{\gamma\, \Delta t\, \mathrm{FOV}}

independently of what the gradient system could deliver. The solver applies
the lowest of the three, so the readout duration depends on the slew rate over
part of the design space and not over the rest.

This example designs one spiral arm over a grid of slew limits and sampling
rates, reads the peak amplitude and peak slew back off the designed waveform,
and identifies which ceiling bounds each design.
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

REGIME_LABEL = {
    "slew": "slew-limited",
    "bandwidth": "at the receiver cap",
    "amplitude": "at the amplitude cap",
}
REGIME_COLOUR = {
    "slew": "tab:blue",
    "bandwidth": "tab:red",
    "amplitude": "tab:green",
}


def duration_figure(grid):
    """Readout duration against the slew limit, one line per sampling rate."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.0))
    for rate, rows in grid.items():
        slews = [row["slew_limit"] for row in rows]
        axis.plot(
            slews,
            [1e3 * row["readout"] for row in rows],
            "-",
            color="0.7",
            zorder=1,
        )
        for row in rows:
            axis.plot(
                row["slew_limit"],
                1e3 * row["readout"],
                "o",
                ms=7,
                color=REGIME_COLOUR[row["regime"]],
                zorder=2,
            )
        axis.annotate(
            f"{rate / 1e3:.0f} kHz",
            (slews[-1], 1e3 * rows[-1]["readout"]),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=9,
        )
    for regime, colour in REGIME_COLOUR.items():
        axis.plot([], [], "o", color=colour, label=REGIME_LABEL[regime])
    axis.set_xlabel("slew limit (T/m/s)")
    axis.set_ylabel("readout duration (ms)")
    axis.set_ylim(bottom=0)
    axis.margins(x=0.16)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    return figure


def waveform_figure(panels):
    """Amplitude and slew along the arm, against the ceilings that bound them."""
    figure, axes = plt.subplots(
        2, len(panels), figsize=(PAGE_WIDTH, 5.0), sharex="col", squeeze=False
    )
    for column, (name, arm) in enumerate(panels.items()):
        top, bottom = axes[0][column], axes[1][column]
        top.plot(1e3 * arm["time"], 1e3 * arm["magnitude"], color="tab:blue")
        top.axhline(1e3 * arm["ceiling_bw"], color="tab:red", ls="--", lw=1.0)
        top.axhline(1e3 * arm["ceiling_grad"], color="tab:green", ls="--", lw=1.0)
        top.set_title(name)
        top.set_ylim(0, 1e3 * 1.15 * arm["ceiling_grad"])
        bottom.plot(1e3 * arm["time"][:-1], arm["slew"], color="tab:blue")
        bottom.axhline(arm["ceiling_slew"], color="0.4", ls="--", lw=1.0)
        bottom.set_ylim(0, 1.15 * arm["ceiling_slew"])
        bottom.set_xlabel("time from the start of the arm (ms)")
    axes[0][0].set_ylabel("|G| (mT/m)")
    axes[1][0].set_ylabel("|dG/dt| (T/m/s)")
    figure.tight_layout()
    return figure


def interleaf_figure(rows):
    """Arm duration and repetition content against the number of interleaves."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6))
    arms = [row["interleaves"] for row in rows]
    axis.plot(arms, [1e3 * row["readout"] for row in rows], "o-", color="tab:blue")
    axis.set_xlabel("interleaves")
    axis.set_ylabel("readout duration (ms)", color="tab:blue")
    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_xticks(arms)
    axis.set_xticklabels([str(n) for n in arms])
    second = axis.twinx()
    second.plot(arms, [row["scan"] for row in rows], "s-", color="tab:red")
    second.set_ylabel("time for a full set of arms (s)", color="tab:red")
    figure.tight_layout()
    return figure


def grid_table(grid):
    """Print the readout duration and binding ceiling of every design."""
    rates = list(grid)
    print(f"{'slew':>6}" + "".join(f"{rate / 1e3:>12.0f} kHz" for rate in rates))
    for index in range(len(grid[rates[0]])):
        row = f"{grid[rates[0]][index]['slew_limit']:6.0f}"
        for rate in rates:
            entry = grid[rate][index]
            row += f"{1e3 * entry['readout']:9.2f} ms {entry['regime'][:4]:>4}"
        print(row)


# sphinx_gallery_end_ignore
import numpy as np

import pypulseqpp as pp
import pypulseqpp.sequences as design

#: Gyromagnetic ratio of the proton (Hz/T).
GAMMA = 42.576e6

FOV = 220e-3
MATRIX = 128
MAX_GRAD_MT_M = 40.0

# %%
# Designing one arm
# -----------------
#
# The readout module designs the arm from the prescription and the system
# limits it is given, so a design is one call and the waveform it produced is
# an attribute of the result. ``design_interleaves`` sets the pitch of the
# spiral, against which the readout duration is measured. It is not the number
# of arms a scan plays.


def spiral_arm(max_slew, sampling_rate_hz, interleaves=16):
    """One spiral arm designed at the given slew limit and sampling rate."""
    system = pp.Opts(
        max_grad=MAX_GRAD_MT_M,
        grad_unit="mT/m",
        max_slew=max_slew,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    # The module solves against the package's derated limits, so those are the
    # ceilings a measurement of the waveform has to be read against.
    return pp.apply_system_derates(system), design.SpiralReadout2D(
        system,
        excitation.rf,
        excitation.gz,
        fov=FOV,
        matrix=MATRIX,
        design_interleaves=interleaves,
        readout_bandwidth_hz=sampling_rate_hz,
    )


# %%
# The amplitude and slew the design reached are measured on the waveform it
# wrote, on its own raster and along the vector rather than per axis, because
# the two in-plane axes play at once.


def measure(system, arm):
    """Duration, waveform and the ceilings the design was bounded by."""
    x = np.asarray(arm.gx.waveform)
    y = np.asarray(arm.gy.waveform)
    times = np.asarray(arm.gx.tt)
    raster = float(times[1] - times[0])
    return {
        "readout": arm.n_samples / arm.bandwidth_hz,
        "time": times - times[0],
        "magnitude": np.hypot(x, y) / GAMMA,
        "slew": np.hypot(np.diff(x), np.diff(y)) / GAMMA / raster,
        "ceiling_grad": system.max_grad / GAMMA,
        "ceiling_slew": system.max_slew / GAMMA,
        "ceiling_bw": arm.bandwidth_hz / (GAMMA * FOV),
    }


def limiting_ceiling(measured, tolerance=0.02):
    """Which ceiling caps the arm's amplitude, or ``'slew'`` if none of them does.

    An arm is at its slew limit wherever it is turning, so reaching that limit
    says nothing on its own. What decides whether more slew rate would shorten
    the traversal is whether the amplitude has reached a ceiling.
    """
    cap = min(measured["ceiling_grad"], measured["ceiling_bw"])
    if measured["magnitude"].max() < (1.0 - tolerance) * cap:
        return "slew"
    return (
        "bandwidth"
        if measured["ceiling_bw"] < measured["ceiling_grad"]
        else "amplitude"
    )


# %%
# The design space
# ----------------
#
# The slew limit is swept over the range a body gradient system covers, and the
# sampling rate over a range whose receiver cap runs from well below the
# gradient amplitude limit to above it.

SLEWS = (40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 210.0)
RATES = (100e3, 250e3, 600e3)

grid = {}
for rate in RATES:
    grid[rate] = []
    for slew in SLEWS:
        system, arm = spiral_arm(slew, rate)
        measured = measure(system, arm)
        grid[rate].append(
            {"slew_limit": slew, "regime": limiting_ceiling(measured), **measured}
        )

# sphinx_gallery_start_ignore
grid_table(grid)
duration_figure(grid)
# sphinx_gallery_end_ignore

# %%
# The three rates behave differently. At the lowest, the receiver's cap is so
# far below the gradient amplitude limit that the arm reaches it within the
# first turn, and the duration is the same across the whole slew range. At the
# middle rate the amplitude climbs at the slew limit until it meets the receiver's
# cap, after which raising the limit changes the duration by about a percent.
# At the highest rate the receiver's cap is above the gradient amplitude limit,
# so the amplitude the arm settles at is the hardware's, and it is only reached
# at the top of the slew range; below that the arm is still climbing when it
# ends.
#
# The flat part of the middle curve is not exactly flat, and the reason is that
# an arm at constant amplitude is still turning. Holding :math:`|G|` while the
# direction rotates costs slew rate of its own, and the tighter the turn the
# more of it, so the slew limit continues to govern the first turns of an arm
# whose amplitude has already stopped growing.

# %%
# The waveform in each regime
# ---------------------------
#
# One design from each regime, with the ceilings drawn on the axes they bound.

# sphinx_gallery_start_ignore
waveform_figure(
    {
        "slew-limited": measure(*spiral_arm(60.0, 250e3)),
        "receiver cap": measure(*spiral_arm(210.0, 250e3)),
        "amplitude cap": measure(*spiral_arm(210.0, 600e3)),
    }
)
# sphinx_gallery_end_ignore

# %%
# In the slew-limited design the amplitude is still climbing when the arm ends.
# In the other two it reaches a ceiling part way out and stays there, and the
# slew falls away from its limit once it does: the remaining traversal is at
# constant speed, and the only turning left is the angular one. The slew rate
# is at its limit early in every one of them, which is why reaching the slew
# limit is not by itself what tells the three apart.
#
# The ceilings are drawn at the system's derated limits rather than at the
# numbers passed in. A design whose two in-plane axes play together is solved
# against a per-axis limit reduced by :math:`\sqrt{2}`, so that the vector
# magnitude drawn here respects the scalar limit.

# %%
# Interleaves against arm duration
# --------------------------------
#
# Within one regime the pitch is the remaining lever: more interleaves cover
# k-space with shorter arms, and the set of them takes correspondingly longer
# to play.

interleaves = []
for count in (4, 8, 16, 32, 48):
    system, arm = spiral_arm(150.0, 250e3, interleaves=count)
    interleaves.append(
        {
            "interleaves": count,
            "readout": arm.n_samples / arm.bandwidth_hz,
            "scan": count * arm.duration,
        }
    )

# sphinx_gallery_start_ignore
print(f"\n{'arms':>6}  {'readout':>11}  {'per arm':>11}  {'full set':>11}")
for row in interleaves:
    count = row["interleaves"]
    print(
        f"{count:6d}  {1e3 * row['readout']:8.2f} ms  "
        f"{1e3 * row['scan'] / count:8.2f} ms  {row['scan']:8.3f} s"
    )
interleaf_figure(interleaves)
# sphinx_gallery_end_ignore

# %%
# The arm duration falls almost as the reciprocal of the interleaf count while
# the time for a full set rises less than proportionally, because each
# repetition carries an excitation and a rewind whose duration does not depend
# on the pitch. Off-resonance and :math:`T_2^*` act over the readout duration,
# so the interleaf count is the remaining way to shorten it once the slew rate
# no longer does.
