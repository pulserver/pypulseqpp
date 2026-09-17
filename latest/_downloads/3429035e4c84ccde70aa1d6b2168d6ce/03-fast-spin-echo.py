"""
================================
Three-dimensional fast spin echo
================================

Four configurations of the three-dimensional fast spin echo: the ordering the
echo train reads its views in, the refocusing-angle schedule it plays, and
trains parameterized separately for the centre and the periphery of k-space.

A single excitation is followed by a CPMG train of ``etl`` refocused echoes,
each reading one ``(line, partition)`` view. Which view is read at which echo
is the sequence's main degree of freedom: the signal decays along the train, so
the ordering fixes the effective contrast, the point-spread function and how
much of the train's decay ends up as blurring.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.4),
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def refocusing_angles(seq, count):
    """Flip angles of the first ``count`` refocusing pulses, in degrees."""
    blocks = np.array([seq.block_events[i] for i in sorted(seq.block_events)])
    angles = []
    for index in np.flatnonzero(blocks[:, 1]) + 1:
        rf = seq.get_block(int(index)).rf
        if rf.use != "refocusing":
            continue
        integral = float(np.abs(rf.signal).sum()) * seq.system.rf_raster_time
        angles.append(np.rad2deg(2 * np.pi * integral))
        if len(angles) == count:
            break
    return np.asarray(angles)


def schedule_figure(schedules):
    """Draw each named refocusing-angle schedule against the echo index."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for name, angles in schedules.items():
        axis.plot(np.arange(1, angles.size + 1), angles, marker="o", ms=3, label=name)
    axis.set_xlabel("echo index")
    axis.set_ylabel("refocusing flip angle (deg)")
    axis.set_ylim(0, 190)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def view_figure(panels):
    """Draw the acquired (line, partition) views, coloured by echo index."""
    figure, axes = plt.subplots(
        1, len(panels), figsize=(PAGE_WIDTH, 3.8), squeeze=False
    )
    for axis, (name, seq) in zip(axes[0], panels.items(), strict=True):
        labels = seq.evaluate_labels(evolution="adc")
        drawn = axis.scatter(
            labels["LIN"], labels["PAR"], c=labels["ECO"], s=2.0, cmap="viridis"
        )
        axis.set_title(name)
        axis.set_xlabel("line index")
        axis.set_aspect("equal")
    axes[0][0].set_ylabel("partition index")
    figure.colorbar(drawn, ax=axes[0], label="echo index", shrink=0.85)
    return figure


def summarize(rows):
    """Print one line per designed sequence: blocks, duration, TE and TR."""
    width = max(len(name) for name, _ in rows)
    print(f"{'configuration':{width}}  {'blocks':>7}  {'duration':>9}  {'TE':>9}")
    for name, seq in rows:
        te = seq.get_definition("TE")
        te = f"{1e3 * te[0]:.1f} ms" if te else "-"
        print(
            f"{name:{width}}  {seq.num_blocks:7d}  {seq.duration()[0]:7.1f} s  {te:>9}"
        )


# sphinx_gallery_end_ignore
from pypulseqpp import sequences

#: The field of view and matrix size every configuration below is designed at.
PRESCRIPTION = {
    "fov_x": 200e-3,
    "fov_y": 200e-3,
    "fov_z": 200e-3,
    "n_x": 64,
    "n_y": 48,
    "n_z": 48,
}

# %%
# Radial ordering at a constant refocusing angle
# ----------------------------------------------
#
# The reference configuration: every refocusing pulse is a full 180 degrees,
# and the views are ordered by their distance from the centre of k-space, so
# that the echo at the prescribed effective TE reads the centre. A constant
# 180-degree train gives the largest echo amplitudes and the highest RF power
# deposition, and its usable length is bounded by T2 decay.

constant = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=60e-3,
    tr=1.0,
    etl=16,
    refocusing_angle_deg=180.0,
    ordering="radial",
    flip_modulation="constant",
)
constant.paper_plot(tr=1)

# %%
# Optimized refocusing angles
# ---------------------------
#
# ``flip_modulation="optimized"`` designs the refocusing schedule with
# torchsim, which the ``design`` extra installs: the angles are chosen so that
# the echo amplitudes follow a prescribed envelope rather than the free decay
# of a constant train. A schedule that starts near 180 degrees and falls
# maintains signal over a longer train at a fraction of the RF power, which is
# what makes a long train usable at all.

optimized = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=60e-3,
    tr=1.0,
    etl=16,
    refocusing_angle_deg=180.0,
    ordering="radial",
    flip_modulation="optimized",
)

# sphinx_gallery_start_ignore
schedule_figure(
    {
        "constant": refocusing_angles(constant, 16),
        "optimized": refocusing_angles(optimized, 16),
    }
)
# sphinx_gallery_end_ignore

# %%
# Shuffled ordering
# -----------------
#
# ``ordering="shuffling"`` replaces the distance ordering with a shuffled
# Poisson-disc set: each echo of the train reads views drawn from the whole of
# k-space rather than from one distance band. The decay along the train is then
# a contrast dimension the reconstruction can resolve, rather than a fixed
# point-spread function applied to one image. The centre of k-space is read at
# several echoes under this ordering, so no single echo time characterizes the
# contrast and ``te`` names none; the ``TE`` definition records the first
# echo.

shuffled = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=None,
    tr=1.0,
    etl=16,
    ordering="shuffling",
    flip_modulation="optimized",
)

# sphinx_gallery_start_ignore
view_figure({"radial ordering": optimized, "shuffled ordering": shuffled})
# sphinx_gallery_end_ignore

# %%
# Individually parameterized trains
# ---------------------------------
#
# ``tr_periphery`` and ``etl_periphery`` give the trains reading the periphery
# of k-space a different repetition time and length from those reading the
# centre. The centre, which sets the contrast, keeps a short train and a short
# TR; the periphery, which sets the resolution and tolerates more decay, is
# read with a longer train, so fewer excitations cover the same number of
# views. The two ends are interpolated over the distance ordering, which is why
# this configuration requires it.

individual = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=60e-3,
    tr=1.0,
    etl=16,
    tr_periphery=1.8,
    etl_periphery=48,
    ordering="radial",
    flip_modulation="optimized",
)

# %%
# The four side by side
# ---------------------
#
# The individually parameterized configuration reads the same 48 x 48 views as
# the others; what changes is how many excitations it takes to read them and
# how long each one waits.

# sphinx_gallery_start_ignore
summarize(
    [
        ("radial ordering, constant 180 deg", constant),
        ("radial ordering, optimized angles", optimized),
        ("shuffled ordering, optimized angles", shuffled),
        ("individually parameterized trains", individual),
    ]
)
# sphinx_gallery_end_ignore
