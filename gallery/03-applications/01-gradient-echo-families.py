"""
======================
Gradient-echo families
======================

Four configurations of the Cartesian gradient-echo sequences the package
ships, and what separates them.

All four excite and read a gradient echo. They differ in what happens to the
transverse magnetization between repetitions, and in what the repetitions are
grouped into: a spoiled sequence destroys it, a balanced one refocuses it, a
multi-echo readout samples several echoes per excitation, and a
magnetization-prepared sequence puts an inversion in front of a train of
repetitions. Each is listed in :doc:`../../../sequences`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def summarize(rows):
    """Print one line per designed sequence: blocks, duration, TE and TR."""
    width = max(len(name) for name, _ in rows)
    print(f"{'sequence':{width}}  {'blocks':>8}  {'duration':>9}  {'TE':>8}  {'TR':>8}")
    for name, seq in rows:
        te = seq.get_definition("TE")
        tr = seq.get_definition("TR")
        te = f"{1e3 * te[0]:.2f} ms" if te else "-"
        tr = f"{1e3 * tr[0]:.2f} ms" if tr else "-"
        print(
            f"{name:{width}}  {seq.num_blocks:8d}  "
            f"{seq.duration()[0]:7.2f} s  {te:>8}  {tr:>8}"
        )


# sphinx_gallery_end_ignore
from pypulseqpp import sequences

# %%
# Spoiled gradient echo
# ---------------------
#
# One line per excitation, the residual transverse magnetization destroyed by
# a readout-axis spoiler and a quadratic RF phase increment. Slices that one TR
# cannot hold are dealt round-robin into packets played one after another,
# which is what makes a multi-slice scan's TR a property of the packet rather
# than of the slice.

spoiled = sequences.gre2D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    n_x=96,
    n_y=96,
    n_slices=3,
    slice_thickness=5e-3,
    flip_angle_deg=12.0,
    te=6e-3,
    tr=None,
)
spoiled.paper_plot()

# %%
# Multi-echo gradient echo
# ------------------------
#
# Several echoes per excitation. A monopolar train, ``flyback=True``, rewinds
# after every echo but the last, so every echo is read in the same direction; a
# bipolar train, ``flyback=False``, reads the even echoes backwards instead,
# which shortens the echo spacing by the rewinder it removes and leaves an
# even/odd trajectory difference for the reconstruction to correct.
# ``n_echoes`` and ``echo_spacing`` then set how the decay is sampled.

multi_echo = sequences.gre_multiecho3D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    fov_z=128e-3,
    n_x=64,
    n_y=64,
    n_z=16,
    n_echoes=4,
    flip_angle_deg=12.0,
    flyback=False,
    tr=None,
    n_dummy=8,
)
multi_echo.paper_plot()

# %%
# Magnetization-prepared gradient echo
# ------------------------------------
#
# One adiabatic inversion per partition, then a train of low-flip spoiled
# repetitions read at an echo spacing of ``esp``. The inversion time ``ti``
# runs from the inversion pulse's centre to the first line's excitation and
# ``tr`` from one inversion to the next, so the contrast follows from the pair
# together with the train length, not from ``ti`` alone.

mprage = sequences.mprage3D_sequence(
    fov_x=256e-3,
    fov_y=256e-3,
    fov_z=176e-3,
    n_x=64,
    n_y=64,
    n_z=32,
    flip_angle_deg=9.0,
    ti=900e-3,
    tr=2.3,
)
mprage.paper_plot(time_range=[0.0, 0.12])

# %%
# Balanced SSFP
# -------------
#
# Every gradient axis returns to k = 0 between pulse centres, so the transverse
# magnetization is carried from one repetition to the next rather than spoiled,
# and TE is half the TR. Each slice's train is entered through a half-flip
# pulse half a repetition ahead of the first excitation and opposite in phase
# to it, after which the excitations alternate between phase pi and 0. Under
# prospective gating each heartbeat opens with a trigger and then plays a
# segment of ``views_per_segment`` lines once per cardiac phase, so a slice is
# covered over as many heartbeats as it has segments.

bssfp = sequences.bssfp2D_sequence(
    fov_x=300e-3,
    fov_y=300e-3,
    n_x=96,
    n_y=96,
    n_slices=1,
    flip_angle_deg=45.0,
    gating="prospective",
    heart_rate_bpm=60.0,
    n_phases=20,
    views_per_segment=8,
)
bssfp.paper_plot()

# %%
# The four side by side
# ---------------------
#
# Scan duration separates the families as clearly as contrast does: the spoiled
# sequences pay one repetition per encoded line, the multi-echo readout gathers
# four contrasts in that same repetition, and the magnetization-prepared and
# gated sequences pay a recovery or a cardiac interval per train.

# sphinx_gallery_start_ignore
summarize(
    [
        ("spoiled 2D, 96 x 96 x 3", spoiled),
        ("multi-echo 3D, 64 x 64 x 16, 4 echoes", multi_echo),
        ("MPRAGE 3D, 64 x 64 x 32", mprage),
        ("bSSFP 2D cine, 96 x 96, 20 phases", bssfp),
    ]
)
# sphinx_gallery_end_ignore
