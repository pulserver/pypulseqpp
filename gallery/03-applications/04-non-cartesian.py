"""
==========================
Non-Cartesian trajectories
==========================

Radial, spiral, PROPELLER and zero-echo-time sequences, and the interleaves
each of them rotates.

A non-Cartesian readout module designs one base interleaf — its acquisition
window and the gradients that prewind to and rewind from k = 0 — and the scan
loop rotates it per shot with a ``ROTATIONS`` extension rather than by writing
a rotated copy of the waveform. One interleaf in the gradient library therefore
serves the whole scan, however many angles it is played at.
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


def trajectory_figure(panels, plane="xy"):
    """Draw each sequence's ADC sampling locations, one panel each."""
    figure, axes = plt.subplots(
        1, len(panels), figsize=(PAGE_WIDTH, 3.4), squeeze=False
    )
    first, second = "xyz".index(plane[0]), "xyz".index(plane[1])
    for axis, (name, seq) in zip(axes[0], panels.items(), strict=True):
        samples = seq.calculate_kspace()[0]
        axis.plot(
            samples[first], samples[second], ".", ms=0.4, color="tab:blue", alpha=0.4
        )
        axis.set_title(name)
        axis.set_xlabel(f"$k_{plane[0]}$ (1/m)")
        axis.set_aspect("equal")
    axes[0][0].set_ylabel(f"$k_{plane[1]}$ (1/m)")
    figure.tight_layout()
    return figure


def summarize(rows):
    """Print blocks, distinct shapes, rotation entries and duration."""
    width = max(len(name) for name, _ in rows)
    print(
        f"{'sequence':{width}}  {'blocks':>7}  {'shapes':>7}  "
        f"{'rotations':>10}  {'duration':>9}"
    )
    for name, seq in rows:
        libraries = seq.test_report_dict()["libraries"]
        print(
            f"{name:{width}}  {seq.num_blocks:7d}  {libraries['Shape']:7d}  "
            f"{libraries['Rotation']:10d}  {seq.duration()[0]:7.2f} s"
        )


# sphinx_gallery_end_ignore
from pypulseqpp import sequences

# %%
# Radial
# ------
#
# One full spoke per repetition, the same readout gradient rotated to each
# angle. The spokes are spaced uniformly over half a turn at the count that
# meets the Nyquist criterion at the edge of k-space, and ``ry`` reads one
# spoke in every ``ry`` of them. Every spoke crosses the centre of k-space, so
# the low spatial frequencies are sampled once per repetition rather than once
# per scan.

radial = sequences.gre_radial2D_sequence(
    fov=220e-3, n=128, n_slices=1, flip_angle_deg=12.0, tr=None, n_dummy=0
)

# %%
# Spiral
# ------
#
# One interleaf per repetition, designed from the prescription rather than from
# a fixed shape: ``n_shots`` interleaves cover k-space together, ``density``
# and ``periphery_undersampling`` set how the sampling density falls with
# radius, and ``transition_speed`` bounds how fast the trajectory may change
# direction where the spiral leaves the centre.

spiral = sequences.gre_spiral2D_sequence(
    fov=220e-3,
    n=128,
    n_shots=16,
    n_slices=1,
    density="constant",
    periphery_undersampling=2.0,
    tr=None,
    n_dummy=0,
)

# sphinx_gallery_start_ignore
trajectory_figure({"radial, golden angle": radial, "spiral, 16 interleaves": spiral})
# sphinx_gallery_end_ignore

# %%
# PROPELLER
# ---------
#
# Each shot reads a rectangular blade of ``blade_width`` Cartesian lines, and
# the blades are rotated to cover k-space. Every blade samples the centre, so
# the shots can be registered against each other before reconstruction, which
# is what makes the family tolerant of motion between shots.

propeller = sequences.se_propeller2D_sequence(
    fov=220e-3, n=96, n_slices=1, blade_width=16, te=15e-3, tr=None, n_dummy=0
)
propeller.paper_plot(tr=1)

# %%
# Zero echo time
# --------------
#
# The readout gradient is already at its amplitude when the hard pulse is
# transmitted, so the acquisition begins without a ramp and the trajectory
# starts at the centre of k-space. One shell of views is written as a single
# continuous waveform that ramps up once at its first view and down once after
# its last, and each shot replays that waveform turned about z. The transmit
# and receive dead time after each pulse leaves the samples at the very centre
# of k-space unacquired, which the ``MissingSamples`` definition records.

zte = sequences.zte3D_sequence(fov=220e-3, n_x=64, flip_angle_deg=3.0, n_dummy=0)

# sphinx_gallery_start_ignore
trajectory_figure(
    {"PROPELLER, 16-line blades": propeller, "ZTE, radial half-spokes": zte}
)
# sphinx_gallery_end_ignore

# %%
# Stack of stars
# --------------
#
# A two-dimensional non-Cartesian trajectory in the plane and a Cartesian
# partition encode along z. The in-plane interleaf is one waveform for the
# whole scan and the partition encode is a scaled template, so a
# three-dimensional scan costs no more distinct gradient waveforms than the
# two-dimensional one it is built from.

stack_of_stars = sequences.gre_stack_of_stars3D_sequence(
    fov=220e-3, n=96, fov_z=96e-3, n_z=24, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
trajectory_figure({"stack of stars, $k_xk_y$": stack_of_stars}, plane="xy")
# sphinx_gallery_end_ignore

# %%
# Distinct waveforms against played blocks
# ----------------------------------------
#
# The shape column counts the entries in the sequence's shape library — the
# distinct compressed waveforms — and the rotation column the ``ROTATIONS``
# extension rows.
#
# The families that rotate a fixed interleaf keep under a dozen shapes whatever
# the angle count, because the angle is a rotation row rather than a waveform.
# Zero echo time is the exception in this table: its shell is one continuous
# waveform whose amplitude steps from view to view, so the samples of the shell
# are themselves the shape, and the library grows with the view count.

# sphinx_gallery_start_ignore
summarize(
    [
        ("radial, 128 spokes", radial),
        ("spiral, 16 interleaves", spiral),
        ("PROPELLER, 16-line blades", propeller),
        ("ZTE, 64 samples per view", zte),
        ("stack of stars, 24 partitions", stack_of_stars),
    ]
)
# sphinx_gallery_end_ignore
