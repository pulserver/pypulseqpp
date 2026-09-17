"""
===============================
Echo-planar structural and fMRI
===============================

The two- and three-dimensional echo-planar sequences in a structural and in a
functional configuration.

Both configurations read an echo train after one excitation. A structural scan
buys resolution and geometric fidelity with segmentation, which shortens the
train each excitation has to cover; a functional scan buys volume repetition
time with a single shot and with undersampling along the encoded axes, and pays
for it in distortion along the phase encode. The parameters that separate them
are the shot count, the undersampling factors and the number of frames in the
time series.
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
    print(
        f"{'configuration':{width}}  {'blocks':>7}  {'duration':>9}  {'TE':>9}  {'TR':>10}"
    )
    for name, seq in rows:
        te, tr = seq.get_definition("TE"), seq.get_definition("TR")
        te = f"{1e3 * te[0]:.1f} ms" if te else "-"
        tr = f"{1e3 * tr[0]:.0f} ms" if tr else "-"
        print(
            f"{name:{width}}  {seq.num_blocks:7d}  "
            f"{seq.duration()[0]:7.2f} s  {te:>9}  {tr:>10}"
        )


def sampling_figure(sequences_by_name, plane="xy"):
    """Draw the ADC sampling locations of each sequence's first volume."""
    figure, axes = plt.subplots(
        1, len(sequences_by_name), figsize=(PAGE_WIDTH, 3.4), squeeze=False
    )
    for axis, (name, seq) in zip(axes[0], sequences_by_name.items(), strict=True):
        samples = seq.calculate_kspace()[0]
        first, second = "xyz".index(plane[0]), "xyz".index(plane[1])
        axis.plot(
            samples[first], samples[second], ".", ms=0.6, color="tab:blue", alpha=0.5
        )
        axis.set_title(name)
        axis.set_xlabel(f"$k_{plane[0]}$ (1/m)")
        axis.set_aspect("equal")
    axes[0][0].set_ylabel(f"$k_{plane[1]}$ (1/m)")
    figure.tight_layout()
    return figure


# sphinx_gallery_end_ignore
from pypulseqpp import sequences

# %%
# Two-dimensional structural
# --------------------------
#
# Four interleaved shots share each slice's phase-encode lines, so the train
# each excitation has to cover is a quarter as long and the accumulated
# off-resonance phase across it a quarter as large. Fat saturation precedes
# every shot, because a chemically shifted fat signal is displaced along the
# phase encode by the same low effective bandwidth that distorts the water
# image.

structural_2d = sequences.epi2D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    n_x=128,
    n_y=128,
    n_slices=5,
    slice_thickness=4e-3,
    n_shots=4,
    fat_saturation=True,
    tr=None,
    n_dummy=0,
)
structural_2d.paper_plot(tr=1)

# %%
# Two-dimensional functional
# --------------------------
#
# One shot per slice, twofold undersampling along the phase encode and a
# multiband factor of two, so each excitation covers two slices at once. The
# volume repetition time is prescribed rather than minimized, which is what
# fixes the sampling interval of the time series; ``n_frames`` volumes are
# played, each carrying its own ``REP`` counter.

functional_2d = sequences.epi2D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    n_x=64,
    n_y=64,
    n_slices=8,
    slice_thickness=3e-3,
    ry=2,
    multiband=2,
    n_frames=10,
    tr=1.0,
    fat_saturation=True,
    n_dummy=2,
)
functional_2d.paper_plot(tr=1)

# %%
# The two trains, in k-space. The structural shots interleave to a fully
# sampled grid; the functional one reads every second line of a grid half as
# large, and its calibration lines are read at the centre first.

# sphinx_gallery_start_ignore
sampling_figure(
    {"structural, 4 shots": structural_2d, "functional, 1 shot, ry=2": functional_2d}
)
# sphinx_gallery_end_ignore

# %%
# Three-dimensional structural
# ----------------------------
#
# The partition encode is grouped into shells ``rz`` partitions high, and each
# shell's lines are split between interleaved shots. A slab-selective
# excitation replaces the slice-selective one, and the third encoded axis is
# phase-encoded rather than excited slice by slice.

structural_3d = sequences.epi3D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    fov_z=96e-3,
    n_x=96,
    n_y=96,
    n_z=32,
    n_shots=4,
    excitation="slab",
    tr=None,
    n_dummy=0,
)
structural_3d.paper_plot(tr=1)

# %%
# Three-dimensional functional
# ----------------------------
#
# Undersampling along both phase-encoded axes, on the skipped-CAIPI lattice:
# the shot climbs ``caipi_shift`` partitions per acquired line, and the shift
# is chosen so that the aliases the lattice produces lie as far apart as
# possible (Stirnberg and Stöcker, Magn Reson Med 2021,
# doi:10.1002/mrm.28486). Sampling the third axis rather than exciting it slice
# by slice is what lets a whole volume be acquired in a few hundred
# milliseconds.

functional_3d = sequences.epi3D_sequence(
    fov_x=220e-3,
    fov_y=220e-3,
    fov_z=120e-3,
    n_x=64,
    n_y=64,
    n_z=24,
    ry=2,
    rz=2,
    n_frames=8,
    tr=None,
    n_dummy=2,
)
functional_3d.paper_plot(tr=1)

# %%
# The partition-encode sampling of the two three-dimensional configurations,
# in the plane the shots differ in:

# sphinx_gallery_start_ignore
sampling_figure(
    {"structural, 4 shots": structural_3d, "functional, ry=2, rz=2": functional_3d},
    plane="yz",
)
# sphinx_gallery_end_ignore

# %%
# The four side by side
# ---------------------
#
# The two functional configurations state their volume repetition time; the
# structural ones report the shortest one their shots admit. Comparing a
# functional TR with a structural one compares a prescription with a lower
# bound, not two measurements of the same quantity.

# sphinx_gallery_start_ignore
summarize(
    [
        ("2D structural, 128 x 128 x 5, 4 shots", structural_2d),
        ("2D functional, 64 x 64 x 8, MB2 ry2, 10 frames", functional_2d),
        ("3D structural, 96 x 96 x 32, 4 shots", structural_3d),
        ("3D functional, 64 x 64 x 24, ry2 rz2, 8 frames", functional_3d),
    ]
)
# sphinx_gallery_end_ignore
