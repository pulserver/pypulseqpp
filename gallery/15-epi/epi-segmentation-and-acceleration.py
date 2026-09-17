r"""
=================================================
Echo train length and geometric distortion in EPI
=================================================

An echo-planar train samples the whole phase-encode axis after one excitation,
so off-resonance accumulates along that axis instead of across repetitions. A
spin at offset :math:`\Delta f` acquires phase :math:`2\pi \Delta f\, m\,
\mathrm{esp}` on echo :math:`m`, which is linear in :math:`k_y` and therefore a
displacement of

.. math::

    \delta y = \Delta f \cdot \mathrm{esp} \cdot N_\mathrm{etl}

pixels in the reconstructed image, whatever k-space step the train takes. The design controls the echo train length: interleaving the lines over several
shots and undersampling the phase encode both shorten it, and the two differ in
what else they change.

This example designs the configurations, reads the echo spacing and the train
length back from each designed sequence, and places them on the plane of
geometric distortion against volume acquisition time.
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


def table(rows):
    """Print one line per configuration."""
    width = max(len(row["label"]) for row in rows)
    print(
        f"{'configuration':{width}}  {'lines':>6}  {'esp':>8}  {'train':>9}  "
        f"{'shift':>9}  {'volume':>8}"
    )
    for row in rows:
        print(
            f"{row['label']:{width}}  {row['etl']:6d}  {1e3 * row['esp']:6.2f} ms  "
            f"{1e3 * row['train']:6.1f} ms  {row['shift_mm']:6.1f} mm  "
            f"{row['duration']:6.2f} s"
        )


def trade_off_figure(rows, title):
    """Distortion against volume time, one point per configuration."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 4.0))
    colours = {"segmented": "tab:blue", "accelerated": "tab:red", "both": "tab:purple"}
    for row in rows:
        axis.plot(
            row["duration"],
            row["shift_mm"],
            "o",
            ms=8,
            color=colours[row["kind"]],
        )
        axis.annotate(
            row["label"],
            (row["duration"], row["shift_mm"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    for kind, colour in colours.items():
        axis.plot([], [], "o", color=colour, label=kind)
    axis.set_xlabel("volume acquisition time (s)")
    axis.set_ylabel(f"displacement at {OFF_RESONANCE_HZ:.0f} Hz (mm)")
    axis.set_title(title)
    axis.set_ylim(bottom=0)
    axis.margins(x=0.22)
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


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

#: Off-resonance the displacement is quoted at, of the order of what a
#: sinus-adjacent voxel sees at 3 T.
OFF_RESONANCE_HZ = 100.0

FOV = 220e-3
MATRIX = 128

# %%
# Two-dimensional configurations
# ------------------------------
#
# Each configuration is designed at the same matrix, field of view and slice
# count, so the only quantities that differ are the shot count and the in-plane
# acceleration. ``tr=None`` asks for the shortest repetition the design admits,
# which makes the reported duration a property of the sampling rather than a
# prescription.


def design_2d(n_shots, ry):
    """A five-slice echo-planar sequence at the common geometry."""
    return sequences.epi2D_sequence(
        fov_x=FOV,
        fov_y=FOV,
        n_x=MATRIX,
        n_y=MATRIX,
        n_slices=5,
        slice_thickness=4e-3,
        n_shots=n_shots,
        ry=ry,
        fat_saturation=True,
        tr=None,
        n_dummy=0,
    )


designs_2d = {
    "single shot": design_2d(1, 1),
    "2 shots": design_2d(2, 1),
    "4 shots": design_2d(4, 1),
    "Ry = 2": design_2d(1, 2),
    "Ry = 3": design_2d(1, 3),
    "2 shots, Ry = 2": design_2d(2, 2),
}

# %%
# The echo spacing and the train length are written into the sequence by the
# design, so the displacement is read off the sequence rather than recomputed
# from the parameters that were asked for.


def measure(designs, matrix):
    """Echo spacing, train length, displacement and duration of each design."""
    rows = []
    for label, seq in designs.items():
        esp = seq.get_definition("EchoSpacing")[0]
        etl = int(seq.get_definition("EPIFactor")[0])
        rows.append(
            {
                "label": label,
                "esp": esp,
                "etl": etl,
                "train": esp * etl,
                "shift_mm": 1e3 * OFF_RESONANCE_HZ * esp * etl * FOV / matrix,
                "duration": seq.duration()[0],
                "kind": kind(label),
            }
        )
    return rows


def kind(label):
    """Whether a configuration is segmented, accelerated, or both."""
    accelerated = "Ry" in label
    if accelerated and ("shots" in label or "Rz" in label):
        return "both"
    return "accelerated" if accelerated else "segmented"


rows_2d = measure(designs_2d, MATRIX)

# sphinx_gallery_start_ignore
table(rows_2d)
# sphinx_gallery_end_ignore

# %%
# Segmentation and undersampling reach the same train length by different
# routes, and the plane separates them. Four interleaved shots and
# :math:`R_y = 2` with two shots both read 32 echoes per excitation and give
# the same displacement; the segmented configuration acquires every line and
# takes twice as long, the accelerated one acquires half of them, needs a
# sensitivity-encoded reconstruction and loses signal-to-noise by at least
# :math:`\sqrt{R_y}`. Segmentation also makes the image sensitive to phase
# differences between shots, which appear as ghosts along the phase encode
# rather than as displacement.

# sphinx_gallery_start_ignore
trade_off_figure(rows_2d, "2D echo-planar, 128 x 128, 5 slices")
# sphinx_gallery_end_ignore

# %%
# The two routes differ in what they leave on the sampling grid. Four shots
# interleave to a complete grid; twofold undersampling reads every second line
# of the same grid, and the calibration lines at the centre are read first.

# sphinx_gallery_start_ignore
sampling_figure({"4 shots": designs_2d["4 shots"], "Ry = 2": designs_2d["Ry = 2"]})
# sphinx_gallery_end_ignore

# %%
# Three-dimensional configurations
# --------------------------------
#
# A slab-selective excitation and a second phase-encoded axis give a second
# acceleration factor, and the two are not interchangeable. The train runs
# along :math:`k_y` within one partition, so :math:`R_y` shortens it and
# :math:`R_z` does not: undersampling the partition axis removes whole trains
# from the volume. The skipped-CAIPI lattice shifts the partition index by
# ``caipi_shift`` per acquired line so that the aliases the lattice produces
# lie as far apart as possible (Stirnberg and Stöcker, Magn Reson Med 2021,
# doi:10.1002/mrm.28486).


def design_3d(n_shots, ry, rz):
    """A slab-selective volumetric echo-planar sequence."""
    return sequences.epi3D_sequence(
        fov_x=FOV,
        fov_y=FOV,
        fov_z=120e-3,
        n_x=96,
        n_y=96,
        n_z=24,
        n_shots=n_shots,
        ry=ry,
        rz=rz,
        excitation="slab",
        tr=None,
        n_dummy=0,
    )


designs_3d = {
    "single shot": design_3d(1, 1, 1),
    "4 shots": design_3d(4, 1, 1),
    "Ry = 2": design_3d(1, 2, 1),
    "Ry = 2, Rz = 2": design_3d(1, 2, 2),
    "Ry = 3, Rz = 2": design_3d(1, 3, 2),
}

rows_3d = measure(designs_3d, 96)

# sphinx_gallery_start_ignore
table(rows_3d)
trade_off_figure(rows_3d, "3D echo-planar, 96 x 96 x 24")
# sphinx_gallery_end_ignore

# %%
# Configurations sharing a train length sit at the same height, and doubling
# :math:`R_z` halves the volume time without changing it. The two factors are
# therefore independent design variables: :math:`R_y` sets the displacement,
# :math:`R_z` sets the volume time, and their product sets the signal-to-noise
# loss and the conditioning of the reconstruction.
