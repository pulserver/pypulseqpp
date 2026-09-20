"""
====================
3D Cartesian MPRAGE
====================

An inversion preparation is followed after the prescribed inversion delay by a
train of low-flip-angle spoiled Cartesian gradient echoes. The inversion time
is measured to the first excitation centre; the corresponding central ADC
sample occurs one TE later. The ordering assigns recovery times within each
inversion cycle to ``(line, partition)`` views. MPRAGE is used for
high-resolution 3D T1-weighted structural imaging.
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
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def _views(seq, n_y, n_z):
    """Every acquisition as (line, partition, index in the train, inversion cycle)."""
    labels = seq.evaluate_labels(evolution="adc")
    index = np.asarray(labels["ECO"])
    return (
        np.asarray(labels["LIN"]) - n_y // 2,
        np.asarray(labels["PAR"]) - n_z // 2,
        index,
        np.cumsum(index == 0) - 1,
    )


def order_figure(seq, n_y, n_z):
    """The index in the train and the inversion cycle of every view."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6), sharey=True)
    line, partition, index, cycle = _views(seq, n_y, n_z)
    for axis, value, label in zip(
        axes,
        (index, cycle),
        ("Readout index in train", "Inversion cycle"),
        strict=True,
    ):
        drawn = axis.scatter(line, partition, c=value, cmap="turbo", s=9, linewidth=0)
        figure.colorbar(drawn, ax=axis, label=label, pad=0.02)
        axis.set_xlabel("$k_y$ (lines from centre)")
        axis.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("$k_z$ (partitions from centre)")
    figure.tight_layout()
    return figure


# sphinx_gallery_end_ignore

# %%
# Timing structure
# ----------------
#
# The inversion, its crusher, the inversion time, the gradient-echo train and
# the recovery that closes the cycle. ``ti=None`` and ``tr=None`` take the
# shortest inversion time and recovery the modules admit, and four lines per
# partition make a train short enough to read at the width of this page; a
# protocol uses an inversion time of several hundred milliseconds and a train
# of a hundred or more readouts.

from pypulseqpp.sequences import mprage3D_sequence

compact = mprage3D_sequence(n_x=128, n_y=4, n_z=4, ti=None, tr=None, n_dummy=0)
print(
    f"{compact.num_blocks} blocks, {compact.duration()[0]:.2f} s, "
    f"TI {compact.get_definition('TI')[0] * 1e3:.0f} ms, "
    f"TR {compact.get_definition('TR')[0] * 1e3:.0f} ms"
)

# %%
compact.paper_plot()

# %%
# Sampling order
# --------------
#
# At a protocol matrix, one inversion reads the sampled lines of one partition:
# the inversion cycle is constant along each row of the map, and the index
# within the train runs outward from the centre of the line axis. The centre of
# k-space is therefore read at the start of a train, one inversion time after
# the inversion, which is what sets the contrast; the periphery is read later,
# as the magnetisation continues to recover.

protocol = mprage3D_sequence(n_x=192, n_y=128, n_z=24, ti=0.9, tr=2.3, n_dummy=0)
print(
    f"{protocol.duration()[0]:.1f} s, "
    f"{int(np.asarray(protocol.evaluate_labels(evolution='adc')['ECO']).max()) + 1} "
    "readouts in the longest train"
)

# sphinx_gallery_start_ignore
order_figure(protocol, 128, 24)
# sphinx_gallery_end_ignore

# %%
# Accelerated sampling
# --------------------
#
# ``ry`` and ``rz`` skip lines and partitions, and ``caipi_shift`` moves each
# line's partitions so that the aliases land away from one another. The train
# shortens with the number of lines each partition keeps, so every view is read
# closer to the inversion and the contrast the inversion time sets is carried
# further into k-space. The scan time does not follow: it is the number of
# inversion cycles times the repetition time, and with the calibration region
# fully sampled every partition is still visited.

accelerated = mprage3D_sequence(
    n_x=192, n_y=128, n_z=24, ry=2, rz=2, caipi_shift=1, ti=0.9, tr=2.3, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':14} {'views':>7} {'cycles':>8} {'per train':>10} {'scan (s)':>9}")
for name, seq in (("1 x 1", protocol), ("2 x 2, shift 1", accelerated)):
    _, _, index, cycle = _views(seq, 128, 24)
    print(
        f"{name:14} {index.size:7d} {int(cycle.max()) + 1:8d} "
        f"{index.size / (int(cycle.max()) + 1):10.1f} {seq.duration()[0]:9.1f}"
    )
order_figure(accelerated, 128, 24)
# sphinx_gallery_end_ignore

# %%
