"""
============================
3D stack-of-spirals MPRAGE
============================

An inversion preparation is followed by a train of low-flip-angle spoiled
spiral gradient echoes with Cartesian partition encoding. Interleaf and
partition order determine the recovery time of the acquired data within and
between inversion cycles. Stack-of-spirals MPRAGE provides rapid T1-weighted
3D structural imaging.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypulseqpp.plot import SAMPLING

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


def _views(seq, n_z):
    """Every acquisition as (interleaf index, partition, index in the train, cycle)."""
    labels = seq.evaluate_labels(evolution="adc")
    index = np.asarray(labels["ECO"])
    return (
        np.asarray(labels["LIN"]),
        np.asarray(labels["PAR"]) - n_z // 2,
        index,
        np.cumsum(index == 0) - 1,
    )


def order_figure(seq, n_z):
    """Which interleaf each partition reads, coloured by position in the train."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.4), sharey=True)
    arm, partition, index, cycle = _views(seq, n_z)
    for axis, value, label in zip(
        axes,
        (index, cycle),
        ("Readout index in train", "Inversion cycle"),
        strict=True,
    ):
        drawn = axis.scatter(arm, partition, c=value, cmap=SAMPLING, s=10, linewidth=0)
        figure.colorbar(drawn, ax=axis, label=label, pad=0.02)
        axis.set_xlabel("interleaf index")
        axis.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("$k_z$ (partitions from centre)")
    figure.tight_layout()
    return figure


# sphinx_gallery_end_ignore

# %%
# Timing structure
# ----------------
#
# Each cycle comprises inversion, inversion delay, an interleaf train at one
# partition and a recovery interval. ``ti=None`` and ``tr=None`` use the
# shortest timing supported by the modules. Four interleaves per partition
# produce a compact timing diagram.

import pypulseqpp as pp
from pypulseqpp.sequences import mprage_stack_of_spirals3D_sequence

compact = mprage_stack_of_spirals3D_sequence(
    n=96, n_z=4, n_shots=4, ti=None, tr=None, n_dummy=0
)
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
# One inversion reads the interleaves of one partition, so the inversion cycle
# is constant along each row. The interleaf train contains substantially fewer
# readouts than an equivalent Cartesian line train, reducing the range of
# inversion-recovery weighting. ``TI`` ends at the first excitation-pulse
# centre; the first interleaf reaches the centre of k-space at ``TI + TE``.

protocol = mprage_stack_of_spirals3D_sequence(
    n=192, n_z=16, n_shots=16, ti=0.9, tr=2.3, n_dummy=0
)

# sphinx_gallery_start_ignore
order_figure(protocol, 16)
# sphinx_gallery_end_ignore

# %%
# Trajectory
# ----------
#
# The interleaves of every partition, over the three k-space axes, coloured by
# shot. Each one is turned from the last so that the set covers the plane.

pp.plot.plot_kspace(protocol, color_by="shot")

# %%
