"""
==========================
3D stack-of-stars MPRAGE
==========================

One inversion per shot, followed by a spoiled gradient-echo train that
reads the radial spokes of one partition. In-plane the acquisition is
radial, so every spoke crosses the centre of k-space and the contrast the
inversion time sets is carried by every readout rather than by a few
central lines.
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


def _views(seq, n_z):
    """Every acquisition as (spoke index, partition, index in the train, cycle)."""
    labels = seq.evaluate_labels(evolution="adc")
    index = np.asarray(labels["ECO"])
    return (
        np.asarray(labels["LIN"]),
        np.asarray(labels["PAR"]) - n_z // 2,
        index,
        np.cumsum(index == 0) - 1,
    )


def order_figure(seq, n_z):
    """Which spoke each partition reads, coloured by position in the train."""
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.4), sharey=True)
    arm, partition, index, cycle = _views(seq, n_z)
    for axis, value, label in zip(
        axes,
        (index, cycle),
        ("Readout index in train", "Inversion cycle"),
        strict=True,
    ):
        drawn = axis.scatter(arm, partition, c=value, cmap="turbo", s=10, linewidth=0)
        figure.colorbar(drawn, ax=axis, label=label, pad=0.02)
        axis.set_xlabel("spoke index")
        axis.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("$k_z$ (partitions from centre)")
    figure.tight_layout()
    return figure


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


# sphinx_gallery_end_ignore

# %%
# Timing structure
# ----------------
#
# The inversion, the inversion time, the spoke train over one partition, and
# the recovery. ``ti=None`` and ``tr=None`` take the shortest inversion time and
# recovery the modules admit, and an angular undersampling that leaves four
# spokes per partition makes a train short enough to read at the width of this
# page.

import pypulseqpp as pp
from pypulseqpp.sequences import mprage_stack_of_stars3D_sequence

compact = mprage_stack_of_stars3D_sequence(
    n=64, n_z=4, ry=32, ti=None, tr=None, n_dummy=0
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
# One inversion reads the spokes of one partition, so the inversion cycle is
# constant along each row. The index within the train runs over the spokes in
# the order they are played, and ``partition_angle_shift`` turns the set from
# one partition to the next so that the spokes of neighbouring partitions do
# not coincide. ``TI`` ends at the first excitation-pulse centre; the centre of
# k-space on its spoke is sampled at ``TI + TE``.

protocol = mprage_stack_of_stars3D_sequence(
    n=192, n_z=16, ry=4, ti=0.9, tr=2.3, n_dummy=0
)

# sphinx_gallery_start_ignore
order_figure(protocol, 16)
# sphinx_gallery_end_ignore

# %%
# Trajectory
# ----------
#
# The spokes of every partition, over the three k-space axes, coloured by shot.

pp.plot.plot_kspace(protocol, color_by="shot")

# %%
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model below is a demonstration, not a
# scanner's.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
grad_ok, grad = safety.check_max_grad(protocol)
slew_ok, slew = safety.check_max_slew(protocol)
cont_ok, cont = safety.check_grad_continuity(protocol)
pns_ok, pns = safety.check_pns(protocol, model)

# sphinx_gallery_start_ignore
safety_table(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / protocol.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.per_axis.value / protocol.system.gamma:.0f} T/m/s",
        ),
        (
            "gradient continuity",
            cont_ok,
            f"{len(cont.discontinuities)} discontinuities",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
    ]
)
# sphinx_gallery_end_ignore
