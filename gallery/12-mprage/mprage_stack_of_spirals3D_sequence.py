"""
============================
3D stack-of-spirals MPRAGE
============================

One inversion per shot, followed by a spoiled gradient-echo train that
reads the spiral interleaves of one partition. An interleaf covers far more
of the plane than a line does, so a partition needs few readouts and the
whole train sits close behind the inversion.
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
        drawn = axis.scatter(arm, partition, c=value, cmap="turbo", s=10, linewidth=0)
        figure.colorbar(drawn, ax=axis, label=label, pad=0.02)
        axis.set_xlabel("interleaf index")
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
# The inversion, the inversion time, the interleaf train over one partition, and
# the recovery. The configuration below shortens the preparation and the train
# so that all of it stays visible at the width of this page.

import pypulseqpp as pp
from pypulseqpp.sequences import mprage_stack_of_spirals3D_sequence

compact = mprage_stack_of_spirals3D_sequence(
    n=96, n_z=8, n_shots=8, ti=0.06, tr=0.2, n_dummy=0
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
# is constant along each row. With an interleaf per readout the train is a
# few tens of readouts long rather than a few hundred, and every view is read
# within a short interval of the inversion time.

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
# The interleaves projected onto the plane, coloured by shot. Each one is
# turned from the last so that the set covers the plane.

pp.plot.plot_kspace(protocol, plane="xy", color_by="shot", show_trajectory=True)

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
            f"{grad.vector.value / protocol.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.vector.value / protocol.system.gamma:.0f} T/m/s",
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
