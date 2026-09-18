"""
=============================
3D stack-of-stars spin echo
=============================

Radial spokes in the plane and Cartesian encoding along the slab axis,
read at the refocused echo.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


# sphinx_gallery_end_ignore

# %%
# Baseline
# --------
#
# A spoke at every partition.

import pypulseqpp as pp
from pypulseqpp.sequences import se_stack_of_stars3D_sequence

baseline = se_stack_of_stars3D_sequence(n=96, n_z=8, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The spokes of every partition, over the three k-space axes.

pp.plot.plot_kspace(baseline, color_by="shot")

# %%
# Angular undersampling
# ---------------------
#
# One spoke in four, which shortens the scan fourfold.

alternative = se_stack_of_stars3D_sequence(
    n=96, n_z=8, ry=4, te=None, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("Nyquist", baseline), ("ry = 4", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot")

# %%
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model below is a demonstration, not a
# scanner's.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
grad_ok, grad = safety.check_max_grad(baseline)
slew_ok, slew = safety.check_max_slew(baseline)
cont_ok, cont = safety.check_grad_continuity(baseline)
pns_ok, pns = safety.check_pns(baseline, model)

# sphinx_gallery_start_ignore
safety_table(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / baseline.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.per_axis.value / baseline.system.gamma:.0f} T/m/s",
        ),
        (
            "gradient continuity",
            cont_ok,
            f"{len(cont.discontinuities)} discontinuities",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
    ],
)
# sphinx_gallery_end_ignore
