"""
=========================
2D radial gradient echo
=========================

One full spoke through the centre of k-space per repetition. Every
readout crosses the centre, so the acquisition is insensitive to motion
between repetitions in a way a Cartesian one is not, and undersampling
shows as streaks rather than as aliasing.
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
# Enough spokes to sample the outer radius at the Nyquist spacing.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_radial2D_sequence

baseline = gre_radial2D_sequence(n=192, n_slices=1, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The spokes, coloured by the order they are played in. Consecutive spokes
# are spread over the disc rather than played side by side.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Angular undersampling
# ---------------------
#
# ``ry`` plays one spoke in three. The centre of k-space stays fully
# sampled, because every spoke crosses it; what thins out is the periphery.

alternative = gre_radial2D_sequence(
    n=192, n_slices=1, ry=3, te=None, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("Nyquist", baseline), ("ry = 3", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot", plane="xy")

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
            f"{grad.vector.value / baseline.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.vector.value / baseline.system.gamma:.0f} T/m/s",
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
