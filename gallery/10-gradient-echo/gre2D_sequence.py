"""
============================
2D Cartesian gradient echo
============================

One excitation and one phase-encode line per repetition, with the
transverse magnetisation spoiled by a gradient and by a quadratic RF phase
increment before the next excitation. The workhorse of the family, and the
sequence the other Cartesian variants are read against.
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
# A full Cartesian sampling of one slice.

import pypulseqpp as pp
from pypulseqpp.sequences import gre2D_sequence

baseline = gre2D_sequence(n_x=192, n_y=192, n_slices=1, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"TE {baseline.get_definition('TE')[0] * 1e3:.2f} ms, "
    f"TR {baseline.get_definition('TR')[0] * 1e3:.2f} ms"
)

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The lines are dealt so that the centre of k-space is read near the middle
# of the scan, which is what the colour by acquisition order shows.

pp.plot.plot_kspace(baseline, color_by="order", plane="xy", show_trajectory=False)

# %%
# In-plane acceleration
# ---------------------
#
# ``ry`` reads one line in two and adds a fully sampled calibration region at
# the centre, which a parallel-imaging reconstruction needs to estimate the
# coil sensitivities from.

alternative = gre2D_sequence(
    n_x=192, n_y=192, n_slices=1, ry=2, n_acs_y=24, te=None, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("full", baseline), ("ry = 2", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="order", plane="xy", show_trajectory=False)

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
