"""
==================
2D balanced SSFP
==================

Every gradient axis returns to zero moment within each repetition and the
RF phase alternates, so the magnetisation reaches a steady state that carries
both relaxation times. The train opens with a half flip, which places the
magnetisation on the axis the steady state oscillates about.
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
# One cardiac phase over a full Cartesian sampling.

import pypulseqpp as pp
from pypulseqpp.sequences import bssfp2D_sequence

baseline = bssfp2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_phases=1, tr=None, n_dummy=0
)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(f"TR {baseline.get_definition('TR')[0] * 1e3:.2f} ms")

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The lines in the order they are read, in segments of ``views_per_segment``.

pp.plot.plot_kspace(baseline, color_by="order", plane="xy", show_trajectory=False)

# %%
# Cine
# ----
#
# ``n_phases`` reads the same segment of lines at several points after the
# trigger, so one breath-hold resolves the cardiac cycle. The segment length
# is what trades temporal resolution against the number of heartbeats the
# scan takes.

alternative = bssfp2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_phases=8, views_per_segment=12, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("1 phase", baseline), ("8 phases", alternative)):
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
