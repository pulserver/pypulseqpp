"""
============================
2D Cartesian gradient echo
============================

A spoiled gradient-echo (SPGR) acquisition applies one low-flip-angle
slice-selective excitation before an unbalanced Cartesian readout in each TR.
Gradient spoiling and quadratic RF/receiver phase cycling suppress coherent
residual transverse magnetisation. TR and flip angle primarily determine T1
weighting, with T2* decay during TE. SPGR is widely used for T1-weighted
structural imaging.
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
