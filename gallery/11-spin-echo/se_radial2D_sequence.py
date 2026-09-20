"""
=====================
2D radial spin echo
=====================

A slice-selective excitation and 180-degree refocusing pulse form one spin echo,
followed by a radial spoke through k-space centre. Spoilers suppress unwanted
coherence before the next TR. TE controls T2 weighting and TR controls
longitudinal recovery. Radial spin echo supports motion-robust structural
imaging.
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
# Enough spokes to sample the outer radius at the Nyquist spacing.

import pypulseqpp as pp
from pypulseqpp.sequences import se_radial2D_sequence

baseline = se_radial2D_sequence(n=192, n_slices=1, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# Colour encodes spoke acquisition order.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Angular undersampling
# ---------------------
#
# Retaining one spoke angle in three reduces peripheral angular sampling;
# every acquired spoke still crosses the k-space origin.

alternative = se_radial2D_sequence(n=192, n_slices=1, ry=3, te=None, tr=None, n_dummy=0)

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
