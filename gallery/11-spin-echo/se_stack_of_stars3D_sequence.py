"""
=============================
3D stack-of-stars spin echo
=============================

A slab-selective excitation and 180-degree refocusing pulse form one spin echo,
followed by a radial spoke with Cartesian partition encoding. Spoilers suppress
unwanted coherence between repetitions. TE and TR determine T2 and
longitudinal recovery weighting. Stack-of-stars spin echo supports
motion-robust 3D structural imaging.
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
