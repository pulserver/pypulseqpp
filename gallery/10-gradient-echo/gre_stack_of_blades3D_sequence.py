"""
==================================
3D stack-of-blades gradient echo
==================================

A slab-selective spoiled excitation is followed by one line from a rotating
in-plane PROPELLER blade with Cartesian partition encoding. Gradient and RF
spoiling suppress residual transverse coherence. TR, flip angle, and TE
determine contrast. The overlapping blade centres support motion-robust 3D
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
# Sixteen-line blades
# -------------------
#
# Sixteen lines per blade, at every partition.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_stack_of_blades3D_sequence

baseline = gre_stack_of_blades3D_sequence(
    n=96, n_z=8, blade_width=16, tr=None, n_dummy=0
)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The blades of every partition, over the three k-space axes.

pp.plot.plot_kspace(baseline, color_by="shot")

# %%
# Wider blades
# ------------
#
# Increasing blade width reduces the number of blade orientations and
# increases the shared central-k-space region.

alternative = gre_stack_of_blades3D_sequence(
    n=96, n_z=8, blade_width=32, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("16 lines", baseline), ("32 lines", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot")

# %%
