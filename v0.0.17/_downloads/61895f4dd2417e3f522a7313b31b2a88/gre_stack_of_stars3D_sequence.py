"""
=================================
3D stack-of-stars gradient echo
=================================

A slab-selective spoiled excitation is followed by one radial spoke with
Cartesian partition encoding. Gradient and RF spoiling suppress residual
transverse coherence between repetitions. TR and flip angle primarily
determine T1 weighting. Stack-of-stars SPGR is used for motion-robust 3D
structural and dynamic imaging.
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
# Nyquist angular sampling
# ------------------------
#
# The Nyquist set of spoke angles at every partition. With the default
# ``partition_angle_shift='none'``, every partition uses the same angles.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_stack_of_stars3D_sequence

baseline = gre_stack_of_stars3D_sequence(n=96, n_z=8, tr=None, n_dummy=0)
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
# Retaining one spoke angle in four reduces the number of repetitions
# fourfold. Every acquired spoke samples the origin of its partition.

alternative = gre_stack_of_stars3D_sequence(n=96, n_z=8, ry=4, tr=None, n_dummy=0)

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
