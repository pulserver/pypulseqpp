"""
=======================================
3D Cartesian multi-echo gradient echo
=======================================

A slab-selective spoiled excitation is followed by several gradient echoes of
one Cartesian ``(line, partition)`` view. Gradient and RF spoiling suppress
residual transverse coherence between repetitions. The multiple echo times
sample T2* decay; TR and flip angle determine the T1 weighting. Applications
include high-resolution structural imaging, R2* mapping, and QSM.
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
# Four echoes per excitation over a slab.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_multiecho3D_sequence

baseline = gre_multiecho3D_sequence(
    n_x=160, n_y=160, n_z=32, n_echoes=4, tr=None, n_dummy=0
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
# Colour encodes acquisition order in the phase-encode plane.

pp.plot.plot_kspace(baseline, color_by="order", plane="yz", show_trajectory=False)

# %%
# Acceleration on both encoded axes
# ---------------------------------
#
# Subsampling both phase-encode axes reduces the number of repetitions. The
# complete echo train remains within each retained repetition.

alternative = gre_multiecho3D_sequence(
    n_x=160, n_y=160, n_z=32, n_echoes=4, ry=2, rz=2, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("full", baseline), ("2 x 2", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="order", plane="yz", show_trajectory=False)

# %%
