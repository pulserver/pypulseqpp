"""
============================
3D Cartesian gradient echo
============================

A 3D spoiled gradient-echo (SPGR) acquisition applies a low-flip-angle slab
excitation before one Cartesian ``(line, partition)`` readout per TR. Gradient
spoiling and quadratic RF/receiver phase cycling suppress coherent residual
transverse magnetisation. TR, flip angle, and TE determine the T1 and T2*
weighting. This sequence is used for high-resolution T1-weighted structural
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
# Fully sampled acquisition
# -------------------------
#
# Every ``(line, partition)`` view inside the ellipse inscribed in the
# phase-encode plane is acquired.

import pypulseqpp as pp
from pypulseqpp.sequences import gre3D_sequence

baseline = gre3D_sequence(n_x=160, n_y=160, n_z=32, te=None, tr=None, n_dummy=0)
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
# Colour encodes acquisition order in the phase-encode plane. All lines of
# one partition are acquired before the next partition.

pp.plot.plot_kspace(baseline, color_by="order", plane="yz", show_trajectory=False)

# %%
# Acceleration on both encoded axes
# ---------------------------------
#
# ``ry`` and ``rz`` subsample the line and partition axes independently.
# With ``ry=rz=2``, the phase-encode plane outside the central calibration
# region is acquired in approximately one quarter of the repetitions; the
# calibration region remains fully sampled.

alternative = gre3D_sequence(
    n_x=160, n_y=160, n_z=32, ry=2, rz=2, te=None, tr=None, n_dummy=0
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
