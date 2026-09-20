"""
========================
3D Cartesian spin echo
========================

A slab-selective excitation and 180-degree refocusing pulse form one spin echo,
followed by a Cartesian ``(line, partition)`` readout. Spoilers suppress
unwanted coherence before the next TR. TE and TR determine T2 and longitudinal
recovery weighting. 3D spin echo supports high-resolution structural imaging.
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
# A full Cartesian sampling of the slab.

import pypulseqpp as pp
from pypulseqpp.sequences import se3D_sequence

baseline = se3D_sequence(n_x=160, n_y=160, n_z=32, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The phase-encode plane in the order it is read.

pp.plot.plot_kspace(baseline, color_by="order", plane="yz", show_trajectory=False)

# %%
# Acceleration on both encoded axes
# ---------------------------------
#
# Skipping lines and partitions leaves a spin echo's contrast alone, because
# the echo time is a property of one repetition rather than of the sampling.

alternative = se3D_sequence(
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
