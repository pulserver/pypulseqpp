"""
=========================
2D radial gradient echo
=========================

A spoiled low-flip-angle excitation is followed by one radial spoke through
k-space centre. Gradient and RF spoiling suppress residual transverse coherence
before the next TR. TR and flip angle primarily determine T1 weighting, with
T2* decay during TE. Radial SPGR is used for motion-robust dynamic and
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
# Enough spokes to sample the outer radius at the Nyquist spacing.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_radial2D_sequence

baseline = gre_radial2D_sequence(n=192, n_slices=1, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")


# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The spokes, coloured by the order they are played in. Consecutive spokes
# are spread over the disc rather than played side by side.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Angular undersampling
# ---------------------
#
# ``ry`` plays one spoke in three. The centre of k-space stays fully
# sampled, because every spoke crosses it; what thins out is the periphery.

alternative = gre_radial2D_sequence(
    n=192, n_slices=1, ry=3, te=None, tr=None, n_dummy=0
)

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
