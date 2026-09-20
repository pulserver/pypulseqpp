"""
=====================
2D spiral spin echo
=====================

A slice-selective excitation and 180-degree refocusing pulse form one spin echo,
followed by a spiral interleaf. Spoilers suppress unwanted coherence before
the next TR. TE controls T2 weighting; off-resonance affects the spiral
readout. Spiral spin echo supports rapid T2-weighted structural imaging.
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
# Sixteen interleaves at a constant pitch.

import pypulseqpp as pp
from pypulseqpp.sequences import se_spiral2D_sequence

baseline = se_spiral2D_sequence(
    n=192, n_shots=16, n_slices=1, te=None, tr=None, n_dummy=0
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
# Each interleaf is the solved arm turned to its own angle.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Variable density
# ----------------
#
# A dual-density arm retains the Nyquist spacing near the origin and increases
# the pitch at larger radii, reducing the readout duration.

alternative = se_spiral2D_sequence(
    n=192,
    n_shots=16,
    n_slices=1,
    density="dual",
    periphery_undersampling=2.0,
    te=None,
    tr=None,
    n_dummy=0,
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("constant", baseline), ("dual density", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot", plane="xy")

# %%
