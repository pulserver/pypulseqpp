"""
============================
2D PROPELLER gradient echo
============================

A spoiled low-flip-angle excitation is followed by one Cartesian line from a
rotating PROPELLER blade. Gradient and RF spoiling suppress residual transverse
coherence between repetitions. TR, flip angle, and TE determine contrast. The
overlapping central k-space region supports motion estimation in structural
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
# Sixteen lines to a blade, at enough blades to cover the disc.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_propeller2D_sequence

baseline = gre_propeller2D_sequence(
    n=192, blade_width=16, n_slices=1, te=None, tr=None, n_dummy=0
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
# The blades, coloured by the order they are played in. Each blade is a band
# of parallel lines; the bands overlap at the centre.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Wider blades
# ------------
#
# A wider blade covers more of the disc, so fewer blades are needed and each
# takes longer. What one blade samples of the centre grows with its width,
# which is what the motion correction works from.

alternative = gre_propeller2D_sequence(
    n=192, blade_width=32, n_slices=1, te=None, tr=None, n_dummy=0
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
pp.plot.plot_kspace(alternative, color_by="shot", plane="xy")

# %%
