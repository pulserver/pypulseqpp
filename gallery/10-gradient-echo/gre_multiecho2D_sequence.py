"""
=======================================
2D Cartesian multi-echo gradient echo
=======================================

A spoiled low-flip-angle excitation is followed by several Cartesian gradient
echoes of the same phase-encode line. Gradient and RF spoiling suppress
residual transverse coherence before the next TR. The echo train samples T2*
decay while TR and flip angle determine the T1 weighting. Multi-echo GRE is
used for T2*/R2* mapping, susceptibility mapping, and structural imaging.
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
# Four echoes after one excitation, read in alternating directions.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_multiecho2D_sequence

baseline = gre_multiecho2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_echoes=4, te=None, tr=None, n_dummy=0
)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print("TE", [round(t * 1e3, 2) for t in baseline.get_definition("TE")], "ms")

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# Colouring by echo index separates the echoes of one excitation; colouring by
# shot separates the excitations.

pp.plot.plot_kspace(baseline, color_by="order", plane="xy", show_trajectory=False)

# %%
# A longer echo train
# -------------------
#
# More echoes sample the decay further into it, at the cost of a longer
# repetition and a later last echo.

alternative = gre_multiecho2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_echoes=8, te=None, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("4 echoes", baseline), ("8 echoes", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="order", plane="xy", show_trajectory=False)

# %%
