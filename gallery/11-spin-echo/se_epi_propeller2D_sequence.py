"""
================================================
2D PROPELLER spin echo with echo-planar blades
================================================

A slice-selective excitation and 180-degree refocusing pulse form a spin echo,
followed by an echo-planar readout of one rotating PROPELLER blade. Spoilers
suppress unwanted coherence between shots. TE controls T2 weighting, while the
EPI train introduces off-resonance sensitivity. This sequence supports rapid,
motion-robust structural imaging.
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
# Sixteen lines to a blade, each blade read after one excitation.

import pypulseqpp as pp
from pypulseqpp.sequences import se_epi_propeller2D_sequence

baseline = se_epi_propeller2D_sequence(
    n_x=192, blade_width=16, n_slices=1, te=60e-3, tr=None, n_dummy=0
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
# The blades in the order they are played; the lines within a blade are read
# in one train.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Wider blades
# ------------
#
# A wider blade is a longer echo-planar train, so the blade covers more of
# the disc and accumulates more off-resonance phase across itself.

alternative = se_epi_propeller2D_sequence(
    n_x=192, blade_width=24, n_slices=1, te=60e-3, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("16 lines", baseline), ("24 lines", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot", plane="xy")

# %%
