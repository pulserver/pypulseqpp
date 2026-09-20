"""
========================
2D PROPELLER spin echo
========================

A slice-selective excitation and 180-degree refocusing pulse form one spin echo,
which reads one line of a rotating PROPELLER blade. Spoilers suppress unwanted
coherence before the next TR. TE controls T2 weighting and TR controls
longitudinal recovery. The overlapping blade centres support motion-robust
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
# Sixteen lines to a blade.

import pypulseqpp as pp
from pypulseqpp.sequences import se_propeller2D_sequence

baseline = se_propeller2D_sequence(
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
# Colour encodes blade acquisition order.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Wider blades
# ------------
#
# A wider blade needs fewer turns to cover the disc and samples more of the
# centre in each.

alternative = se_propeller2D_sequence(
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
