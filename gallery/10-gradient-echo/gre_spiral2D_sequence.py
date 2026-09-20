"""
=========================
2D spiral gradient echo
=========================

A spoiled low-flip-angle excitation is followed by one spiral interleaf.
Gradient and RF spoiling suppress residual transverse coherence between
repetitions. TR and flip angle primarily determine T1 weighting; off-resonance
and T2* decay affect the spiral readout. Spiral SPGR supports rapid dynamic and
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
# Sixteen interleaves at a constant pitch, which sample the disc at the
# Nyquist spacing.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_spiral2D_sequence

baseline = gre_spiral2D_sequence(
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
# Each interleaf is the same solved arm turned to its own angle.

pp.plot.plot_kspace(baseline, color_by="shot", plane="xy")

# %%
# Variable density
# ----------------
#
# ``density='dual'`` designs the pitch so that the centre keeps the Nyquist
# spacing while the periphery is sampled more sparsely, which shortens the
# arm at the cost of aliasing that a reconstruction has to handle.

alternative = gre_spiral2D_sequence(
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
