"""
========================
2D Cartesian spin echo
========================

A slice-selective excitation and 180-degree refocusing pulse form one spin echo,
followed by a Cartesian readout. Spoilers suppress unwanted coherence before
the next TR. TE controls T2 weighting and TR controls longitudinal recovery.
Spin echo is used for conventional T1-, T2-, and proton-density-weighted
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
# A full Cartesian sampling of one slice, at the shortest echo time the
# pulses and the readout allow.

import pypulseqpp as pp
from pypulseqpp.sequences import se2D_sequence

baseline = se2D_sequence(n_x=192, n_y=192, n_slices=1, te=None, tr=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"TE {baseline.get_definition('TE')[0] * 1e3:.2f} ms, "
    f"TR {baseline.get_definition('TR')[0] * 1e3:.2f} ms"
)

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The lines in the order they are read.

pp.plot.plot_kspace(baseline, color_by="order", plane="xy", show_trajectory=False)

# %%
# Partial Fourier
# ---------------
#
# ``partial_fourier_y`` omits the lines furthest from the centre on one side
# and leaves the reconstruction to use the conjugate symmetry of k-space to
# replace them, which shortens the scan at the cost of noise and of
# sensitivity to the phase the object carries.

alternative = se2D_sequence(
    n_x=192, n_y=192, n_slices=1, partial_fourier_y=0.75, te=None, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("full", baseline), ("6/8 along y", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="order", plane="xy", show_trajectory=False)

# %%
