"""
==================
3D balanced SSFP
==================

A slab-selective low-flip-angle excitation and balanced Cartesian readout repeat
with alternating RF phase and zero net gradient moment in every TR. The
preserved transverse coherence establishes a high-SNR steady state governed by
T2/T1 and off-resonance. A half-flip preparation reduces transient oscillation.
3D bSSFP is used for high-SNR structural imaging.
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
from pypulseqpp.sequences import bssfp3D_sequence

baseline = bssfp3D_sequence(n_x=160, n_y=160, n_z=32, tr=None)
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
# Skipping lines and partitions shortens the scan without disturbing the
# steady state, which depends on the repetition time and the flip angle
# rather than on which view is read.

alternative = bssfp3D_sequence(n_x=160, n_y=160, n_z=32, ry=2, rz=2, tr=None)

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
