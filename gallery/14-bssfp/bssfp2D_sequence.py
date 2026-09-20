"""
==================
2D balanced SSFP
==================

A low-flip-angle excitation and balanced Cartesian gradient-echo readout repeat
with alternating RF phase. Zero net gradient moment in every TR preserves
transverse coherence and establishes a steady state governed by T2/T1 and
off-resonance. A half-flip preparation reduces transient oscillation. 2D bSSFP
is widely used for cardiac cine and dynamic cardiac imaging.
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
# One cardiac phase over a full Cartesian sampling.

import pypulseqpp as pp
from pypulseqpp.sequences import bssfp2D_sequence

baseline = bssfp2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_phases=1, tr=None, n_dummy=0
)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(f"TR {baseline.get_definition('TR')[0] * 1e3:.2f} ms")

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The lines in the order they are read, in segments of ``views_per_segment``.

pp.plot.plot_kspace(baseline, color_by="order", plane="xy", show_trajectory=False)

# %%
# Cine
# ----
#
# ``n_phases`` reads the same segment of lines at several points after the
# trigger, so one breath-hold resolves the cardiac cycle. The segment length
# is what trades temporal resolution against the number of heartbeats the
# scan takes.

alternative = bssfp2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_phases=8, views_per_segment=12, tr=None, n_dummy=0
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("1 phase", baseline), ("8 phases", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="order", plane="xy", show_trajectory=False)

# %%
