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

diagram = bssfp2D_sequence(
    n_x=48,
    n_y=12,
    n_slices=1,
    n_phases=1,
    readout_bandwidth_hz=25_000,
    tr=None,
    n_dummy=1,
)
baseline = bssfp2D_sequence(
    n_x=192, n_y=192, n_slices=1, n_phases=1, tr=None, n_dummy=0
)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(f"TR {baseline.get_definition('TR')[0] * 1e3:.2f} ms")

# %%
# Sequence diagram
# ----------------

diagram.paper_plot()

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
# Prospective gating acquires each segment once per requested cardiac phase
# after a trigger. Retrospective gating cycles the segment throughout one
# heartbeat and records the cycle index in ``PHS`` for later cardiac binning.
# Segment length sets the temporal footprint of each cardiac phase.

prospective = bssfp2D_sequence(
    n_x=96,
    n_y=48,
    n_slices=1,
    n_phases=6,
    views_per_segment=8,
    gating="prospective",
    tr=None,
    n_dummy=0,
)
retrospective = bssfp2D_sequence(
    n_x=96,
    n_y=48,
    n_slices=1,
    n_phases=6,
    views_per_segment=8,
    gating="retrospective",
    tr=None,
    n_dummy=0,
)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (
    ("ungated", baseline),
    ("prospective", prospective),
    ("retrospective", retrospective),
):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(prospective, color_by="order", plane="xy", show_trajectory=False)

# %%
