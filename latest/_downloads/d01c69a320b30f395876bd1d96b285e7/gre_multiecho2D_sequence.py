"""
=====================================
2D Cartesian multi-echo gradient echo
=====================================

Each excitation is followed by several readout lobes, so one phase-encode line
is sampled at several echo times and the decay across them measures T2*.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre_multiecho2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_multiecho2D_sequence

seq = gre_multiecho2D_sequence(
    n_x=192,
    n_y=192,
    n_slices=1,
    n_echoes=4,
    te=None,
    tr=None,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(tr=48)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)
