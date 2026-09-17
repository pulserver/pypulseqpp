"""
================
3D balanced SSFP
================

The balanced gradient structure of the two-dimensional sequence over a
partition-encoded slab, with each train opened by a half flip.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/bssfp3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import bssfp3D_sequence

seq = bssfp3D_sequence(
    n_x=160,
    n_y=160,
    n_z=32,
    tr=None,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(tr=40)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="yz", show_trajectory=False)
