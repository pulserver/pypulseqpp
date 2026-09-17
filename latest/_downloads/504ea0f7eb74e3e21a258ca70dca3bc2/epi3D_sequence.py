"""
======================
3D echo-planar imaging
======================

A slab-selective excitation and a partition encode over the echo-planar train,
with the skipped-CAIPI lattice available when both encoded axes are
undersampled.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/epi3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import epi3D_sequence

seq = epi3D_sequence(
    n_x=128,
    n_y=128,
    n_z=24,
    tr=None,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(tr=1)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="yz", show_trajectory=False)
