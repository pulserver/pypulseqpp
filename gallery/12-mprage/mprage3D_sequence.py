"""
===================
3D Cartesian MPRAGE
===================

Each shot applies one inversion and then reads every sampled line of a single
partition as a spoiled gradient-echo train, so the partition encode is constant
within a shot.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/mprage3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import mprage3D_sequence

seq = mprage3D_sequence(
    n_x=192,
    n_y=128,
    n_z=24,
    ti=0.9,
    tr=2.3,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(time_range=(0.0, 1.0))

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="yz", show_trajectory=False)
