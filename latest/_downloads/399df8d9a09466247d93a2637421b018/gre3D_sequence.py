"""
==========================
3D Cartesian gradient echo
==========================

A slab-selective excitation replaces the slice-selective one and the second
phase encode samples the partition axis, so one repetition reads one
``(line, partition)`` view.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre3D_sequence

seq = gre3D_sequence(
    n_x=160,
    n_y=160,
    n_z=32,
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

seq.paper_plot(tr=40)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="yz", show_trajectory=False)
