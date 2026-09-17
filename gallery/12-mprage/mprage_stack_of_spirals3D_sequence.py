"""
==========================
3D stack-of-spirals MPRAGE
==========================

Each shot applies one inversion and then reads the interleaves of a single
partition.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/mprage_stack_of_spirals3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import mprage_stack_of_spirals3D_sequence

seq = mprage_stack_of_spirals3D_sequence(
    n=192,
    n_z=16,
    n_shots=16,
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

pp.plot.plot_kspace(seq, color_by="order", show_trajectory=False)
