"""
================================
3D stack-of-blades gradient echo
================================

PROPELLER blades in the plane and a Cartesian partition encode along z.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre_stack_of_blades3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_stack_of_blades3D_sequence

seq = gre_stack_of_blades3D_sequence(
    n=192,
    n_z=16,
    blade_width=16,
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

pp.plot.plot_kspace(seq, color_by="shot", show_trajectory=False)
