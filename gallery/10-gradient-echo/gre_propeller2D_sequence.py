"""
==========================
2D PROPELLER gradient echo
==========================

Each shot reads a rectangular blade of Cartesian lines and the blades are
rotated to cover k-space. Every blade samples the centre, so the shots can be
registered against each other before reconstruction.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre_propeller2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_propeller2D_sequence

seq = gre_propeller2D_sequence(
    n=192,
    blade_width=16,
    n_slices=1,
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

pp.plot.plot_kspace(seq, color_by="shot", plane="xy", show_trajectory=False)
