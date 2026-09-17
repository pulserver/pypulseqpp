"""
=================
3D zero echo time
=================

The readout gradient is already at amplitude when the hard pulse is
transmitted, so acquisition begins without a ramp and the trajectory starts at
the centre of k-space.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/zte3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import zte3D_sequence

seq = zte3D_sequence(
    n_x=128,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(time_range=(0.0, 0.004))

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="shot", show_trajectory=False, block_range=(1, 2000))
