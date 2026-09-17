"""
======================
3D Cartesian spin echo
======================

One ``(line, partition)`` view is read per excitation, at a refocused echo.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/se3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import se3D_sequence

seq = se3D_sequence(
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
