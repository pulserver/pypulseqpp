"""
==========================
2D Cartesian gradient echo
==========================

One phase-encode line is read per repetition and the transverse magnetisation is
spoiled between them, so the signal is a steady state of the flip angle, the
repetition time and T1.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre2D_sequence

seq = gre2D_sequence(
    n_x=192,
    n_y=192,
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

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)
