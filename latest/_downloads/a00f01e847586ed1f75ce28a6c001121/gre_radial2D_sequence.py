"""
=======================
2D radial gradient echo
=======================

Every repetition reads a full spoke through the centre of k-space, so the low
spatial frequencies are sampled once per repetition rather than once per scan.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre_radial2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_radial2D_sequence

seq = gre_radial2D_sequence(
    n=192,
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

pp.plot.plot_kspace(seq, color_by="shot", plane="xy")
