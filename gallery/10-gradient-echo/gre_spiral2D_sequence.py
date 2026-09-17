"""
=======================
2D spiral gradient echo
=======================

One spiral interleaf is read per repetition, designed from the prescription
rather than from a fixed shape, and rotated to each of ``n_shots`` angles.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/gre_spiral2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import gre_spiral2D_sequence

seq = gre_spiral2D_sequence(
    n=192,
    n_shots=16,
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

seq.paper_plot(tr=8)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="shot", plane="xy")
