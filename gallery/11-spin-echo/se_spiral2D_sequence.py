"""
===================
2D spiral spin echo
===================

One spiral interleaf is read at the refocused echo of each excitation.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/se_spiral2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import se_spiral2D_sequence

seq = se_spiral2D_sequence(
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
