"""
======================
2D echo-planar imaging
======================

One excitation is followed by a train of readout lobes of alternating polarity
with phase-encode blips between them, so the whole phase-encode axis is covered
in one or a few shots.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/epi2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import epi2D_sequence

seq = epi2D_sequence(
    n_x=128,
    n_y=128,
    n_slices=1,
    n_shots=2,
    tr=None,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(tr=1)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)
