"""
==============================================
2D PROPELLER spin echo with echo-planar blades
==============================================

A whole blade is read after one excitation as an echo-planar train, so a
PROPELLER coverage is acquired in as many shots as there are blades.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/se_epi_propeller2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import se_epi_propeller2D_sequence

seq = se_epi_propeller2D_sequence(
    n_x=192,
    blade_width=16,
    n_slices=1,
    te=60e-3,
    tr=None,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# One repetition, with the others drawn underneath in grey.

seq.paper_plot(tr=4)

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)
