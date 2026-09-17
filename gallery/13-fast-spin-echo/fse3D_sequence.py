"""
=================
3D fast spin echo
=================

One excitation is followed by a CPMG train of refocusing pulses, and one
``(line, partition)`` view is read at each echo. The amplitude left at an echo
weights whichever view that echo reads.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/fse3D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import fse3D_sequence

seq = fse3D_sequence(
    n_x=192,
    n_y=160,
    n_z=32,
    etl=16,
    te=None,
    tr=0.6,
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

pp.plot.plot_kspace(seq, color_by="order", plane="yz", show_trajectory=False)
