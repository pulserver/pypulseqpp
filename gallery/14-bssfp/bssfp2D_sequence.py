"""
================
2D balanced SSFP
================

Every gradient axis returns to zero moment within each repetition and the RF
phase alternates, so the steady state depends on the off-resonance accumulated
over one repetition time.
"""

# %%
# The sequence is designed by one call. Every parameter of the prescription is
# documented on its :doc:`API page </generated/sequences/bssfp2D_sequence>`.

import pypulseqpp as pp
from pypulseqpp.sequences import bssfp2D_sequence

seq = bssfp2D_sequence(
    n_x=192,
    n_y=192,
    n_slices=1,
    n_phases=1,
    tr=None,
    n_dummy=0,
)
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")

# %%
# Sequence diagram
# ----------------
#
# Structural repetition detection reports the whole train as this sequence's
# repeating unit, so the window drawn here is one repetition time from the
# middle of the scan rather than a repetition index.

repetition_time = seq.get_definition("TR")[0]
seq.paper_plot(time_range=(96 * repetition_time, 97 * repetition_time))

# %%
# Acquisition order
# -----------------

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)
