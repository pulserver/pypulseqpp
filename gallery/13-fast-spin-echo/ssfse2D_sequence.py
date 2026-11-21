"""
==================================
2D single-shot fast spin echo
==================================

One excitation and one CPMG train read a whole slice: every phase-encode
line is one echo of the same train. Partial Fourier along the phase encode
and parallel imaging shorten the train, so that T2 decay along it blurs the
image little, and reading the lines in increasing ky from just below the
centre keeps the effective echo time short. Refocusing angles below 180
degrees lower the RF power of the long train; a target-signal train holds
the echo amplitude steady across it. With an inversion time, a
slice-selective adiabatic inversion precedes every excitation and the train
becomes a single-shot FLAIR. Single-shot FSE is used for T2-weighted and
FLAIR imaging that has to be free of motion between lines.
"""

# sphinx_gallery_start_ignore
import warnings

# blochsim's simulator is compiled with torch.jit.script, which newer torch
# releases flag as deprecated; the warning concerns torch, not this sequence.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
# sphinx_gallery_end_ignore

# %%
# A short train
# -------------
#
# A small matrix keeps the diagram legible: the excitation, then one
# variable-angle refocusing pulse, phase encode, readout and rewinder per
# echo.

import pypulseqpp as pp
from pypulseqpp.sequences import ssfse2D_sequence

diagram = ssfse2D_sequence(n_x=64, n_y=32, n_acs_y=4)
diagram.paper_plot()

# %%
# Refocusing trains
# -----------------
#
# The same prescription with a constant, a TRAPS and a target-signal train.

trains = {
    name: ssfse2D_sequence(flip_modulation=name, n_slices=1)
    for name in ("constant", "traps", "pd_target")
}

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
for name, seq in trains.items():
    flips = np.asarray(seq.get_definition("RefocusingFlipAngles"))
    ax.plot(np.arange(1, flips.size + 1), flips, label=name)
ax.set_xlabel("echo")
ax.set_ylabel("refocusing angle (degrees)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Sampling order
# --------------
#
# Colour encodes the order the lines are read in: the train starts a few
# lines below the centre of k-space and climbs to the edge.

seq = trains["pd_target"]
print(
    f"{int(seq.get_definition('EchoTrainLength')[0])} echoes, "
    f"{seq.get_definition('EchoSpacing')[0] * 1e3:.2f} ms echo spacing, "
    f"effective TE {seq.get_definition('TE')[0] * 1e3:.1f} ms"
)
pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)

# %%
# Single-shot FLAIR
# -----------------
#
# A slab inversion twice the slice thickness, 2.5 s before each excitation,
# nulls fluid. The slices are played even ones first, so one slice's train
# does not saturate the slice played after it.

flair = ssfse2D_sequence(n_slices=4, ti=2.5)
print(f"{flair.duration()[0]:.1f} s for four slices")
flair.paper_plot()
