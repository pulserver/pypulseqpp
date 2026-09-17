"""
=========================
Excitation modules
=========================

The excitation family produces one RF pulse and the gradients that go with it.
Every member reports the events it built and, through
:meth:`~pypulseqpp.sequences.RfModule.sim_rf`, the Bloch response of the pulse
it holds.
"""

# %%
# Slice-selective, non-selective and spectrally selective
# -------------------------------------------------------
#
# The slice-selective module designs an SLR pulse with its selection gradient
# and rephaser, and reports the ``selection_amplitude`` a slice offset is
# converted against. The non-selective module plays a pulse alone. The
# spectral-spatial module selects a band in frequency as well as in space.

import numpy as np

import pypulseqpp as pp
import pypulseqpp.sequences as design

system = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=150.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=30e-6,
)

selective = design.SpatialSelectiveExcitation(
    system, flip_angle_deg=90.0, thickness_m=5e-3, duration_s=3e-3
)
hard = design.NonSelectiveExcitation(system, flip_angle_deg=90.0, duration_s=500e-6)
refocusing = design.SpatialSelectiveRefocusing(
    system, flip_angle_deg=180.0, thickness_m=5e-3, duration_s=3e-3
)

for name, module in (
    ("SpatialSelectiveExcitation", selective),
    ("NonSelectiveExcitation", hard),
    ("SpatialSelectiveRefocusing", refocusing),
):
    print(f"{name:28} {len(module.blocks)} blocks, {module.duration * 1e3:5.2f} ms")

# %%
# The simulated profile
# ---------------------
#
# ``sim_rf`` simulates the pulse across off-resonance. Under a selection
# gradient that is the slice profile: a spin at position ``z`` is off-resonance
# by ``selection_amplitude * z``.

magnetisation, frequency = selective.sim_rf()[1:3]
position = frequency / selective.selection_amplitude

import matplotlib.pyplot as plt

figure, axis = plt.subplots(figsize=(6.4, 3.2))
axis.plot(1e3 * position, np.abs(magnetisation))
axis.axvspan(-2.5, 2.5, color="0.9", lw=0, zorder=0)
axis.set_xlim(-12, 12)
axis.set_xlabel("position (mm)")
axis.set_ylabel(r"$|M_{xy}|$")
axis.set_title("nominal slice in grey")
figure.tight_layout()

# %%
# Simultaneous multi-slice
# ------------------------
#
# ``SmsExcitation`` modulates the selective pulse so that several slices are
# excited together, and ``MultibandExcitation`` places saturation bands at a
# stated frequency offset. A single slice is moved off centre instead by a
# frequency offset of ``selection_amplitude * position`` on the pulse.

sms = design.SmsExcitation(
    system,
    flip_angle_deg=60.0,
    thickness_m=5e-3,
    duration_s=4e-3,
    n_bands=3,
    slice_gap_m=30e-3,
)
print(f"SmsExcitation {len(sms.blocks)} blocks, {sms.duration * 1e3:.2f} ms")

seq = pp.Sequence(system=system)
for block in sms.blocks:
    seq.add_block(*block)
seq.paper_plot(tr=1)
