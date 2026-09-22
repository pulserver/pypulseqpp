"""
========================================
Individually optimized 3D fast spin echo
========================================

Individually parameterized 3D FSE assigns different echo-train lengths and
repetition times to the shots that acquire central and peripheral k-space
[BUO25]_. Train length, TR and the minimum and maximum angles of the
refocusing schedule [BUS08b]_ vary smoothly between these limits, and the
views are assigned by an adaptive radial order. Longer trains and a
different TR at the periphery can reduce scan time, while the contrast at the
centre of k-space is set by the parameters of the central shots.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypulseqpp.plot import SAMPLING

PAGE_WIDTH = 8.6
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})
# sphinx_gallery_end_ignore
from pypulseqpp import sequences

Fse3DApp = sequences.fse3D_sequence.Fse3DApp

P = {
    "n_x": 80,
    "n_y": 40,
    "n_z": 16,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 12,
    "etl_periphery": 24,
    "te": 36e-3,
    "tr": 0.45,
    "tr_periphery": 0.65,
    "n_dummy": 0,
    "ordering": "radial",
    "flip_modulation": "optimized",
    "wave_amplitude": 0.0,
}
app = Fse3DApp(**P)
seq = app.design()

# %%
# Train parameters
# ----------------
#
# Train length and TR follow a cubic smooth-step transition,
# :math:`3u^2 - 2u^3` with :math:`u` from 0 at the first (central) shot to 1
# at the last (peripheral) shot [BUO25]_. The refocusing schedules of representative shots are plotted
# up to each shot's own train length.

# sphinx_gallery_start_ignore
indices = np.unique(np.linspace(0, len(app.trains) - 1, 4, dtype=int))
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2))
for i in indices:
    n = app.lengths[i]
    label = f"shot {i}: ETL {n}, TR {app.times[i] * 1e3:.0f} ms"
    axes[0].plot(np.arange(1, n + 1), app.flips[i, :n], label=label)
axes[0].set(xlabel="Echo index", ylabel="Refocusing flip angle (degrees)")
axes[1].plot(np.arange(len(app.trains)), app.lengths, label="ETL")
axes[1].set(xlabel="Shot index", ylabel="Echo-train length")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.98),
    frameon=False,
)
fig.tight_layout(rect=(0, 0, 1, 0.75))
# sphinx_gallery_end_ignore

# %%
# Adaptive radial ordering
# ------------------------
#
# The ordering ranks ``(shot, echo)`` slots jointly by distance from the
# effective-TE echo and by position in the central-to-peripheral transition
# [BUO25]_. Views are ranked by k-space radius; the innermost views fill the
# slots nearest the effective-TE echo of the central shots, and within each group of one view per shot the views
# are assigned to shots in order of angle. Colour gives the train length and
# TR of the shot that acquired each view.

# sphinx_gallery_start_ignore
labels = seq.evaluate_labels(evolution="adc")
echo = np.asarray(labels["ECO"])
shot = np.cumsum(echo == 0) - 1
ky = np.asarray(labels["LIN"]) - P["n_y"] // 2
kz = np.asarray(labels["PAR"]) - P["n_z"] // 2
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.5), sharey=True)
for ax, val, label in zip(
    axes,
    (np.asarray(app.lengths)[shot], np.asarray(app.times)[shot] * 1e3),
    ("Echo-train length", "TR (ms)"),
    strict=True,
):
    art = ax.scatter(ky, kz, c=val, cmap=SAMPLING, s=15, linewidth=0)
    fig.colorbar(art, ax=ax, label=label, pad=0.02)
    ax.set_xlabel(r"$k_y$ (lines from centre)")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$k_z$ (partitions from centre)")
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# References
# ----------
#
# .. [BUO25] Buonincontri G, et al. ISMRM 2025, abstract 566-05-007.
#
# .. [BUS08b] Busse RF, Brau ACS, Vu A, Michelich CR, Bayram E, Kijowski R,
#    Reeder SB, Rowley HA. Effects of refocusing flip angle modulation and
#    view ordering in 3D fast spin echo. *Magnetic Resonance in Medicine*.
#    2008;60(3):640-649. https://doi.org/10.1002/mrm.21680
