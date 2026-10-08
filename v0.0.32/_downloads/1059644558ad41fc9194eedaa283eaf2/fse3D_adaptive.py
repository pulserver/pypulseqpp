"""
========================================
Individually optimized 3D fast spin echo
========================================

Individually parameterized 3D FSE assigns different echo-train lengths and
repetition times to the shots that acquire central and peripheral k-space
[BUO25]_. From the central to the peripheral shots, TR and the minimum and
maximum angles of the refocusing schedule [BUS08b]_ follow a cubic
smooth-step transition between their two limits; the echo-train length
follows the same transition, rounded to integer values. The views are
assigned by an adaptive radial order. Longer trains and a different TR at the
periphery can reduce scan time, while the contrast at the centre of k-space
is set by the parameters of the central shots.
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

from pypulseqpp.plot import SAMPLING

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
# sphinx_gallery_end_ignore
from pypulseqpp import sequences

P = {
    "n_x": 96,
    "n_y": 64,
    "n_z": 20,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 40,
    "etl_periphery": 72,
    "te": 200e-3,
    "tr": 1.2,
    "tr_periphery": 1.6,
    "refocusing_angle_deg": 120.0,
    "n_dummy": 0,
    "ordering": "radial",
    "flip_modulation": "optimized",
    "wave_amplitude": 0.0,
}
seq = sequences.fse3D_sequence(**P)
n_shots = int(seq.get_definition("NumShots")[0])

# %%
# Train parameters
# ----------------
#
# TR and the schedule's minimum and maximum angles follow a cubic smooth-step
# transition, :math:`3u^2 - 2u^3` with :math:`u` from 0 at the first (central)
# shot to 1 at the last (peripheral) shot [BUO25]_; the echo-train length
# follows the same transition rounded to integers. Every schedule passes
# through the prescribed 120 degrees at the effective-TE echo. The schedules
# of representative shots are plotted up to each shot's own train length:
# the central shots reach lower minimum angles, and the peripheral shots have
# shallower minima and longer trains.
#
# The schedules are read from the pulses the sequence plays. The flip angle of
# each refocusing pulse is that of its pulse definition scaled by the relative
# amplitude of the instance, which is zero past the shot's own train length.
# The TR of a shot is the interval from its excitation to the next; the last
# shot's is recorded as ``TRPeriphery``.

instances, rf_times = seq.rf_instances(), seq.rf_times(compat=False)
use = np.asarray(rf_times.use)
flip = np.array([rf.flip_deg for rf in instances.definitions])[instances.definition]
angles = (flip * instances.amplitude)[use == "refocusing"].reshape(n_shots, -1)
lengths = np.count_nonzero(angles, axis=1)
excitations = np.asarray(rf_times.t)[use == "excitation"]
tr = np.append(np.diff(excitations), seq.get_definition("TRPeriphery")[0])

# sphinx_gallery_start_ignore
indices = np.unique(np.linspace(0, n_shots - 1, 4, dtype=int))
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2), layout="constrained")
for i in indices:
    n = lengths[i]
    label = f"shot {i}: ETL {n}, TR {tr[i] * 1e3:.0f} ms"
    axes[0].plot(np.arange(1, n + 1), angles[i, :n], label=label)
axes[0].set(xlabel="Echo index", ylabel="Refocusing flip angle (degrees)")
axes[1].plot(np.arange(n_shots), lengths, label="ETL")
axes[1].set(xlabel="Shot index", ylabel="Echo-train length")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncols=2, loc="outside upper center")
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
echo = np.zeros(len(np.atleast_1d(labels["LIN"])), dtype=int)
place = -1
at = 0
for n in range(1, len(seq.block_events) + 1):
    block = seq.get_block(n)
    rf = getattr(block, "rf", None)
    if rf is not None and rf.use == "excitation":
        place = -1
    if getattr(block, "adc", None) is not None:
        place += 1
        echo[at] = place
        at += 1
shot = np.cumsum(echo == 0) - 1
ky = np.asarray(labels["LIN"]) - P["n_y"] // 2
kz = np.asarray(labels["PAR"]) - P["n_z"] // 2
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.4), sharey=True)
for ax, val, label in zip(
    axes,
    (lengths[shot], tr[shot] * 1e3),
    ("Echo-train length", "TR (ms)"),
    strict=True,
):
    art = ax.scatter(ky, kz, c=val, cmap=SAMPLING, s=6, linewidth=0)
    fig.colorbar(art, ax=ax, label=label, pad=0.02, shrink=0.8)
    ax.set_xlabel(r"$k_y$ (lines from centre)")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$k_z$ (partitions from centre)")
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# References
# ----------
#
# .. [BUO25] Buonincontri G, et al. *Proceedings of the International Society
#    for Magnetic Resonance in Medicine*. 2025; abstract 566-05-007.
#
# .. [BUS08b] Busse RF, Brau ACS, Vu A, Michelich CR, Bayram E, Kijowski R,
#    Reeder SB, Rowley HA. Effects of refocusing flip angle modulation and
#    view ordering in 3D fast spin echo. *Magnetic Resonance in Medicine*.
#    2008;60(3):640-649. https://doi.org/10.1002/mrm.21680
