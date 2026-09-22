"""
=============================
Shuffled echo-resolved 3D FSE
=============================

Shuffled 3D FSE uses the same refocusing-train design as conventional FSE
and differs in two separate choices. The support is a variable-density
Poisson-disc draw, which determines which views are acquired. The ordering
is that of T2 Shuffling [TAM17]_: each train is a group of contiguous views in
raster order, and the echo position of each view within its train is random,
which determines when each view is acquired. Together, the variable-density
support and the random echo positions give each echo time a subset of views
spread over the sampled extent without a regular pattern, the sampling
condition of echo-resolved subspace reconstruction [TAM17]_; no
reconstruction is performed here.
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
    "n_x": 128,
    "n_y": 96,
    "n_z": 48,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 64,
    "te": 100e-3,
    "tr": 1.4,
    "ry": 2,
    "rz": 2,
    "n_acs_y": 8,
    "n_acs_z": 6,
    "n_dummy": 0,
    "ordering": "shuffling",
    "flip_modulation": "optimized",
    "wave_amplitude": 0.0,
}
app = Fse3DApp(**P)
seq = app.design()

# %%
# Variable-density sampling
# -------------------------
#
# A fully sampled calibration region is embedded in a variable-density
# Poisson-disc support, selected with
# ``make_cartesian_plane_sampling(..., sampling="poisson")``. The echo index of
# each view is then assigned by :func:`~pypulseqpp.make_shuffling_order` rather
# than by its distance from the k-space centre.

# sphinx_gallery_start_ignore
labels = seq.evaluate_labels(evolution="adc")
echo = np.asarray(labels["ECO"])
ky = np.asarray(labels["LIN"]) - P["n_y"] // 2
kz = np.asarray(labels["PAR"]) - P["n_z"] // 2
te_ms = (echo + 1) * app.fse.esp * 1e3
fig, axes = plt.subplots(
    1, 2, figsize=(PAGE_WIDTH, 3.0), sharey=True, layout="constrained"
)
axes[0].scatter(ky, kz, s=3, color="0.5", linewidth=0)
axes[0].set_title("Sampled views")
art = axes[1].scatter(ky, kz, c=te_ms, cmap=SAMPLING, s=3, linewidth=0)
fig.colorbar(art, ax=axes, label="Echo time (ms)", pad=0.02, shrink=0.8)
axes[1].set_title("Echo-time distribution")
for ax in axes:
    ax.set_xlabel(r"$k_y$ (lines from centre)")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$k_z$ (partitions from centre)")
# sphinx_gallery_end_ignore

# %%
# Echo-time distribution
# ----------------------
#
# The refocusing train is designed as in
# :doc:`/generated/gallery/13-fast-spin-echo/fse3D_sequence`, whose figures
# show the schedule; shuffling changes the assignment of views to echoes, not
# the refocusing schedule. With a 64-echo train, each echo index occurs
# throughout the sampled extent, so the contrast evolution along the train is
# not confined to a radial k-space band. The figure gives the distance of
# every acquired view from the k-space centre against its echo time: every
# echo time samples views from the centre to the edge of the support.

# sphinx_gallery_start_ignore
radius = np.hypot(ky / (P["n_y"] / 2), kz / (P["n_z"] / 2))
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
ax.scatter(te_ms, radius, s=4, color="0.4", linewidth=0)
ax.set(
    xlabel="Echo time (ms)",
    ylabel="Normalised k-space radius",
    ylim=(0, 1.05),
)
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# References
# ----------
#
# .. [TAM17] Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS,
#    Lustig M. T2 shuffling: sharp, multicontrast, volumetric fast spin-echo
#    imaging. *Magnetic Resonance in Medicine*. 2017;77(1):180-195.
#    https://doi.org/10.1002/mrm.26102
