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
    "n_x": 80,
    "n_y": 48,
    "n_z": 24,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 16,
    "te": 48e-3,
    "tr": 0.5,
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
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6), sharey=True)
axes[0].scatter(ky, kz, s=13, color="0.5", linewidth=0)
axes[0].set_title("Sampled views")
art = axes[1].scatter(ky, kz, c=te_ms, cmap=SAMPLING, s=14, linewidth=0)
fig.colorbar(art, ax=axes[1], label="Echo time (ms)", pad=0.02)
axes[1].set_title("Echo-time distribution")
for ax in axes:
    ax.set_xlabel(r"$k_y$ (lines from centre)")
    ax.set_aspect("equal")
axes[0].set_ylabel(r"$k_z$ (partitions from centre)")
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Echo-time distribution
# ----------------------
#
# The refocusing train is designed as in the conventional FSE example;
# shuffling changes the assignment of views to echoes, not the refocusing
# schedule. Each echo index occurs throughout the sampled extent, so the
# contrast evolution along the train is not confined to a radial k-space
# band. The histogram gives the number of acquired views per echo time.

# sphinx_gallery_start_ignore
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.hist(
    te_ms,
    bins=np.arange(
        te_ms.min() - 0.5 * app.fse.esp * 1e3,
        te_ms.max() + app.fse.esp * 1e3,
        app.fse.esp * 1e3,
    ),
    color="0.5",
)
ax.set(xlabel="Echo time (ms)", ylabel="Acquired views")
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
