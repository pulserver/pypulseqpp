"""
==============================
Conventional 3D fast spin echo
==============================

A slab-selective excitation is followed by a CPMG fast-spin-echo refocusing
train, with one Cartesian ``(line, partition)`` view acquired at each echo.
Variable refocusing angles control stimulated-echo pathways and T2-dependent
signal evolution. Radial view ordering assigns this evolution to k-space and
therefore determines the modulation transfer function and image blurring. 3D
FSE is used for T2- and proton-density-weighted structural imaging.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypulseqpp.plot import SAMPLING

PAGE_WIDTH = 8.6
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


def views(seq, ny, nz):
    """Return acquired coordinates, echo indices, and shot indices."""
    labels = seq.evaluate_labels(evolution="adc")
    echo = np.asarray(labels["ECO"])
    return (
        np.asarray(labels["LIN"]) - ny // 2,
        np.asarray(labels["PAR"]) - nz // 2,
        echo,
        np.cumsum(echo == 0) - 1,
    )


def order_figure(seq, ny, nz):
    """Plot echo and shot indices on the acquired ky-kz grid."""
    ky, kz, echo, shot = views(seq, ny, nz)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.5), sharey=True)
    for ax, value, label in zip(
        axes, (echo, shot), ("Echo index", "Shot index"), strict=True
    ):
        art = ax.scatter(ky, kz, c=value, cmap=SAMPLING, s=12, linewidth=0)
        fig.colorbar(art, ax=ax, label=label, pad=0.02)
        ax.set_xlabel(r"$k_y$ (lines from centre)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$k_z$ (partitions from centre)")
    fig.tight_layout()
    return fig


# sphinx_gallery_end_ignore

from pypulseqpp import sequences

Fse3DApp = sequences.fse3D_sequence.Fse3DApp

DIAGRAM = {
    "n_x": 64,
    "n_y": 24,
    "n_z": 8,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 6,
    "te": None,
    "tr": None,
    "n_dummy": 0,
    "ordering": "radial",
    "flip_modulation": "optimized",
    "wave_amplitude": 0.0,
}
diagram_app = Fse3DApp(**DIAGRAM)
diagram = diagram_app.design()

ANALYSIS = {
    "n_x": 96,
    "n_y": 64,
    "n_z": 20,
    "fov_x": 0.20,
    "fov_y": 0.20,
    "fov_z": 0.12,
    "etl": 48,
    "te": 120e-3,
    "tr": 1.2,
    "n_dummy": 0,
    "ordering": "radial",
    "flip_modulation": "optimized",
    "wave_amplitude": 0.0,
}
app = Fse3DApp(**ANALYSIS)
seq = app.design()
print(
    f"{len(app.trains)} shots; {app.fse.esp * 1e3:.2f} ms echo spacing; "
    f"{seq.get_definition('TE')[0] * 1e3:.1f} ms effective TE"
)

# %%
# Sequence diagram
# ----------------
#
# Each echo comprises a variable-angle refocusing pulse, phase and partition
# prephasing, one frequency-encoded ADC event, and rephasing. The effective TE
# is the echo assigned to k-space centre.
diagram.paper_plot()

# %%
# Refocusing schedule and echo signal
# -----------------------------------
#
# The optimized schedule is obtained with ``torchsim``'s configuration-state
# FSE simulator. The objective balances signal at the effective TE, peripheral
# k-space signal, and RF power for the tissue models defined by the sequence.
# The same simulator evaluates the resulting T2-dependent echo envelope.
import torchsim

angles = np.asarray(app.flips[0, : app.lengths[0]])
signal = np.abs(
    np.asarray(torchsim.fse_sim(flip=angles, ESP=app.fse.esp * 1e3, T1=1200.0, T2=60.0))
)
# sphinx_gallery_start_ignore
time_ms = np.arange(1, len(angles) + 1) * app.fse.esp * 1e3
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
axes[0].plot(np.arange(1, len(angles) + 1), angles)
axes[0].set(xlabel="Echo index", ylabel="Refocusing flip angle (degrees)")
axes[1].plot(time_ms, signal)
axes[1].set(xlabel="Echo time (ms)", ylabel="Relative echo amplitude")
fig.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Echo and shot order
# -------------------
#
# Radial ordering assigns views near k-space centre to the effective-TE echo
# and progressively larger radii to echoes farther from it. Echo index records
# position within a train; shot index identifies views acquired after the same
# excitation.

# sphinx_gallery_start_ignore
order_figure(seq, ANALYSIS["n_y"], ANALYSIS["n_z"])
# sphinx_gallery_end_ignore

# %%
# K-space weighting
# -----------------
#
# The echo envelope weights each acquired view according to its echo index.
# Radial assignment converts temporal signal evolution into a predominantly
# radial modulation transfer function; its Fourier transform contributes to
# image blurring along both phase-encode axes.

# sphinx_gallery_start_ignore
ky, kz, echo, _ = views(seq, ANALYSIS["n_y"], ANALYSIS["n_z"])
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.8))
art = ax.scatter(ky, kz, c=signal[echo], cmap=SAMPLING, s=16, linewidth=0)
fig.colorbar(art, ax=ax, label="Relative echo amplitude")
ax.set(xlabel=r"$k_y$ (lines from centre)", ylabel=r"$k_z$ (partitions from centre)")
fig.tight_layout()
# sphinx_gallery_end_ignore
