"""
==============================
Conventional 3D fast spin echo
==============================

A slab-selective excitation is followed by a CPMG fast-spin-echo refocusing
train, with one Cartesian ``(line, partition)`` view acquired at each echo.
Refocusing angles below 180 degrees add stimulated-echo pathways to the echo
signal [HEN88]_, and variable refocusing angles modulate the T2-dependent
signal evolution along the train [BUS08a]_. Radial view ordering maps this
evolution onto the ``(k_y, k_z)`` plane and therefore determines the
modulation transfer function and image blurring [BUS08a]_. 3D FSE is used for
T2- and proton-density-weighted structural imaging.
"""

# sphinx_gallery_start_ignore
import warnings

# torchsim's simulator is compiled with torch.jit.script, which newer torch
# releases flag as deprecated; the warning concerns torch, not this sequence.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")

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
# The refocusing schedule has the form of [BUS08a]_: the angle decreases from
# a maximum to a minimum over the first five echoes, increases to the
# prescribed angle at the effective-TE echo and returns to the maximum at the
# end of the train. The minimum and maximum angles, bounded by the prescribed
# angle, are optimized with the extended phase graph (EPG) FSE simulator of ``torchsim`` [HEN88]_ [WEI15]_.
# The cost combines an echo-to-echo signal-variation measure of blurring, the
# contrast between two of the sequence's design tissues at the effective-TE
# echo, and a penalty on RF power above that of the initial schedule. The same
# simulator evaluates the resulting T2-dependent echo envelope.
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
# Radial ordering [BUS08a]_ assigns views near k-space centre to the
# effective-TE echo and progressively larger radii to echoes farther from it. Echo index records
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

# %%
# References
# ----------
#
# .. [HEN88] Hennig J. Multiecho imaging sequences with low refocusing flip
#    angles. *Journal of Magnetic Resonance*. 1988;78(3):397-407.
#    https://doi.org/10.1016/0022-2364(88)90128-X
#
# .. [WEI15] Weigel M. Extended phase graphs: dephasing, RF pulses, and
#    echoes - pure and simple. *Journal of Magnetic Resonance Imaging*.
#    2015;41(2):266-295. https://doi.org/10.1002/jmri.24619
#
# .. [BUS08a] Busse RF, Brau ACS, Vu A, Michelich CR, Bayram E, Kijowski R,
#    Reeder SB, Rowley HA. Effects of refocusing flip angle modulation and
#    view ordering in 3D fast spin echo. *Magnetic Resonance in Medicine*.
#    2008;60(3):640-649. https://doi.org/10.1002/mrm.21680
