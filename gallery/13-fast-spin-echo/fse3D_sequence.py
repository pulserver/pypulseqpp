"""
==================
3D fast spin echo
==================

One excitation followed by a CPMG train of refocusing pulses, with one
``(line, partition)`` view acquired per echo. Signal amplitude at echo
:math:`m` weights the corresponding k-space view.
Echo ordering therefore determines the modulation transfer function and
point-spread function.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def _views(seq, n_y, n_z):
    """Every acquisition as (line, partition, echo index, shot index)."""
    labels = seq.evaluate_labels(evolution="adc")
    echo = np.asarray(labels["ECO"])
    return (
        np.asarray(labels["LIN"]) - n_y // 2,
        np.asarray(labels["PAR"]) - n_z // 2,
        echo,
        np.cumsum(echo == 0) - 1,
    )


def order_figure(seq, n_y, n_z):
    """The echo index and the shot index of every view, side by side."""
    line, partition, echo, shot = _views(seq, n_y, n_z)
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.6), sharey=True)
    for axis, value, label in zip(
        axes, (echo, shot), ("Echo index", "Shot index"), strict=True
    ):
        drawn = axis.scatter(line, partition, c=value, cmap="turbo", s=9, linewidth=0)
        figure.colorbar(drawn, ax=axis, label=label, pad=0.02)
        axis.set_xlabel("$k_y$ (lines from centre)")
        axis.grid(alpha=0.2, lw=0.4)
    axes[0].set_ylabel("$k_z$ (partitions from centre)")
    figure.tight_layout()
    return figure


def envelope_figure(envelopes, esp_ms, weighting):
    """The echo amplitudes, and the weight they give each line."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.0))
    for name, amplitude in envelopes.items():
        left.plot(np.arange(len(amplitude)) * esp_ms, amplitude, lw=1.3, label=name)
    left.set_xlabel("time in the train (ms)")
    left.set_ylabel("echo amplitude")
    left.legend(frameon=False, fontsize=9)
    for name, (offsets, weight) in weighting.items():
        right.plot(offsets, weight, lw=1.3, label=name)
    right.set_xlabel("$k_y$ (lines from centre)")
    right.set_ylabel("weight")
    right.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    return figure


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


# sphinx_gallery_end_ignore

# %%
# Baseline
# --------
#
# A short train limits T2 weighting across the train. Centre-out view ordering
# assigns the earliest echoes to the centre of k-space.


from pypulseqpp.sequences import fse3D_sequence

PRESCRIPTION = {
    "n_x": 128,
    "n_y": 96,
    "n_z": 16,
    "fov_x": 0.2,
    "fov_y": 0.2,
    "fov_z": 0.1,
}

baseline = fse3D_sequence(**PRESCRIPTION, etl=16, te=None, tr=None, n_dummy=0)
print(
    f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.1f} s, "
    f"echo spacing {baseline.get_definition('EchoSpacing')[0] * 1e3:.2f} ms, "
    f"TE {baseline.get_definition('TE')[0] * 1e3:.1f} ms"
)

# %%
# Sequence diagram
# ----------------
#
# The automatically detected repetition contains the excitation, the CPMG
# train with a crusher pair around every refocusing pulse, and the phase and
# partition encodes before and after each readout. The solid trace is a
# representative train; the shaded traces retain the range of encodes.

baseline.paper_plot()

# %%
# Echo order and shot order
# -------------------------
#
# Echo index identifies the position of a view within one CPMG train and thus
# its T2 weighting. Shot index identifies the excitation and repetition that
# acquired the view. Centre-out ordering assigns the least attenuated echoes
# to central k-space.

# sphinx_gallery_start_ignore
order_figure(baseline, PRESCRIPTION["n_y"], PRESCRIPTION["n_z"])
# sphinx_gallery_end_ignore

# %%
# Train length
# ------------
#
# A longer train requires fewer excitations but samples later points of the T2
# decay, increasing attenuation toward the edge of k-space.

long_train = fse3D_sequence(**PRESCRIPTION, etl=48, te=None, tr=None, n_dummy=0)

# sphinx_gallery_start_ignore
print(f"{'':10} {'ETL':>5} {'shots':>7} {'scan (s)':>10} {'train (ms)':>12}")
for name, seq in (("ETL 16", baseline), ("ETL 48", long_train)):
    _, _, echo, shot = _views(seq, PRESCRIPTION["n_y"], PRESCRIPTION["n_z"])
    print(
        f"{name:10} {int(echo.max()) + 1:5d} {int(shot.max()) + 1:7d} "
        f"{seq.duration()[0]:10.1f} "
        f"{(int(echo.max()) + 1) * seq.get_definition('EchoSpacing')[0] * 1e3:12.1f}"
    )
# sphinx_gallery_end_ignore

# %%
# T2 weighting
# ------------
#
# The simulated signal envelope uses the stored refocusing-angle schedule and
# echo spacing. Each line's weight is
# the envelope at the echo index that read it, averaged over the partitions it
# was read at.

import torchsim

#: Relaxation times (ms) of the tissue the envelopes are simulated for.
T1_MS, T2_MS = 1200.0, 60.0

envelopes, weighting = {}, {}
for name, seq in (("ETL 16", baseline), ("ETL 48", long_train)):
    angles = np.asarray(seq.get_definition("RefocusingFlipAngles"))
    esp_ms = 1e3 * seq.get_definition("EchoSpacing")[0]
    amplitude = np.abs(
        np.asarray(torchsim.fse_sim(flip=angles, ESP=esp_ms, T1=T1_MS, T2=T2_MS))
    )
    line, _, echo, _ = _views(seq, PRESCRIPTION["n_y"], PRESCRIPTION["n_z"])
    offsets = np.unique(line)
    weighting[name] = (
        offsets,
        np.array([amplitude[echo[line == offset]].mean() for offset in offsets]),
    )
    envelopes[name] = amplitude

# sphinx_gallery_start_ignore
envelope_figure(envelopes, 1e3 * baseline.get_definition("EchoSpacing")[0], weighting)
# sphinx_gallery_end_ignore

# %%
# Central k-space receives nearly the same weight for both train lengths because
# it is acquired first. The longer train attenuates outer k-space more strongly,
# increasing the point-spread width along the phase-encode axes.
#
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model below is a demonstration, not a
# scanner's.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
grad_ok, grad = safety.check_max_grad(baseline)
slew_ok, slew = safety.check_max_slew(baseline)
cont_ok, cont = safety.check_grad_continuity(baseline)
pns_ok, pns = safety.check_pns(baseline, model)

# sphinx_gallery_start_ignore
safety_table(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / baseline.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.per_axis.value / baseline.system.gamma:.0f} T/m/s",
        ),
        (
            "gradient continuity",
            cont_ok,
            f"{len(cont.discontinuities)} discontinuities",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
    ]
)
# sphinx_gallery_end_ignore
