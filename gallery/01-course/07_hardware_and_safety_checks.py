r"""
=============================
7. Hardware and safety checks
=============================

A sequence that passes ``check_timing`` is one the scanner can play, block by
block. Whether it should is a question about the whole scan: does any gradient
exceed the amplitude or slew-rate limit once it is rotated, does the switching
stimulate peripheral nerves, does the gradient waveform drive the coil at one
of its mechanical resonances, how much RF power does the subject absorb.
pypulseqpp computes each of these from the sequence, and this lesson runs them
on your MPRAGE.

A passing check does not establish that a sequence is safe to run on a
scanner or on a subject. The PNS, mechanical-resonance and SAR models used
here are synthetic demonstrations. Scanner-specific checks and hardware
monitoring are separate.

**Learning objectives**

- Check gradient amplitude, slew rate and continuity on the played
  (rotated) waveforms.
- Compute the peripheral nerve stimulation response, and see where it peaks.
- Compare the gradient spectrum with the coil's forbidden bands.
- Estimate SAR with a set of virtual observation points.

Previous: :doc:`06_radial_mprage`. Next: :doc:`08_sequence_functions`, where
your MPRAGE becomes a function of its protocol.
"""

# sphinx_gallery_start_ignore
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypulseqpp.plot._style import MUTED, SERIES

PAGE_WIDTH = 7.8  # inches, the width of the documentation column
warnings.filterwarnings("ignore", message="Specified RF delay")


def summary(rows):
    """Print a check, its verdict and the reading it turned on."""
    for name, ok, reading in rows:
        print(f"{name:30} {'pass' if ok else 'FAIL':6} {reading}")


# sphinx_gallery_end_ignore

# %%
# Your MPRAGE
# -----------
#
# pypulseqpp ships a complete version of the MPRAGE you built in lesson 6,
# :doc:`mprage_stack_of_stars3D </generated/gallery/12-mprage/mprage_stack_of_stars3D_sequence>`.
# You call it with your scanner and your protocol: 220 mm and 128 in-plane,
# 16 partitions over 80 mm, one inversion every 2.5 s. ``ry=3`` plays every
# third spoke of the 202 a fully sampled plane needs, so 68 per inversion, as
# in lesson 6. How a shipped sequence takes its protocol is the subject of the
# next lesson.
import pypulseqpp as pp
from pypulseqpp import safety, sequences

system = pp.Opts(
    max_grad=32.0,
    grad_unit="mT/m",
    max_slew=130.0,
    slew_unit="T/m/s",
    rf_dead_time=100e-6,
    rf_ringdown_time=20e-6,
    adc_dead_time=10e-6,
)
protocol = {
    "fov": 220e-3,
    "n": 128,
    "fov_z": 80e-3,
    "n_z": 16,
    "ry": 3,
    "tr": 2.5,
    "flip_angle_deg": 8.0,
    "n_dummy": 0,
}
seq = sequences.mprage_stack_of_stars3D_sequence(system, **protocol)
ok, errors = seq.check_timing()
print(f"timing ok: {ok}, scan time {seq.duration()[0]:.1f} s")

# %%
# Gradient amplitude, slew rate, continuity
# -----------------------------------------
#
# Every spoke is the readout rotated in the plane, so each axis plays a
# different share of it. The checks look at the gradients *as played*, after
# each block's rotation, and compare them with ``system``. They report the
# largest value on any one axis, which is what the hardware limits apply to,
# and the largest vector value, which is not the norm of the per-axis peaks
# because the axes peak at different times. The continuity check looks for a
# gradient that jumps between two blocks, which the scanner cannot play.
grad_ok, grad = safety.check_max_grad(seq)
slew_ok, slew = safety.check_max_slew(seq)
cont_ok, cont = safety.check_grad_continuity(seq)

mT_per_m = 1e3 / system.gamma
print(
    f"amplitude: largest {grad.per_axis.value * mT_per_m:.1f} mT/m on "
    f"{grad.per_axis.axis}, vector {grad.vector.value * mT_per_m:.1f} mT/m, "
    f"limit {grad.limit * mT_per_m:.0f} mT/m"
)
print(
    f"slew rate: largest {slew.per_axis.value / system.gamma:.0f} T/m/s, "
    f"limit {slew.limit / system.gamma:.0f} T/m/s"
)
print(f"continuity: {len(cont.discontinuities)} jumps")

# %%
# Peripheral nerve stimulation
# ----------------------------
#
# A switching gradient induces an electric field in the body, which can
# stimulate peripheral nerves. The nerve model turns the slew rate on each
# axis into a response, as a fraction of the stimulation threshold; the axes
# are combined as a root sum of squares, and the check passes while the
# combined response stays below 1.
#
# The model here is a *chronaxie* model with three textbook coefficients. A
# scanner supplies its own model, which you read from its file.
model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
pns_ok, pns = safety.check_pns(seq, model, trace=True)
print(
    f"peak {pns.peak.value:.2f} of threshold at {pns.peak.time:.3f} s, "
    f"in block {pns.peak.block}"
)

# %%
# With ``trace=True`` the check also returns the response over time, so you
# can see where the peak comes from. The figure shows the spokes around it.

# sphinx_gallery_start_ignore
window = (pns.time > pns.peak.time - 10e-3) & (pns.time < pns.peak.time + 10e-3)
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.8, 3.0), layout="constrained")
for i, entry in enumerate(pns.axes):
    axis.plot(
        1e3 * pns.time[window],
        entry.response[window],
        lw=0.9,
        color=SERIES[i],
        label=f"$G_{entry.axis}$",
    )
axis.plot(
    1e3 * pns.time[window],
    pns.response[window],
    lw=1.6,
    color=MUTED,
    label="combined",
)
axis.axhline(1.0, color=SERIES[3], ls="--", lw=1.0, label="threshold")
axis.set_xlabel("time (ms)")
axis.set_ylabel("fraction of threshold")
axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()
# sphinx_gallery_end_ignore

# %%
# The response peaks on the ramps of the merged readout and spoiler, where
# the slew rate is highest. Your MPRAGE stays below the threshold.
#
# Mechanical resonance
# --------------------
#
# A gradient coil is a mechanical structure, and it has resonances: frequency
# bands where a gradient waveform makes it vibrate strongly. Scanners forbid
# those bands, each with a tolerance on the gradient amplitude it may carry.
#
# In this example the coil has two, 550 to 650 Hz at 6 mT/m and 1100 to 1300 Hz at
# 4 mT/m. The check computes the spectrum of the gradients in windows of 50 ms
# along the scan, and counts the windows where a band exceeds its tolerance:
bands = [
    safety.ForbiddenBand(axis=None, f_min=550.0, f_max=650.0, tolerance=6.0),
    safety.ForbiddenBand(axis=None, f_min=1100.0, f_max=1300.0, tolerance=4.0),
]
mech_ok, mech = safety.check_mech_resonance(seq, bands, window_width=50e-3)
for band in mech.bands:
    print(
        f"{band.peak:.1f} mT/m at {band.frequency:.0f} Hz, "
        f"{band.violations} windows over the tolerance"
    )

# %%
# The spectrum of the window where the first band peaks:

# sphinx_gallery_start_ignore
spectrum = safety.mech_resonance_spectrum(
    seq, window=mech.bands[0].window, window_width=50e-3
)
figure, axis = plt.subplots(figsize=(PAGE_WIDTH * 0.8, 3.0), layout="constrained")
axis.plot(
    spectrum.frequency,
    abs(spectrum.amplitude).max(axis=0),
    lw=1.0,
    color=SERIES[0],
    label="largest axis",
)
for i, entry in enumerate(bands):
    axis.axvspan(
        entry.f_min,
        entry.f_max,
        color=SERIES[3],
        alpha=0.15,
        lw=0,
        label="forbidden band" if i == 0 else None,
    )
    axis.hlines(entry.tolerance, entry.f_min, entry.f_max, color=SERIES[3], ls="--")
axis.set_xlim(0, 2000)
axis.set_xlabel("frequency (Hz)")
axis.set_ylabel("amplitude (mT/m)")
axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()
# sphinx_gallery_end_ignore

# %%
# Each spoke points in a different direction, so the train does not repeat
# the same waveform on any one axis, and its energy is spread over the
# spectrum rather than concentrated in a few lines. A Cartesian or EPI train
# repeats the same gradients every TR or every echo, and its spectrum is a
# comb at that rate and its harmonics: there, a forbidden band is avoided by
# changing the echo spacing so that no tooth falls in it.
#
# Specific absorption rate
# ------------------------
#
# The RF pulses deposit power in the subject, measured as the specific
# absorption rate, in W/kg. The check uses *virtual observation points*
# (VOPs), a compressed model of the local SAR of a coil, and averages the
# power over the sequence's repeating unit. The inversion pulse dominates it:
# 10 ms of adiabatic sweep, against the 8 degree excitations.
#
# :func:`~pypulseqpp.safety.example_vops` is a **synthetic** model: it has the
# shape of a real one, and its numbers mean nothing about any coil or any
# subject. A real one is read from a file with
# :func:`~pypulseqpp.safety.read_vops`.
demonstration = safety.example_vops()
sar_ok, sar = safety.check_sar(
    seq,
    demonstration.model,
    drive_per_hz=demonstration.drive_per_hz,
    default_shim=demonstration.cp_shim,
)
print(
    f"local {sar.worst_local.sar:.2f} W/kg (limit {sar.local_limit}), "
    f"global {sar.worst_global.sar:.2f} W/kg (limit {sar.global_limit})"
)

# %%
# Everything together
# -------------------

# sphinx_gallery_start_ignore
summary(
    [
        ("timing", ok, f"{len(errors)} errors"),
        ("gradient amplitude", grad_ok, f"{grad.per_axis.value * mT_per_m:.1f} mT/m"),
        ("slew rate", slew_ok, f"{slew.per_axis.value / system.gamma:.0f} T/m/s"),
        ("gradient continuity", cont_ok, f"{len(cont.discontinuities)} jumps"),
        ("nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
        (
            "mechanical resonance",
            mech_ok,
            f"{sum(b.violations for b in mech.bands)} windows over",
        ),
        ("SAR", sar_ok, f"{sar.worst_local.sar:.2f} W/kg local"),
    ]
)
# sphinx_gallery_end_ignore

# %%
# How each model works is explained in :doc:`/explanations/safety-checks`.
#
# As a spec
# ---------
#
# What this lesson built, stated the way you would ask an agent for it:
#
# .. code-block:: text
#
#    Run pypulseqpp's checks on mprage_stack_of_stars3D_sequence (220 mm,
#    128 matrix, 16 partitions over 80 mm, ry=3, TR 2.5 s, 8 degrees) for a
#    32 mT/m, 130 T/m/s scanner: check_timing, check_max_grad,
#    check_max_slew, check_grad_continuity, check_pns with a chronaxie model
#    (334 us, 23.4 T/s, 0.333), check_mech_resonance against 550-650 Hz at
#    6 mT/m and 1100-1300 Hz at 4 mT/m in 50 ms windows, and check_sar with
#    example_vops. Plot the gradient spectrum against the bands, and print
#    every verdict.
