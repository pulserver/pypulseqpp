"""
==============================
Checking a sequence for safety
==============================

The package computes five checks over a finished sequence: the gradient
amplitude and slew rate the hardware is asked for, the continuity of the
gradient waveform across block boundaries, the nerve response the slew implies,
the gradient spectrum inside a scanner's forbidden bands, and the power a
transmit array deposits. Each returns a verdict and a report, and this page
runs all of them over one sequence.

A passing check does not establish that a sequence is safe to run on a scanner
or on a subject. The models used here are demonstrations: a scanner applies its
own, and its predownload gate and hardware monitor run whatever these say. The
physics behind each check is in
:doc:`/explanations/safety/index`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def summary(rows):
    """Print a check, its verdict and the reading it turned on."""
    print(f"{'check':30} {'result':8} {'reading':>24}")
    for name, ok, reading in rows:
        print(f"{name:30} {'pass' if ok else 'FAIL':8} {reading:>24}")


# sphinx_gallery_end_ignore

# %%
# The sequence
# ------------
#
# An echo-planar readout, which drives one gradient axis hard and repetitively
# and so has something to say to every check.

from pypulseqpp import safety
from pypulseqpp.sequences import epi2D_sequence

seq = epi2D_sequence(n_x=96, n_y=96, n_slices=1, n_shots=1, n_dummy=0)
print(f"{seq.num_blocks} blocks, {seq.duration()[0] * 1e3:.1f} ms")

# %%
# Gradient hardware
# -----------------
#
# The amplitude and slew checks compare the physical, rotated waveform against
# the system limits the sequence was designed under. They report the largest
# per-axis reading and the largest vector reading, which is not the norm of the
# per-axis peaks: two axes reach their own peaks at different times.
# The verdict is on the per-axis reading, which is what the hardware limits;
# the vector reading is reported beside it. ``check_grad_continuity`` looks for
# steps between blocks, which a scanner would have to slew through in no time
# at all.

grad_ok, grad = safety.check_max_grad(seq)
slew_ok, slew = safety.check_max_slew(seq)
cont_ok, cont = safety.check_grad_continuity(seq)

print(
    f"limit {grad.limit / seq.system.gamma * 1e3:.0f} mT/m per axis; "
    f"largest {grad.per_axis.value / seq.system.gamma * 1e3:.1f} mT/m on "
    f"{grad.per_axis.axis}, vector {grad.vector.value / seq.system.gamma * 1e3:.1f} mT/m"
)

# %%
# Peripheral nerve stimulation
# ----------------------------
#
# A changing gradient induces an electric field in the subject, and the nerve
# model turns the slew on each axis into a response as a fraction of the
# threshold at which stimulation is reported. The axes are combined as a
# root sum of squares, and the check passes while that stays below one.
#
# The chronaxie model below takes its three coefficients from the
# strength-duration relationship; a scanner supplies a SAFE model instead,
# which :func:`~pypulseqpp.safety.read_safe_model` reads from an ``.asc`` file.

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
pns_ok, pns = safety.check_pns(seq, model, trace=True)

print(
    f"peak {pns.peak.value:.2f} of threshold at {pns.peak.time * 1e3:.1f} ms, "
    f"in block {pns.peak.block}"
)
print("per axis:", ", ".join(f"{axis.axis} {axis.value:.2f}" for axis in pns.axes))

# %%
# ``trace=True`` returns the response the peak was taken from, so a diagram of
# it is the check's own calculation rather than a second one.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
for entry in pns.axes:
    axis.plot(pns.time * 1e3, entry.response, lw=0.8, label=f"$G_{entry.axis}$")
axis.plot(pns.time * 1e3, pns.response, lw=1.5, color="black", label="combined")
axis.axhline(1.0, color="tab:red", ls="--", lw=1.0)
axis.plot(pns.peak.time * 1e3, pns.peak.value, "o", color="tab:red", ms=5)
axis.set_xlabel("time (ms)")
axis.set_ylabel("response, fraction of threshold")
axis.set_title("Peripheral nerve stimulation response")
axis.legend(
    frameon=False, ncol=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0)
)
figure.tight_layout(rect=(0, 0, 0.82, 1))
# sphinx_gallery_end_ignore

# %%
# Mechanical resonance
# --------------------
#
# A gradient coil has mechanical modes, and driving one of them shakes the
# magnet. A scanner declares the frequency ranges to stay out of and how much
# amplitude it tolerates in each. The check takes the gradient spectrum in
# overlapping windows and reads the largest amplitude inside each band.

bands = [
    safety.ForbiddenBand(axis=None, f_min=550.0, f_max=650.0, tolerance=6.0),
    safety.ForbiddenBand(axis="y", f_min=1100.0, f_max=1300.0, tolerance=4.0),
]
mech_ok, mech = safety.check_mech_resonance(seq, bands, window_width=20e-3)

for band in mech.bands:
    print(
        f"{band.f_min:6.0f}-{band.f_max:6.0f} Hz on {band.axis or 'every axis':10}: "
        f"{band.peak:5.2f} mT/m at {band.frequency:6.0f} Hz "
        f"against {band.threshold:4.1f}, {band.violations} windows over"
    )

# %%
# ``mech_resonance_spectrum`` returns one window's spectrum through the same
# windowed pass, so the figure and the verdict read the same numbers. The
# readout train is periodic, so its spectrum is a comb at the echo-spacing
# frequency and its harmonics.

spectrum = safety.mech_resonance_spectrum(
    seq, window=mech.bands[0].window, window_width=20e-3
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
for name, amplitude in zip(spectrum.axes, spectrum.amplitude, strict=True):
    axis.plot(spectrum.frequency, amplitude, lw=0.9, label=f"$G_{name}$")
for band in mech.bands:
    axis.axvspan(band.f_min, band.f_max, color="tab:red", alpha=0.15, lw=0)
    axis.hlines(
        band.threshold, band.f_min, band.f_max, color="tab:red", ls="--", lw=1.0
    )
axis.set_xlim(0, 2500)
axis.set_xlabel("frequency (Hz)")
axis.set_ylabel("amplitude (mT/m)")
axis.set_title(
    f"Mechanical-resonance spectrum, window {spectrum.window} "
    f"at {spectrum.window_start * 1e3:.0f} ms"
)
axis.legend(
    frameon=False, ncol=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0)
)
figure.tight_layout(rect=(0, 0, 0.84, 1))
# sphinx_gallery_end_ignore

# %%
# Specific absorption rate
# ------------------------
#
# The SAR check integrates the power a transmit array deposits over the
# sequence's own repeating unit, against a set of virtual observation points.
# :func:`~pypulseqpp.safety.example_vops` returns a **synthetic** model with a
# circularly polarised shim: it is shaped like a real one and its numbers mean
# nothing about any coil or any subject. A scanner's model is read from a file
# with :func:`~pypulseqpp.safety.read_vops`.

demonstration = safety.example_vops()
sar_ok, sar = safety.check_sar(
    seq,
    demonstration.model,
    drive_per_hz=demonstration.drive_per_hz,
    default_shim=demonstration.cp_shim,
)

print(f"{len(sar.windows.first)} windows over a {sar.tr_size}-block repeating unit")
print(
    f"worst local {sar.worst_local.sar:.2f} W/kg against {sar.local_limit} W/kg, "
    f"worst global {sar.worst_global.sar:.2f} W/kg against {sar.global_limit} W/kg"
)

# %%
# Every verdict together
# ----------------------

# sphinx_gallery_start_ignore
summary(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / seq.system.gamma * 1e3:.1f} mT/m",
        ),
        ("slew rate", slew_ok, f"{slew.per_axis.value / seq.system.gamma:.0f} T/m/s"),
        (
            "gradient continuity",
            cont_ok,
            f"{len(cont.discontinuities)} discontinuities",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
        (
            "mechanical resonance",
            mech_ok,
            f"{max(band.violations for band in mech.bands)} windows over",
        ),
        ("specific absorption rate", sar_ok, f"{sar.worst_local.sar:.2f} W/kg local"),
    ]
)
# sphinx_gallery_end_ignore

# %%
# This configuration exceeds the nerve model's threshold, which is what an
# echo-planar train at a short echo spacing does on a body gradient system.
# Lengthening the echo spacing, reading fewer lines per train or lowering the
# slew rate the readout is designed under all move it; the notebook on
# :doc:`echo-planar imaging </generated/gallery/15-epi/epi2D_sequence>` shows
# what each of those costs.
