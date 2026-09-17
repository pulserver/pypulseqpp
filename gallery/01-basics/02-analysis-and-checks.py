"""
==========================
Analysis and safety checks
==========================

The analyses a finished sequence is inspected with, and the hardware checks
that decide whether it can be played.

The sequence analysed is the shipped multi-slice echo-planar example, which
drives the gradient system hard enough for the checks to have something to
report. What each check computes, and the criterion it applies, is in
:doc:`../../../explanations/safety/index`.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.4),
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)
# sphinx_gallery_end_ignore
import numpy as np
from scipy.spatial.transform import Rotation

import pypulseqpp as pp
from pypulseqpp import safety, sequences

seq = sequences.epi2D_sequence(
    n_x=64, n_y=64, n_slices=3, fat_saturation=True, tr=None, n_dummy=0
)

# %%
# Structure and timing
# --------------------
#
# :meth:`~pypulseqpp.Sequence.test_report` states what the sequence encodes:
# the event counts, the timing, the resolution the k-space positions imply and
# the gradient extremes reached on each physical axis. It is upstream
# PyPulseq's report, computed over the compiled core.

print(seq.test_report())

# %%
# :meth:`~pypulseqpp.Sequence.check_timing` is a separate question: whether
# every event time is addressable on the raster it is played on, and whether
# the transmit and receive dead times the system declares are respected.

is_ok, report = seq.check_timing()
print(f"timing: {'ok' if is_ok else report}")

# %%
# Waveforms and k-space
# ---------------------
#
# :meth:`~pypulseqpp.Sequence.waveforms_and_times` expands the block table into
# one time-and-amplitude array per channel, and
# :meth:`~pypulseqpp.Sequence.calculate_kspace` integrates the gradient
# waveforms into the k-space location of every ADC sample, in 1/m, with each
# block's rotation applied. Both are what the figures below are drawn from.

k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
print(f"{k_traj_adc.shape[1]} samples over {t_excitation.size} excitations")

pp.plot.plot_kspace(seq, tr_range=[1, 1], color_by="echo")

# %%
# One shot of the same sequence as a timing diagram: fat saturation, the
# excitation, a navigator echo, and the train whose readout gradient reverses
# between echoes while the phase-encode blips step through k-space.

seq.paper_plot(tr=1)

# %%
# Gradient amplitude, slew rate and continuity
# --------------------------------------------
#
# The three gradient checks work on the physical axes, after each block's
# rotation is applied. Each returns a verdict and a report naming where the
# extreme was found. Amplitude and slew rate compare the largest per-axis peak
# with the limit; the simultaneous vector magnitude is reported beside it, and
# is not the norm of independently attained axis peaks.

for check in (
    safety.check_max_grad,
    safety.check_max_slew,
    safety.check_grad_continuity,
):
    is_ok, report = check(seq)
    print(f"{check.__name__}: {'within limits' if is_ok else 'exceeded'}")

#: Hz/m to mT/m, and Hz/m/s to T/m/s: amplitudes are stored in the
#: gyromagnetic-ratio-free units the file format uses.
to_mT_per_m = 1e3 / seq.system.gamma
to_T_per_m_per_s = 1.0 / seq.system.gamma

is_ok, report = safety.check_max_grad(seq)
print(
    f"per-axis peak {report.per_axis.value * to_mT_per_m:.1f} mT/m "
    f"on {report.per_axis.axis}, block {report.per_axis.block}"
)
print(f"vector peak   {report.vector.value * to_mT_per_m:.1f} mT/m")
print(f"limit         {report.limit * to_mT_per_m:.1f} mT/m")

# %%
# A check takes the limits it compares against from the sequence, or from an
# :class:`~pypulseqpp.Opts` passed to it. Reading a sequence designed elsewhere
# against the limits of the system it will actually be played on is the same
# call with a different second argument.

derated = pp.apply_system_derates(seq.system, grad_derate=0.7, slew_derate=0.7)
is_ok, report = safety.check_max_slew(seq, derated)
print(f"under a 70% derate: {'within limits' if is_ok else 'exceeded'}")
print(
    f"peak {report.per_axis.value * to_T_per_m_per_s:.0f} T/m/s, "
    f"limit {report.limit * to_T_per_m_per_s:.0f} T/m/s"
)

# %%
# Peripheral nerve stimulation
# ----------------------------
#
# :func:`~pypulseqpp.safety.check_pns` runs a nerve model over the slew of each
# physical axis and compares the root-sum-square of the per-axis responses with
# the model's threshold. The model below is upstream PyPulseq's SAFE example
# hardware description, whose coefficients differ per axis; a site's own
# description is read from a Siemens ``.asc`` file by
# :func:`~pypulseqpp.safety.read_safe_model`, and a rheobase-chronaxie model is
# stated directly as a :class:`~pypulseqpp.safety.ChronaxieModel`.

from pypulseq.utils.safe_pns_prediction import safe_example_hw

is_ok, report = safety.check_pns(seq, safe_example_hw())
print(
    f"peak {100 * report.peak.value:.0f}% of threshold "
    f"at {report.peak.time * 1e3:.1f} ms, block {report.peak.block}"
)
for axis in report.axes:
    print(f"  {axis.axis}: {100 * axis.value:.0f}%")

# %%
# The single-shot train exceeds this description's threshold, which is the
# verdict the check exists to produce. The largest response is on y, whose
# stimulation limit is the lowest of the three axes in this description, so the
# blips rather than the readout reversals decide the result. A sequence over
# threshold is redesigned — a lower slew rate, a longer echo spacing, or more
# shots — rather than re-checked.
#
# The response is evaluated for an axial prescription unless a rotation is
# given. An oblique prescription redistributes the same logical waveform over
# physical axes whose coefficients differ, so the combined response changes
# with the prescribed orientation.

for degrees in (0.0, 30.0, 60.0, 90.0):
    rotation = Rotation.from_euler("x", degrees, degrees=True).as_matrix()
    _, rotated = safety.check_pns(seq, safe_example_hw(), rotation=rotation)
    print(f"{degrees:4.0f} deg about x: {100 * rotated.peak.value:.0f}% of threshold")

# %%
# Mechanical resonance
# --------------------
#
# :func:`~pypulseqpp.safety.check_mech_resonance` slides a window along each
# physical gradient axis and compares every window's amplitude spectrum with
# the forbidden bands guarding that axis. An echo-planar train is a
# near-periodic drive at the reciprocal of twice the echo spacing, so it
# concentrates its gradient spectrum in a narrow line that a band either
# contains or does not.

bands = [
    safety.ForbiddenBand(axis=None, f_min=530.0, f_max=590.0, tolerance=2.0),
    safety.ForbiddenBand(axis=None, f_min=1090.0, f_max=1180.0, tolerance=0.0),
]
is_ok, report = safety.check_mech_resonance(seq, bands, min_threshold=10.0)
print(
    f"{report.windows} windows of {report.window_width * 1e3:.0f} ms, "
    f"{report.frequency_step:.1f} Hz bins, {report.backend}"
)
for band in report.bands:
    print(
        f"  {band.f_min:.0f}-{band.f_max:.0f} Hz: peak {band.peak:.2f} mT/m "
        f"at {band.frequency:.0f} Hz on {band.peak_axis}, "
        f"threshold {band.threshold:.1f} mT/m, {band.violations} violations"
    )

# %%
# A band whose table states a tolerance is judged against it; a band that
# states none is judged against ``min_threshold``. The worst window of every
# band is reported whether or not it violates, so the margin is readable from a
# sequence that passes as well as from one that does not. Below, the spectrum
# of the readout axis over the window the check found worst, with the bands
# drawn on it.


# sphinx_gallery_start_ignore
def _spectrum_figure(seq, report):
    """Draw the x-axis spectrum of the worst window, with every band on it."""
    worst = max(report.bands, key=lambda entry: entry.peak / entry.threshold)
    start, width = worst.window_start, report.window_width
    raster = seq.system.grad_raster_time
    times, amplitudes = seq.waveforms_and_times()[0][0]
    grid = np.arange(start, start + width, raster)
    sampled = np.interp(grid, times, amplitudes, left=0.0, right=0.0)
    taper = np.hanning(grid.size)
    padded = 3 * grid.size
    spectrum = np.fft.rfft((sampled - sampled.mean()) * taper, n=padded)
    frequency = np.fft.rfftfreq(padded, raster)
    magnitude = 2 * np.abs(spectrum) / taper.sum() * 1e3 / seq.system.gamma

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
    axis.plot(frequency, magnitude, color="0.25", linewidth=1.0)
    for entry in report.bands:
        axis.axvspan(entry.f_min, entry.f_max, color="tab:red", alpha=0.15, lw=0)
        axis.hlines(entry.threshold, entry.f_min, entry.f_max, color="tab:red", lw=1.4)
    axis.set_xlim(0, 2500)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("amplitude (mT/m)")
    axis.set_title(
        f"x gradient over the {width * 1e3:.0f} ms window at {start * 1e3:.0f} ms; "
        "forbidden bands and their thresholds in red"
    )
    figure.tight_layout()
    return figure


_spectrum_figure(seq, report)
# sphinx_gallery_end_ignore

# %%
# Specific absorption rate
# ------------------------
#
# :func:`~pypulseqpp.safety.check_sar` averages SAR from virtual observation
# points over each repetition the block definitions repeat with, and compares
# the worst repetition with a local and a global limit.
# :func:`~pypulseqpp.safety.example_vops` supplies a synthetic eight-channel
# model, its drive calibration and its circularly polarised shim; a measured
# model is read from a file with :func:`~pypulseqpp.safety.read_vops`.

example = safety.example_vops()
is_ok, report = safety.check_sar(
    seq,
    example.model,
    drive_per_hz=example.drive_per_hz,
    default_shim=example.cp_shim,
)
print(f"windows: {report.windows.first.size}, {report.tr_size} blocks each")
print(
    f"worst local  {report.worst_local.sar:.2f} W/kg "
    f"(limit {report.local_limit:.1f}) at VOP {report.worst_local.vop}"
)
print(
    f"worst global {report.worst_global.sar:.2f} W/kg (limit {report.global_limit:.1f})"
)

# %%
# Writing
# -------
#
# :meth:`~pypulseqpp.Sequence.write` produces Pulseq 1.5.1 text.
# :meth:`~pypulseqpp.Sequence.write_binary` produces the binary form of the
# same content, and :meth:`~pypulseqpp.Sequence.write_v141` the 1.4.1 text an
# older interpreter reads, folding ppm offsets into absolute offsets and
# dropping the fields that revision has no column for.

# sphinx_gallery_start_ignore
import tempfile
from pathlib import Path

directory = tempfile.TemporaryDirectory()
base = Path(directory.name)
# sphinx_gallery_end_ignore
seq.write(str(base / "epi.seq"))
seq.write_binary(str(base / "epi.bin"))

for name in ("epi.seq", "epi.bin"):
    print(f"{name}: {(base / name).stat().st_size / 1024:.0f} kB")
# sphinx_gallery_start_ignore
directory.cleanup()
# sphinx_gallery_end_ignore
