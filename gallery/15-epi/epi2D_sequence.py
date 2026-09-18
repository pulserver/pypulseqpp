"""
========================
2D echo-planar imaging
========================

One excitation followed by a train of readout lobes of alternating polarity,
with a phase-encode blip between them, so the whole phase-encode axis is
covered after a single pulse. Off-resonance then accumulates along that axis
instead of across repetitions, and the train length is what decides how far it
displaces the image.
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


def _views(seq, n_y):
    """Imaging views as (shot, echo index within the train, line from centre)."""
    labels = seq.evaluate_labels(evolution="adc")
    lin = np.asarray(labels["LIN"]) - n_y // 2
    nav, seg = np.asarray(labels["NAV"]), np.asarray(labels["SEG"])
    imaging = nav == 0
    starts = np.flatnonzero(imaging & ~np.roll(imaging, 1))
    stops = np.append(starts[1:], len(lin))
    out = []
    for a, b in zip(starts, stops, strict=True):
        keep = imaging[a:b]
        lines = lin[a:b][keep]
        out.append((int(seg[a]), np.arange(len(lines)), lines))
    return out


def traversal_figure(designs, n_y):
    """Line read against echo index, one panel per design."""
    figure, axes = plt.subplots(1, len(designs), figsize=(PAGE_WIDTH, 3.0), sharey=True)
    for axis, (title, seq) in zip(np.atleast_1d(axes), designs.items(), strict=True):
        trains = _views(seq, n_y)
        colours = plt.get_cmap("turbo")(
            np.linspace(0.1, 0.9, max(len({shot for shot, _, _ in trains}), 2))
        )
        for shot, echo, line in trains:
            axis.plot(echo, line, "-", lw=0.8, color=colours[shot], alpha=0.8)
            axis.plot(echo, line, ".", ms=4, color=colours[shot])
        axis.set_xlabel("echo index in train")
        axis.set_title(title)
        axis.grid(alpha=0.25, lw=0.4)
    np.atleast_1d(axes)[0].set_ylabel("$k_y$ (lines from centre)")
    figure.tight_layout()
    return figure


def coverage_figure(designs, n_y):
    """Which lines each design acquires, as a row per design."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 1.1 + 0.4 * len(designs)))
    for row, seq in enumerate(designs.values()):
        lines = np.concatenate([line for _, _, line in _views(seq, n_y)])
        axis.plot(
            np.sort(lines),
            np.full(lines.size, row),
            "|",
            ms=9,
            color=plt.get_cmap("turbo")(0.15 + 0.35 * row),
        )
    axis.set_yticks(range(len(designs)), list(designs))
    axis.set_xlabel("$k_y$ (lines from centre)")
    axis.set_ylim(-0.6, len(designs) - 0.4)
    axis.grid(axis="x", alpha=0.25, lw=0.4)
    figure.tight_layout()
    return figure


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


# sphinx_gallery_end_ignore

# %%
# Baseline: single shot
# ---------------------
#
# One excitation reads every line of the phase-encode axis. The train is as
# long as the matrix, and the echo spacing times the train length is what a
# spin at a given off-resonance is displaced by.

from pypulseqpp.sequences import epi2D_sequence

single = epi2D_sequence(n_x=96, n_y=96, n_slices=1, n_shots=1, n_dummy=0)
print(
    f"{single.num_blocks} blocks, {single.duration()[0] * 1e3:.1f} ms, "
    f"TE {single.get_definition('TE')[0] * 1e3:.2f} ms"
)

# %%
# Sequence diagram
# ----------------

single.paper_plot()

# %%
# Segmentation and in-plane acceleration
# --------------------------------------
#
# Both shorten the train, and they differ in what else they change.
# ``n_shots`` interleaves the lines over several excitations, so every line is
# still acquired and the scan takes proportionally longer. ``ry`` skips lines
# instead, which leaves the scan time alone and needs a parallel-imaging
# reconstruction to fill what was skipped. A spin at offset :math:`\Delta f`
# gains :math:`2\pi \Delta f\, \mathrm{esp}` of phase per echo, which is
# linear in :math:`k_y` and therefore a displacement of
# :math:`\Delta f \cdot \mathrm{esp} \cdot N_\mathrm{etl}` pixels: both
# routes shorten the train, and both shorten the distortion with it.

segmented = epi2D_sequence(n_x=96, n_y=96, n_slices=1, n_shots=3, n_dummy=0)
accelerated = epi2D_sequence(n_x=96, n_y=96, n_slices=1, ry=3, n_dummy=0, n_acs_y=0)

designs = {"1 shot": single, "3 shots": segmented, "ry = 3": accelerated}

# sphinx_gallery_start_ignore
# A spin at offset df gains 2*pi*df*esp of phase per echo, which is linear in
# k_y and so a displacement of df * esp * etl pixels, whatever step the train
# takes. The last column is that displacement per hertz of off-resonance.
print(
    f"{'':10} {'echoes':>7} {'trains':>7} {'per train':>10} {'TE (ms)':>9} "
    f"{'scan (ms)':>10} {'px per Hz':>10}"
)
for title, seq in designs.items():
    trains = _views(seq, 96)
    echoes = sum(len(line) for _, _, line in trains)
    per_train = echoes / len(trains)
    esp = seq.get_definition("EchoSpacing")[0]
    print(
        f"{title:10} {echoes:7d} {len(trains):7d} {per_train:10.1f} "
        f"{seq.get_definition('TE')[0] * 1e3:9.2f} {seq.duration()[0] * 1e3:10.1f} "
        f"{esp * per_train:10.3f}"
    )
# sphinx_gallery_end_ignore

# %%
# Echo traversal
# --------------
#
# The line each echo reads, against its index in the train. A single shot
# traverses the axis one line at a time; a segmented acquisition traverses it
# in steps of ``n_shots``, with each shot starting one line further on;
# acceleration traverses it in steps of ``ry`` and stops there.

# sphinx_gallery_start_ignore
traversal_figure(designs, 96)
# sphinx_gallery_end_ignore

# %%
# Which lines are acquired
# ------------------------
#
# Segmentation and acceleration produce the same train length from different
# sets of lines: the segmented acquisition covers the axis, the accelerated one
# leaves two lines in three unread.

# sphinx_gallery_start_ignore
coverage_figure(designs, 96)
# sphinx_gallery_end_ignore

# %%
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model and the forbidden bands below are
# demonstrations, not a scanner's.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
bands = [safety.ForbiddenBand(axis=None, f_min=550.0, f_max=650.0, tolerance=6.0)]

grad_ok, grad = safety.check_max_grad(single)
slew_ok, slew = safety.check_max_slew(single)
pns_ok, pns = safety.check_pns(single, model)
mech_ok, mech = safety.check_mech_resonance(single, bands, window_width=20e-3)

# sphinx_gallery_start_ignore
safety_table(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / single.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.per_axis.value / single.system.gamma:.0f} T/m/s",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
        ("mechanical resonance", mech_ok, f"{mech.bands[0].peak:.1f} mT/m in band"),
    ]
)
# sphinx_gallery_end_ignore

# %%
# Mechanical resonance
# --------------------
#
# The readout train is a periodic gradient waveform, so its spectrum is a comb
# at the echo-spacing frequency and its harmonics. A forbidden band that one of
# those lines falls in is driven for as long as the train lasts.
# ``mech_resonance_spectrum`` returns the windowed spectrum the check reads.

spectrum = safety.mech_resonance_spectrum(
    single, window=mech.bands[0].window, window_width=20e-3
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.8))
for name, amplitude in zip(spectrum.axes, spectrum.amplitude, strict=True):
    axis.plot(spectrum.frequency, amplitude, lw=0.9, label=f"$G_{name}$")
band = mech.bands[0]
axis.axvspan(band.f_min, band.f_max, color="tab:red", alpha=0.15, lw=0)
axis.axhline(band.threshold, color="tab:red", ls="--", lw=1.0)
axis.set_xlim(0, 2000)
axis.set_xlabel("frequency (Hz)")
axis.set_ylabel("amplitude (mT/m)")
axis.set_title("forbidden band shaded, its threshold dashed")
axis.legend(frameon=False, ncol=3, fontsize=9)
figure.tight_layout()
# sphinx_gallery_end_ignore
