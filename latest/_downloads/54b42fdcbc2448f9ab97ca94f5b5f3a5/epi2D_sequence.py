"""
========================
2D echo-planar imaging
========================

A slice-selective excitation is followed by alternating readout gradients and
phase-encode blips that acquire multiple Cartesian lines in one echo train.
Spoilers suppress residual transverse coherence between repetitions.
Off-resonance phase accumulates during the train and produces geometric
distortion along the phase-encode axis. EPI supports rapid structural imaging
and functional MRI.
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


# sphinx_gallery_end_ignore

# %%
# Baseline: single shot
# ---------------------
#
# One excitation acquires the complete phase-encode axis. Echo-train length
# equals the number of acquired lines and determines the accumulated
# off-resonance phase across k-space.

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
# Segmentation and in-plane acceleration both reduce echo-train length.
# ``n_shots`` interleaves the lines over several excitations, so every line is
# still acquired. ``ry`` skips lines within one excitation and requires a
# parallel-imaging reconstruction for the omitted lines. Both reduce the
# echo-train duration. A spin at offset :math:`\Delta f`
# gains :math:`2\pi \Delta f\, \mathrm{esp}` of phase per echo, which is
# linear in :math:`k_y` and therefore a displacement of
# :math:`\Delta f \cdot \mathrm{esp} \cdot N_\mathrm{etl}` pixels: both
# routes shorten the train, and both shorten the distortion with it.

segmented = epi2D_sequence(n_x=96, n_y=96, n_slices=1, n_shots=3, n_dummy=0)
accelerated = epi2D_sequence(n_x=96, n_y=96, n_slices=1, ry=3, n_dummy=0, n_acs_y=0)

designs = {"1 shot": single, "3 shots": segmented, "ry = 3": accelerated}

# sphinx_gallery_start_ignore
# The final column reports displacement per hertz of off-resonance, calculated
# as echo spacing multiplied by echo-train length.
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
# The ordinate gives the phase-encode line acquired at each echo index. A
# single shot
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
