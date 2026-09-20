"""
========================
3D echo-planar imaging
========================

A slab-selective excitation is followed by alternating readout gradients with
phase-encode and partition blips. Segmented skipped-CAIPI traversal distributes
a three-dimensional Cartesian lattice among shots. Spoilers suppress residual
transverse coherence between repetitions; off-resonance accumulates during
each echo train. 3D EPI supports rapid structural and functional imaging.
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


def _trains(seq):
    """Each train's views in echo order, as (shot, lines, partitions)."""
    labels = seq.evaluate_labels(evolution="adc")
    lin, par = np.asarray(labels["LIN"]), np.asarray(labels["PAR"])
    nav, seg = np.asarray(labels["NAV"]), np.asarray(labels["SEG"])
    imaging = nav == 0
    starts = np.flatnonzero(imaging & ~np.roll(imaging, 1))
    stops = np.append(starts[1:], len(lin))
    return [
        (int(seg[a]), lin[a:b][imaging[a:b]], par[a:b][imaging[a:b]])
        for a, b in zip(starts, stops, strict=True)
    ]


def _parabola(p0, p1, n=13):
    """The path between two samples: triangular blips integrate to a parabola."""
    x = np.linspace(p0[0], p1[0], n)
    a = 2 * (p1[1] - p0[1]) / (p1[0] - p0[0]) ** 2
    half = n // 2
    return x, np.concatenate(
        (
            a * (x[:half] - p0[0]) ** 2 + p0[1],
            -a * (x[half:] - p1[0]) ** 2 + p1[1],
        )
    )


def traversal_figure(seq, ry, rz, n_y, n_z, cells=3, ax=None):
    """The sampling lattice over a few cells, with the traversal drawn on it."""
    paths = _trains(seq)
    shift = int(seq.get_definition("CaipiShift")[0])
    shots = len({shot for shot, _, _ in paths})

    width = cells * ry * rz
    y0 = n_y // 2 - width // 2
    z0 = int(min(par.min() for _, _, par in paths if par.min() >= n_z // 2))
    partitions = np.arange(z0, z0 + rz)

    sampled = np.zeros((rz, width), dtype=bool)
    for _, lin, par in paths:
        inside = (lin >= y0) & (lin < y0 + width) & np.isin(par, partitions)
        sampled[par[inside] - z0, lin[inside] - y0] = True

    ax = ax or plt.gca()
    ax.pcolormesh(
        np.arange(width + 1),
        np.arange(rz + 1),
        sampled + 0.3,
        cmap="gray",
        vmin=0,
        vmax=1,
        edgecolor=[0.35, 0.35, 0.35],
        lw=0.6,
    )
    for shot, lin, par in paths[:shots]:
        keep = (lin >= y0 - ry * rz) & (lin < y0 + width + ry * rz)
        y, z = lin[keep] - y0 + 0.5, par[keep] - z0 + 0.5
        leading = shot == 0
        colour = (
            "tab:red" if leading else plt.get_cmap("copper")(0.2 + 0.5 * shot / shots)
        )
        for a in range(len(y) - 1):
            ax.plot(
                *_parabola((y[a], z[a]), (y[a + 1], z[a + 1])),
                color=colour,
                lw=1.6 if leading else 0.9,
                zorder=3 if leading else 2,
            )
        ax.scatter(y, z, s=18 if leading else 7, color=colour, zorder=4)

    first = (shots * shift) % rz
    second = (rz - first) % rz
    smallest = min(first, second)
    cycle = 1 if smallest == 0 else (rz // smallest if rz % smallest == 0 else rz)
    ax.set_xlim(0, width)
    ax.set_ylim(0, rz)
    ax.invert_yaxis()
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$k_y$")
    ax.set_ylabel("$k_z$")
    ax.set_title(
        rf"${shots}\cdot{{{ry}\times{rz}}}_{{z{shift}}}$:  "
        rf"$b^{{(1)}}={first},\ b^{{(2)}}={second},\ n={cycle}$"
    )
    return ax


# sphinx_gallery_end_ignore

# %%
# Baseline: one shot per shell
# ----------------------------
#
# A single shot per shell reads every lattice line of that shell, blipping to
# each line's partition as it goes. ``ry`` and ``rz`` set the lattice; the
# CAIPI shift follows from them and is written into the sequence definitions.

from pypulseqpp.sequences import epi3D_sequence

baseline = epi3D_sequence(n_x=64, n_y=64, n_z=16, ry=1, rz=8, n_shots=1, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"CAIPI shift {int(baseline.get_definition('CaipiShift')[0])}, "
    f"TE {baseline.get_definition('TE')[0] * 1e3:.2f} ms, "
    f"TR {baseline.get_definition('TR')[0] * 1e3:.1f} ms"
)

# %%
# Sequence diagram
# ----------------

baseline.paper_plot()

# %%
# Skipped-CAIPI traversal
# -----------------------
#
# Each cell of the lattice is one ``(line, partition)`` view, white where it is
# sampled. The path is the order the echoes of one train read those views,
# drawn as the parabolas the triangular blips integrate to. The partition jumps
# between consecutive echoes are the CAIPI blips: they alternate between
# amplitudes :math:`b^{(1)} = (S \cdot \Delta z) \bmod R_z` and
# :math:`b^{(2)} = (R_z - b^{(1)}) \bmod R_z`, and the pattern repeats every
# :math:`n` echoes.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.4))
traversal_figure(baseline, ry=1, rz=8, n_y=64, n_z=16, ax=axis)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Segmented: three shots per shell
# --------------------------------
#
# Raising ``n_shots`` divides the same lattice between shots, each reading
# every third lattice line. The echo train shortens in proportion, which is
# what shortens the readout window and the geometric distortion along the
# phase-encode axis, and the blips grow because a shot steps three lattice
# lines at a time.

segmented = epi3D_sequence(n_x=64, n_y=64, n_z=16, ry=1, rz=8, n_shots=3, n_dummy=0)

# sphinx_gallery_start_ignore
print(f"{'':9} {'echoes':>7} {'trains':>7} {'per train':>10} {'TE (ms)':>9}")
for name, seq in (("1 shot", baseline), ("3 shots", segmented)):
    trains = _trains(seq)
    echoes = sum(len(lines) for _, lines, _ in trains)
    print(
        f"{name:9} {echoes:7d} {len(trains):7d} {echoes / len(trains):10.1f} "
        f"{seq.get_definition('TE')[0] * 1e3:9.2f}"
    )
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 4.6))
for axis, seq in zip(axes, (baseline, segmented), strict=True):
    traversal_figure(seq, ry=1, rz=8, n_y=64, n_z=16, ax=axis)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# In-plane acceleration
# ---------------------
#
# ``ry`` skips lines, which shortens the train further and widens the lattice
# along :math:`k_y`. The sampled views stay on one lattice, so the aliases stay
# where the CAIPI shift puts them.

accelerated = epi3D_sequence(n_x=64, n_y=64, n_z=16, ry=2, rz=4, n_shots=2, n_dummy=0)
print(
    f"ry=2, rz=4: CAIPI shift {int(accelerated.get_definition('CaipiShift')[0])}, "
    f"{accelerated.num_blocks} blocks, {accelerated.duration()[0]:.2f} s"
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.4))
traversal_figure(accelerated, ry=2, rz=4, n_y=64, n_z=16, ax=axis)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
