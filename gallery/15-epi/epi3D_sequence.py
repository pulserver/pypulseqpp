"""
========================
3D echo-planar imaging
========================

One excitation per shot, followed by a train of readout lobes of alternating
polarity that covers a shell of partitions. The views sampled form a CAIPIRINHA
lattice: line ``y`` is read when ``(y - n_y // 2) % ry == 0``, and the partition
it is read at advances by the CAIPI shift from one lattice line to the next.
A shot reads every ``n_shots``-th lattice line, which is skipped-CAIPI sampling
(Stirnberg and Stöcker, Magn Reson Med 2021, doi:10.1002/mrm.28486); one shot
per shell is blipped-CAIPI.
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


def traversal_figure(seq, ry, rz, n_shots, n_y, n_z, cells=3, ax=None):
    """Draw consecutive acquired views from the sequence labels, separated by shot."""
    paths = _trains(seq)
    shift = int(seq.get_definition("CaipiShift")[0])
    shots = n_shots

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
    for shot, (_, lin, par) in enumerate(paths[:shots]):
        keep = (lin >= y0 - ry * rz) & (lin < y0 + width + ry * rz)
        y = lin[keep] - y0 + 0.5
        # Fold equivalent shells onto one lattice cell in kz. The labels retain
        # the acquired partition and the path remains separated by shot.
        z = (par[keep] - z0) % rz + 0.5
        z = z + 0.06 * (shot - (shots - 1) / 2)
        leading = shot == 0
        colour = (
            "tab:red" if leading else plt.get_cmap("copper")(0.2 + 0.5 * shot / shots)
        )
        ax.plot(
            y,
            z,
            color=colour,
            lw=1.6 if leading else 0.9,
            label=f"shot {shot}",
            zorder=2,
        )
        for a in range(len(y) - 1):
            ax.annotate(
                "",
                xy=(y[a + 1], z[a + 1]),
                xytext=(y[a], z[a]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "lw": 1.2 if leading else 0.6,
                    "mutation_scale": 8 if leading else 5,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
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
    if shots > 1:
        ax.legend(frameon=False, ncol=shots, fontsize=8, loc="upper center")
    ax.set_title(
        rf"${shots}\cdot{{{ry}\times{rz}}}_{{z{shift}}}$:  "
        rf"$b^{{(1)}}={first},\ b^{{(2)}}={second},\ n={cycle}$"
    )
    return ax


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


# sphinx_gallery_end_ignore

# %%
# Accelerated acquisition
# -----------------------
#
# The baseline uses in-plane and partition acceleration, three shots per shell,
# and a nonzero CAIPI shift. Each shot reads every third sampled lattice line;
# successive echoes therefore contain both the skipped-line displacement and
# the partition jump.

from pypulseqpp.sequences import epi3D_sequence

baseline = epi3D_sequence(
    n_x=64,
    n_y=64,
    n_z=16,
    ry=2,
    rz=4,
    n_shots=3,
    n_dummy=0,
)
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
# connected directly in acquired order, with an arrow for every inter-echo
# step. No line joins separate shots. The partition jumps between consecutive
# echoes are the CAIPI blips: they alternate between
# amplitudes :math:`b^{(1)} = (S \cdot \Delta z) \bmod R_z` and
# :math:`b^{(2)} = (R_z - b^{(1)}) \bmod R_z`, and the pattern repeats every
# :math:`n` echoes. Equivalent shells are folded onto one lattice cell; a small
# vertical display offset separates coincident paths from different shots.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.4))
traversal_figure(baseline, ry=2, rz=4, n_shots=3, n_y=64, n_z=16, ax=axis)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Single-shot comparison
# ----------------------
#
# Reducing ``n_shots`` to one puts the same lattice in one longer train per
# shell. Three shots shorten the readout window and the geometric distortion
# along the phase-encode axis; the inter-echo jumps grow because each shot
# steps three sampled lattice lines at a time.

single_shot = epi3D_sequence(
    n_x=64,
    n_y=64,
    n_z=16,
    ry=2,
    rz=4,
    n_shots=1,
    n_dummy=0,
)

# sphinx_gallery_start_ignore
print(f"{'':9} {'echoes':>7} {'trains':>7} {'per train':>10} {'TE (ms)':>9}")
for name, seq in (("3 shots", baseline), ("1 shot", single_shot)):
    trains = _trains(seq)
    echoes = sum(len(lines) for _, lines, _ in trains)
    print(
        f"{name:9} {echoes:7d} {len(trains):7d} {echoes / len(trains):10.1f} "
        f"{seq.get_definition('TE')[0] * 1e3:9.2f}"
    )
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 4.6))
for axis, seq in zip(axes, (baseline, single_shot), strict=True):
    traversal_figure(
        seq,
        ry=2,
        rz=4,
        n_shots=3 if seq is baseline else 1,
        n_y=64,
        n_z=16,
        ax=axis,
    )
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# In-plane acceleration
# ---------------------
#
# ``ry`` skips lines, which shortens the train further and widens the lattice
# along :math:`k_y`. The sampled views stay on one lattice, so the aliases stay
# where the CAIPI shift puts them.

accelerated = epi3D_sequence(
    n_x=64,
    n_y=64,
    n_z=16,
    ry=3,
    rz=2,
    n_shots=3,
    n_dummy=0,
)
print(
    f"ry=3, rz=2: CAIPI shift {int(accelerated.get_definition('CaipiShift')[0])}, "
    f"{accelerated.num_blocks} blocks, {accelerated.duration()[0]:.2f} s"
)

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.4))
traversal_figure(accelerated, ry=3, rz=2, n_shots=3, n_y=64, n_z=16, ax=axis)
figure.tight_layout()
# sphinx_gallery_end_ignore

# %%
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model below is a demonstration, not a
# scanner's. This configuration exceeds its threshold, which is what an
# echo-planar train at a short echo spacing does on a body gradient system.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
grad_ok, grad = safety.check_max_grad(baseline)
slew_ok, slew = safety.check_max_slew(baseline)
cont_ok, cont = safety.check_grad_continuity(baseline)
pns_ok, pns = safety.check_pns(baseline, model, trace=True)

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

# %%
# Peripheral nerve stimulation
# ----------------------------
#
# The readout train drives one axis hard and repetitively, so the response
# reaches its peak within the first echoes and stays there. ``check_pns``
# returns the response it took its peak from.

# sphinx_gallery_start_ignore
figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.8))
for entry in pns.axes:
    axis.plot(pns.time * 1e3, entry.response, lw=0.8, label=f"$G_{entry.axis}$")
axis.plot(pns.time * 1e3, pns.response, lw=1.4, color="black", label="combined")
axis.axhline(1.0, color="tab:red", ls="--", lw=1.0)
axis.set_xlim(0, 60)
axis.set_xlabel("time (ms)")
axis.set_ylabel("response, fraction of threshold")
axis.set_title("threshold dashed")
axis.legend(frameon=False, ncol=4, fontsize=9)
figure.tight_layout()
# sphinx_gallery_end_ignore
