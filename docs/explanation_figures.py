"""Figures for the explanation pages, rendered from the pypulseqpp being built.

Each entry of :data:`FIGURES` is a function returning a Matplotlib figure, and
:func:`render` writes one PNG per entry into the directory the pages reference.
Every figure is designed and analysed here rather than drawn once and committed,
so a figure cannot outlive the behaviour it reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAGE_WIDTH = 7.4  # inches, the width of the documentation column


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    return plt


def _physical_waveforms(seq, axes="xyz"):
    """Return (time, amplitude) arrays per physical axis, in s and mT/m."""
    waveforms = seq.waveforms_and_times()[0]
    scale = 1e3 / seq.system.gamma
    return [(w[0], w[1] * scale) for w in waveforms[: len(axes)]]


def axis_peaks_against_vector():
    """Per-axis peaks, their root-sum-square, and the simultaneous vector peak.

    The three axis peaks of a sequence are reached at different times, so their
    root-sum-square describes an instant the sequence never plays.
    """
    from pypulseqpp import safety, sequences

    plt = _pyplot()
    seq = sequences.gre_radial2D_sequence(
        fov=220e-3, n=64, n_slices=1, tr=None, n_dummy=0
    )
    scale = 1e3 / seq.system.gamma
    _, report = safety.check_max_grad(seq)
    axis_peaks = np.array([peak.value * scale for peak in report.axes])
    simultaneous = report.vector.value * scale

    period = seq.get_definition("TR")[0]
    grid = np.arange(0.0, period, seq.system.grad_raster_time)
    played = np.array(
        [
            np.interp(grid, times, amplitudes, left=0.0, right=0.0)
            for times, amplitudes in _physical_waveforms(seq)
        ]
    )
    magnitude = np.linalg.norm(played, axis=0)

    figure, (trace, bars) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 2.9), width_ratios=(2.0, 1.0)
    )
    for row, name in zip(played, ("$G_x$", "$G_y$", "$G_z$"), strict=True):
        trace.plot(grid * 1e3, row, lw=0.9, label=name)
    trace.plot(grid * 1e3, magnitude, lw=1.4, color="0.2", label="$|G|$")
    trace.set_xlabel("time (ms)")
    trace.set_ylabel("gradient amplitude (mT/m)")
    trace.set_title("one repetition")
    figure.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.36, 1.03), columnspacing=1.0)

    heights = [*axis_peaks, float(np.linalg.norm(axis_peaks)), simultaneous]
    colors = ["0.7", "0.7", "0.7", "tab:red", "tab:green"]
    bars.bar(["x", "y", "z", "RSS", "$|G|$"], heights, color=colors)
    for index, height in enumerate(heights):
        bars.text(index, height + 1.0, f"{height:.0f}", ha="center", fontsize=8)
    bars.set_ylim(0, 1.25 * max(heights))
    bars.set_ylabel("peak amplitude (mT/m)")
    bars.set_title("whole sequence")
    figure.tight_layout()
    return figure


def continuity_seam():
    """A step within the slew limit and one beyond it, at a block boundary.

    Continuity is the slew-rate inequality applied across a boundary, with one
    gradient raster period as the time available. Each panel is drawn from the
    two blocks it checks, and its title is the verdict the check returns.
    """
    import pypulseqpp as pp
    from pypulseqpp import safety

    plt = _pyplot()
    system = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
    raster = system.grad_raster_time
    scale = 1e3 / system.gamma
    allowed = system.max_slew * raster  # the largest legal step, in Hz/m

    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.9), sharey=True)
    for axis, fraction in ((axes[0], 0.8), (axes[1], 4.0)):
        ends_at = fraction * allowed
        ramp = np.linspace(0.0, ends_at, 8)
        seq = pp.Sequence(system=system)
        seq.add_block(
            pp.make_arbitrary_grad("x", ramp, system=system, first=0.0, last=ends_at)
        )
        seq.add_block(pp.make_trapezoid("x", area=2e-4, duration=10 * raster, system=system))
        is_ok, report = safety.check_grad_continuity(seq)

        boundary = ramp.size * raster
        following = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0]) * 0.6 * allowed
        axis.plot(np.arange(ramp.size) * raster * 1e6, ramp * scale, lw=1.4, color="0.2")
        axis.plot(
            (boundary + np.arange(following.size) * raster) * 1e6,
            following * scale,
            lw=1.4,
            color="0.2",
        )
        axis.axvline(boundary * 1e6, color="0.75", lw=0.8)
        axis.annotate(
            "",
            xy=(boundary * 1e6, 0.0),
            xytext=(boundary * 1e6, ends_at * scale),
            arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.1),
        )
        axis.text(
            boundary * 1e6 + 4,
            0.5 * ends_at * scale,
            f"$\\Delta G$ = {fraction:.1f} x the\nlargest legal step",
            color="tab:red",
            va="center",
            fontsize=8,
        )
        axis.set_title(
            f"{len(report.discontinuities)} discontinuity reported"
            if report.discontinuities
            else "no discontinuity reported"
        )
        axis.set_xlabel("time (us)")
    axes[0].set_ylabel("$G_x$ (mT/m)")
    figure.tight_layout()
    return figure


def strength_duration():
    """The slew rate at threshold against the duration it is held for.

    Each point is a single triangular gradient pulse whose amplitude is scaled
    until the check reports unity; the response of both model families is linear
    in amplitude, so one evaluation per duration fixes the threshold. The
    horizontal asymptote is the rheobase and the knee is at the chronaxie.
    """
    from pypulseq.utils.safe_pns_prediction import safe_example_hw

    import pypulseqpp as pp
    from pypulseqpp import safety

    plt = _pyplot()
    #: Wide enough not to constrain the probe pulses, which are not played.
    system = pp.Opts(
        max_grad=500.0, grad_unit="mT/m", max_slew=100000.0, slew_unit="T/m/s"
    )
    reference_mt_per_m = 10.0

    def response(ramp, model):
        seq = pp.Sequence(system=system)
        seq.add_block(
            pp.make_trapezoid(
                "x",
                amplitude=reference_mt_per_m * system.gamma / 1e3,
                rise_time=ramp,
                flat_time=0.0,
                fall_time=ramp,
                system=system,
            )
        )
        seq.add_block(pp.make_delay(10e-3))
        return safety.check_pns(seq, model)[1].peak.value

    models = {
        "chronaxie 360 us, rheobase 20 T/m/s": safety.ChronaxieModel(
            chronaxie=360e-6, rheobase=20.0, alpha=0.333
        ),
        "SAFE example description, x axis": safe_example_hw(),
    }
    ramps = np.array(
        [
            pp.round_to_raster(float(ramp), system.grad_raster_time)
            for ramp in np.geomspace(40e-6, 8e-3, 24)
        ]
    )

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for name, model in models.items():
        threshold = np.array(
            [reference_mt_per_m / response(ramp, model) for ramp in ramps]
        )
        axis.loglog(ramps * 1e3, 1e-3 * threshold / ramps, marker="o", ms=3, label=name)
    asymptote = 20.0 / 0.333  # the chronaxie model's rheobase over its alpha
    axis.axhline(asymptote, color="0.6", ls="--", lw=0.9)
    axis.text(3.0, 1.05 * asymptote, "rheobase / alpha", color="0.4", fontsize=8)
    axis.axvline(0.36, color="0.6", ls=":", lw=0.9)
    axis.text(0.38, 300.0, "chronaxie", color="0.4", fontsize=8, rotation=90)
    axis.set_xlabel("ramp duration (ms)")
    axis.set_ylabel("slew rate at threshold (T/m/s)")
    axis.set_title("Strength-duration relation of the two model families")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    return figure


def gradient_spectra():
    """Windowed gradient spectra of a Cartesian and an echo-planar readout.

    The alternating train concentrates its power in a line at the reciprocal of
    twice its echo spacing; the conventional readout spreads its power low.
    """
    from pypulseqpp import sequences

    plt = _pyplot()
    designed = {
        "spoiled gradient echo": sequences.gre2D_sequence(
            n_x=64, n_y=64, n_slices=1, tr=None
        ),
        "single-shot echo planar": sequences.epi2D_sequence(
            n_x=64, n_y=64, n_slices=1, tr=None, n_dummy=0, fat_saturation=False
        ),
    }

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    width = 40e-3
    for name, seq in designed.items():
        raster = seq.system.grad_raster_time
        times, amplitudes = seq.waveforms_and_times()[0][0]
        start = max(0.0, 0.5 * (seq.duration()[0] - width))
        grid = np.arange(start, start + width, raster)
        sampled = np.interp(grid, times, amplitudes, left=0.0, right=0.0)
        taper = np.hanning(grid.size)
        padded = 3 * grid.size
        spectrum = np.fft.rfft((sampled - sampled.mean()) * taper, n=padded)
        frequency = np.fft.rfftfreq(padded, raster)
        magnitude = 2 * np.abs(spectrum) / taper.sum() * 1e3 / seq.system.gamma
        axis.plot(frequency, magnitude, lw=1.0, label=name)

    axis.set_xlim(0, 3000)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("$G_x$ amplitude (mT/m)")
    axis.set_title(f"{width * 1e3:.0f} ms window at the middle of each sequence")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    return figure


#: Each figure's file name, without the extension, and the function that draws it.
FIGURES = {
    "axis_peaks_against_vector": axis_peaks_against_vector,
    "continuity_seam": continuity_seam,
    "strength_duration": strength_duration,
    "gradient_spectra": gradient_spectra,
}


def render(into: str | Path) -> None:
    """Draw every figure in :data:`FIGURES` into ``into``, one PNG each."""
    plt = _pyplot()
    directory = Path(into)
    directory.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        figure = draw()
        figure.savefig(directory / f"{name}.png", bbox_inches="tight")
        plt.close(figure)
