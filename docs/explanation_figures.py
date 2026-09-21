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


def rotation_against_per_axis_limit():
    """Per-axis peaks in the logical frame and on the physical axes after a rotation.

    A rotation preserves the vector magnitude at every instant and redistributes
    it over the physical axes, so the largest per-axis amplitude a prescription
    can produce is bounded by the vector magnitude rather than by the logical
    per-axis peaks.
    """
    from scipy.spatial.transform import Rotation

    import pypulseqpp as pp
    from pypulseqpp import safety, sequences

    plt = _pyplot()
    system = pp.Opts(max_grad=32.0, grad_unit="mT/m", max_slew=140.0, slew_unit="T/m/s")
    logical = sequences.gre2D_sequence(
        system=system,
        fov_x=0.2,
        fov_y=0.2,
        n_x=128,
        n_y=128,
        n_slices=1,
        tr=None,
        n_dummy=0,
        readout_bandwidth_hz=400e3,
    )
    rotation = Rotation.from_euler("zy", [45.0, 45.0], degrees=True)
    physical = pp.TransformFOV(rotation=rotation).apply_to_sequence(logical)

    scale = 1e3 / system.gamma
    limit = system.max_grad * scale
    grid = np.arange(0.0, logical.get_definition("TR")[0], system.grad_raster_time)

    figure = plt.figure(figsize=(PAGE_WIDTH, 4.4))
    layout = figure.add_gridspec(
        2, 2, width_ratios=(2.0, 1.2), hspace=0.5, wspace=0.34,
        left=0.09, right=0.98, top=0.86, bottom=0.10,
    )
    traces = [figure.add_subplot(layout[0, 0])]
    traces.append(figure.add_subplot(layout[1, 0], sharex=traces[0], sharey=traces[0]))
    bars = figure.add_subplot(layout[:, 1])

    peaks = {}
    titles = ("logical axes", "physical axes, double-oblique prescription")
    for axis, seq, title in zip(traces, (logical, physical), titles, strict=True):
        played = np.array(
            [
                np.interp(grid, times, amplitudes, left=0.0, right=0.0)
                for times, amplitudes in _physical_waveforms(seq)
            ]
        )
        magnitude = np.linalg.norm(played, axis=0)
        axis.fill_between(grid * 1e3, magnitude, color="0.55", alpha=0.20, lw=0)
        axis.plot(grid * 1e3, magnitude, lw=1.0, color="0.45", label="$|G|$")
        for row, name in zip(played, ("$G_x$", "$G_y$", "$G_z$"), strict=True):
            axis.plot(grid * 1e3, row, lw=1.2, label=name)
        for sign in (1.0, -1.0):
            axis.axhline(sign * limit, color="tab:red", lw=0.9, ls="--")
        axis.set_title(title)
        axis.set_ylabel("$G$ (mT/m)")
        axis.margins(y=0.18)
        _, report = safety.check_max_grad(seq)
        peaks[title] = [peak.value * scale for peak in report.axes]
        peaks[title].append(report.vector.value * scale)
    traces[1].set_xlabel("time (ms)")
    traces[0].text(
        0.1, limit, "max_grad", color="tab:red", va="bottom", ha="left", fontsize=8
    )
    figure.legend(
        *traces[0].get_legend_handles_labels(),
        frameon=False,
        ncols=4,
        loc="upper left",
        bbox_to_anchor=(0.07, 1.0),
        columnspacing=1.2,
    )

    names = ("x", "y", "z", "$|G|$")
    offsets = np.arange(len(names))
    for shift, (title, heights) in zip((-0.19, 0.19), peaks.items(), strict=True):
        drawn = bars.bar(offsets + shift, heights, width=0.36, label=title.split(",")[0])
        # Only the per-axis peaks are compared with the limit; the vector peak
        # is reported beside them and carries no verdict.
        for index, (rectangle, height) in enumerate(zip(drawn, heights, strict=True)):
            if index < 3 and height > limit:
                rectangle.set_color("tab:red")
            bars.text(
                rectangle.get_x() + 0.5 * rectangle.get_width(),
                height + 0.8,
                f"{height:.0f}",
                ha="center",
                fontsize=6.5,
            )
    bars.axhline(limit, color="tab:red", lw=0.9, ls="--")
    bars.set_xticks(offsets, names)
    bars.set_ylim(0, 1.18 * max(max(heights) for heights in peaks.values()))
    bars.set_ylabel("peak amplitude (mT/m)")
    bars.set_title("peaks over the scan")
    bars.legend(frameon=False, fontsize=7, loc="upper left")
    return figure


def continuity_seam():
    """Legal and illegal physical-axis steps at a block boundary."""
    import pypulseqpp as pp

    plt = _pyplot()
    system = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
    raster = system.grad_raster_time
    allowed = system.max_slew * raster
    scale = 1e3 / system.gamma
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.7), sharey=True)
    for axis, fraction, verdict in zip(
        axes, (0.8, 4.0), ("pass", "fail"), strict=True
    ):
        endpoint = fraction * allowed
        before = np.linspace(0.0, endpoint, 8)
        boundary = before.size * raster
        after = np.array([0.0, 0.35, 0.6, 0.6, 0.35, 0.0]) * allowed
        axis.plot(
            np.arange(before.size) * raster * 1e6,
            before * scale,
            lw=1.5,
            color="0.2",
        )
        axis.plot(
            (boundary + np.arange(after.size) * raster) * 1e6,
            after * scale,
            lw=1.5,
            color="0.2",
        )
        axis.axvline(boundary * 1e6, color="0.65", lw=0.9, ls="--")
        axis.annotate(
            "",
            xy=(boundary * 1e6, 0.0),
            xytext=(boundary * 1e6, endpoint * scale),
            arrowprops={"arrowstyle": "<->", "color": "tab:red", "lw": 1.2},
        )
        axis.text(
            boundary * 1e6 - 2,
            0.5 * endpoint * scale,
            r"$\Delta G$",
            color="tab:red",
            ha="right",
            va="center",
        )
        axis.set_title(f"{fraction:.1f} × limit — {verdict}")
        axis.set_xlabel("time (µs)")
        axis.margins(x=0.08, y=0.2)
    axes[0].set_ylabel(r"$G_x$ (mT/m)")
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


def pns_response():
    """Checker-backed PNS response of a short-echo-spacing EPI shot."""
    import pypulseqpp as pp
    from pypulseqpp import safety, sequences

    plt = _pyplot()
    seq = sequences.epi2D_sequence(
        n_x=48,
        n_y=48,
        n_slices=1,
        n_dummy=0,
        tr=None,
        fat_saturation=False,
        readout_bandwidth_hz=500e3,
    )
    model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
    _, report = safety.check_pns(seq, model, trace=True)
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for entry in report.axes:
        axis.plot(
            report.time * 1e3,
            entry.response,
            lw=0.8,
            label=rf"$R_{entry.axis}(t)$",
        )
    axis.plot(report.time * 1e3, report.response, color="black", lw=1.4, label=r"$R(t)$")
    axis.axhline(1.0, color="tab:red", ls="--", lw=1.0, label="threshold")
    axis.plot(
        report.peak.time * 1e3,
        report.peak.value,
        "o",
        color="tab:red",
        ms=5,
        label="reported peak",
    )
    axis.set_xlabel("time (ms)")
    axis.set_ylabel("response (fraction of threshold)")
    axis.set_title("EPI peripheral-nerve-stimulation response")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout(rect=(0, 0, 0.80, 1))
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
    "rotation_against_per_axis_limit": rotation_against_per_axis_limit,
    "continuity_seam": continuity_seam,
    "strength_duration": strength_duration,
    "pns_response": pns_response,
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
