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

    from figure_style import FIGURE_RCPARAMS

    plt.rcParams.update(
        {
            **FIGURE_RCPARAMS,
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
    trace.plot(grid * 1e3, magnitude, lw=1.4, color="0.5", label="$|G|$")
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
    """The worst in-plane gradient vector under a prescription rotation.

    A rotation preserves the vector magnitude and redistributes it over the
    physical axes, so a vector longer than ``max_grad`` leaves the per-axis
    limit box at some orientations however its components are shared out at
    the design orientation.
    """
    from scipy.spatial.transform import Rotation

    import pypulseqpp as pp
    from pypulseqpp import safety, sequences

    plt = _pyplot()
    system = pp.Opts(max_grad=32.0, grad_unit="mT/m", max_slew=140.0, slew_unit="T/m/s")
    #: The two in-plane axes play together in the prewinder and in the
    #: rewinder-and-spoiler block, so a design solved against the per-axis
    #: limit puts a vector of `sqrt(2)` times that limit on them.
    designs = {
        "solved against max_grad": system,
        "solved against max_grad / $\\sqrt{2}$": pp.apply_system_derates(
            system, grad_derate=2**-0.5, slew_derate=2**-0.5
        ),
    }
    prescription = np.arange(0.0, 180.0, 2.5)

    scale = 1e3 / system.gamma
    limit = system.max_grad * scale
    vectors, sweeps = {}, {}
    for name, limits in designs.items():
        seq = sequences.gre2D_sequence(
            system=limits,
            fov_x=0.2,
            fov_y=0.2,
            n_x=128,
            n_y=128,
            n_slices=1,
            tr=None,
            n_dummy=0,
            readout_bandwidth_hz=400e3,
        )
        grid = np.arange(0.0, seq.get_definition("TR")[0], system.grad_raster_time)
        played = np.array(
            [
                np.interp(grid, times, amplitudes, left=0.0, right=0.0)
                for times, amplitudes in _physical_waveforms(seq)
            ]
        )
        instant = int(np.argmax(np.hypot(played[0], played[1])))
        vectors[name] = played[:2, instant]
        sweeps[name] = [
            max(peak.value for peak in safety.check_max_grad(turned)[1].axes) * scale
            for turned in (
                pp.TransformFOV(
                    rotation=Rotation.from_euler("z", angle, degrees=True)
                ).apply_to_sequence(seq)
                for angle in prescription
            )
        ]

    figure = plt.figure(figsize=(PAGE_WIDTH, 5.9))
    layout = figure.add_gridspec(
        2, 2, height_ratios=(1.35, 1.0), hspace=0.62, wspace=0.28,
        left=0.10, right=0.97, top=0.80, bottom=0.09,
    )
    planes = [figure.add_subplot(layout[0, column]) for column in (0, 1)]
    sweep = figure.add_subplot(layout[1, :])

    turn = np.arange(0.0, 360.0, 15.0)
    span = 1.45 * limit
    for axis, (name, vector) in zip(planes, vectors.items(), strict=True):
        magnitude = float(np.hypot(*vector))
        axis.add_patch(
            plt.Circle(
                (0, 0), limit, facecolor="tab:green", alpha=0.10, lw=0, zorder=0
            )
        )
        axis.add_patch(
            plt.Rectangle(
                (-limit, -limit), 2 * limit, 2 * limit,
                facecolor="none", edgecolor="tab:red", lw=1.0, ls="--", zorder=1,
            )
        )
        circle = np.linspace(0.0, 2 * np.pi, 361)
        axis.plot(
            magnitude * np.cos(circle), magnitude * np.sin(circle),
            color="0.55", lw=0.8, ls=":", zorder=1,
        )
        for angle in turn:
            radians = np.deg2rad(angle)
            rotated = np.array(
                [
                    [np.cos(radians), -np.sin(radians)],
                    [np.sin(radians), np.cos(radians)],
                ]
            ) @ vector
            outside = np.max(np.abs(rotated)) > limit
            axis.annotate(
                "",
                xy=tuple(rotated),
                xytext=(0.0, 0.0),
                zorder=3 if outside else 2,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "tab:red" if outside else "0.5",
                    "lw": 1.1,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
        axis.set_title(f"{name}\n$|G|$ = {magnitude:.1f} mT/m", fontsize=9)
        axis.set_xlim(-span, span)
        axis.set_ylim(-span, span)
        axis.set_aspect("equal")
        axis.set_xlabel("$G_x$ (mT/m)")
    planes[0].set_ylabel("$G_y$ (mT/m)")

    handles = [
        plt.Line2D([], [], color="tab:red", lw=1.0, ls="--",
                   label=f"per-axis limit, {limit:.0f} mT/m"),
        plt.Rectangle((0, 0), 1, 1, facecolor="tab:green", alpha=0.20, lw=0,
                      label="inside the limit at every orientation"),
        plt.Line2D([], [], color="0.5", lw=1.1, label="within the per-axis limit"),
        plt.Line2D([], [], color="tab:red", lw=1.1, label="over the per-axis limit"),
    ]
    figure.legend(
        handles=handles, frameon=False, ncols=2, loc="upper left",
        bbox_to_anchor=(0.10, 1.0), columnspacing=1.4,
    )

    for name, peaks in sweeps.items():
        sweep.plot(prescription, peaks, lw=1.4, label=name)
    sweep.axhline(limit, color="tab:red", lw=0.9, ls="--")
    sweep.set_xlim(prescription[0], prescription[-1])
    sweep.set_xticks(np.arange(0.0, 181.0, 30.0))
    sweep.set_xlabel("prescription rotation about z (degrees)")
    sweep.set_ylabel("largest per-axis\namplitude (mT/m)")
    sweep.legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.42),
    )
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
            color="0.5",
        )
        axis.plot(
            (boundary + np.arange(after.size) * raster) * 1e6,
            after * scale,
            lw=1.5,
            color="0.5",
        )
        axis.axvline(boundary * 1e6, color="0.55", lw=0.9, ls="--")
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
    axis.axhline(asymptote, color="0.55", ls="--", lw=0.9)
    axis.text(3.0, 1.05 * asymptote, "rheobase / alpha", color="0.55", fontsize=8)
    axis.axvline(0.36, color="0.55", ls=":", lw=0.9)
    axis.text(0.38, 300.0, "chronaxie", color="0.55", fontsize=8, rotation=90)
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
    axis.plot(report.time * 1e3, report.response, color="0.5", lw=1.4, label=r"$R(t)$")
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
