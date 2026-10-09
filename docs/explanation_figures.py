"""Figures for the explanation pages, rendered from the pypulseqpp being built.

Each entry of :data:`FIGURES` is a function returning a Matplotlib figure, and
:func:`render` writes one PNG per entry into the directory the pages reference.
Every figure is designed and analysed here rather than drawn once and committed,
so a figure cannot outlive the behaviour it reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pypulseqpp.plot._style import FAINT, MUTED, SERIES

PAGE_WIDTH = 7.8  # inches, the width of the documentation column


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figure_style import FIGURE_RCPARAMS

    plt.rcParams.update(FIGURE_RCPARAMS)
    return plt


def _played_waveforms(seq, axes="xyz"):
    """Return (time, amplitude) arrays per axis after each block's rotation, in s and mT/m."""
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
            for times, amplitudes in _played_waveforms(seq)
        ]
    )
    magnitude = np.linalg.norm(played, axis=0)

    figure, (trace, bars) = plt.subplots(
        1, 2, figsize=(PAGE_WIDTH, 3.4), width_ratios=(2.0, 1.0), layout="constrained"
    )
    for row, name in zip(played, ("$G_x$", "$G_y$", "$G_z$"), strict=True):
        trace.plot(grid * 1e3, row, lw=0.9, label=name)
    trace.plot(grid * 1e3, magnitude, lw=1.4, color="0.5", label="$|G|$")
    trace.set_xlabel("time (ms)")
    trace.set_ylabel("gradient amplitude (mT/m)")
    trace.set_title("one repetition")
    figure.legend(ncols=4, loc="outside upper left", columnspacing=1.0)

    heights = [*axis_peaks, float(np.linalg.norm(axis_peaks)), simultaneous]
    colors = [MUTED, MUTED, MUTED, "C7", "C2"]
    bars.bar(["x", "y", "z", "RSS", "$|G|$"], heights, color=colors)
    for index, height in enumerate(heights):
        bars.text(index, height + 1.0, f"{height:.0f}", ha="center", fontsize="x-small")
    bars.set_ylim(0, 1.25 * max(heights))
    bars.set_ylabel("peak amplitude (mT/m)")
    bars.set_title("whole sequence")
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
                for times, amplitudes in _played_waveforms(seq)
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

    figure = plt.figure(figsize=(PAGE_WIDTH, 7.0), layout="constrained")
    layout = figure.add_gridspec(2, 2, height_ratios=(1.5, 1.0))
    planes = [figure.add_subplot(layout[0, column]) for column in (0, 1)]
    sweep = figure.add_subplot(layout[1, :])

    turn = np.arange(0.0, 360.0, 15.0)
    span = 1.45 * limit
    for axis, (name, vector) in zip(planes, vectors.items(), strict=True):
        magnitude = float(np.hypot(*vector))
        axis.add_patch(
            plt.Circle((0, 0), limit, facecolor="C2", alpha=0.10, lw=0, zorder=0)
        )
        axis.add_patch(
            plt.Rectangle(
                (-limit, -limit),
                2 * limit,
                2 * limit,
                facecolor="none",
                edgecolor="C7",
                lw=1.0,
                ls="--",
                zorder=1,
            )
        )
        circle = np.linspace(0.0, 2 * np.pi, 361)
        axis.plot(
            magnitude * np.cos(circle),
            magnitude * np.sin(circle),
            color="0.55",
            lw=0.8,
            ls=":",
            zorder=1,
        )
        for angle in turn:
            radians = np.deg2rad(angle)
            rotated = (
                np.array(
                    [
                        [np.cos(radians), -np.sin(radians)],
                        [np.sin(radians), np.cos(radians)],
                    ]
                )
                @ vector
            )
            outside = np.max(np.abs(rotated)) > limit
            axis.annotate(
                "",
                xy=tuple(rotated),
                xytext=(0.0, 0.0),
                zorder=3 if outside else 2,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "C7" if outside else "0.5",
                    "lw": 1.1,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
        axis.set_title(f"{name}\n$|G|$ = {magnitude:.1f} mT/m", fontsize="small")
        axis.set_xlim(-span, span)
        axis.set_ylim(-span, span)
        axis.set_aspect("equal")
        axis.set_xlabel("$G_x$ (mT/m)")
    planes[0].set_ylabel("$G_y$ (mT/m)")

    handles = [
        plt.Line2D(
            [],
            [],
            color="C7",
            lw=1.0,
            ls="--",
            label=f"per-axis limit, {limit:.0f} mT/m",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="C2",
            alpha=0.20,
            lw=0,
            label="inside the limit at every orientation",
        ),
        plt.Line2D([], [], color="0.5", lw=1.1, label="within the per-axis limit"),
        plt.Line2D([], [], color="C7", lw=1.1, label="over the per-axis limit"),
    ]
    figure.legend(handles=handles, ncols=2, loc="outside upper center")

    for name, peaks in sweeps.items():
        sweep.plot(prescription, peaks, lw=1.4, label=name)
    sweep.axhline(limit, color="C7", lw=0.9, ls="--")
    sweep.set_xlim(prescription[0], prescription[-1])
    sweep.set_xticks(np.arange(0.0, 181.0, 30.0))
    sweep.set_xlabel("prescription rotation about z (degrees)")
    sweep.set_ylabel("largest per-axis\namplitude (mT/m)")
    sweep.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2)
    return figure


def continuity_seam():
    """Legal and illegal steps at a block boundary."""
    import pypulseqpp as pp

    plt = _pyplot()
    system = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
    raster = system.grad_raster_time
    allowed = system.max_slew * raster
    scale = 1e3 / system.gamma
    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.7), sharey=True)
    for axis, fraction, verdict in zip(axes, (0.8, 4.0), ("pass", "fail"), strict=True):
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
            arrowprops={"arrowstyle": "<->", "color": "C7", "lw": 1.2},
        )
        axis.text(
            boundary * 1e6 - 2,
            0.5 * endpoint * scale,
            r"$\Delta G$",
            color="C7",
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

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6), layout="constrained")
    for name, model in models.items():
        threshold = np.array(
            [reference_mt_per_m / response(ramp, model) for ramp in ramps]
        )
        axis.loglog(ramps * 1e3, 1e-3 * threshold / ramps, marker="o", ms=3, label=name)
    asymptote = 20.0 / 0.333  # the chronaxie model's rheobase over its alpha
    axis.axhline(asymptote, color="0.55", ls="--", lw=0.9)
    axis.text(
        0.045, 1.08 * asymptote, "rheobase / alpha", color="0.55", fontsize="x-small"
    )
    axis.axvline(0.36, color="0.55", ls=":", lw=0.9)
    axis.text(0.38, 300.0, "chronaxie", color="0.55", fontsize="x-small", rotation=90)
    axis.set_xlabel("ramp duration (ms)")
    axis.set_ylabel("slew rate at threshold (T/m/s)")
    axis.set_title("Strength-duration relation of the two model families")
    figure.legend(loc="outside upper center", ncols=2)
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
    )[-1]
    model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
    _, report = safety.check_pns(seq, model, trace=True)
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6), layout="constrained")
    for entry in report.axes:
        axis.plot(
            report.time * 1e3,
            entry.response,
            lw=0.8,
            label=rf"$R_{entry.axis}(t)$",
        )
    axis.plot(report.time * 1e3, report.response, color="0.5", lw=1.4, label=r"$R(t)$")
    axis.axhline(1.0, color="C7", ls="--", lw=1.0, label="threshold")
    axis.plot(
        report.peak.time * 1e3,
        report.peak.value,
        "o",
        color="C7",
        ms=5,
        label="reported peak",
    )
    axis.set_xlabel("time (ms)")
    axis.set_ylabel("response (fraction of threshold)")
    axis.set_title("EPI peripheral-nerve-stimulation response")
    figure.legend(loc="outside upper center", ncols=6, columnspacing=1.2)
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
        )[-1],
    }

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.6), layout="constrained")
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
    figure.legend(loc="outside upper center", ncols=2)
    return figure


def safety_checks_on_one_sequence():
    """Three checks on one echo-planar shot, each a played quantity against its limit.

    The gradient amplitude against ``max_grad``, the nerve response against its
    threshold, and the windowed spectrum against a forbidden band placed on the
    train's fundamental, all read from the same played waveforms.
    """
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
    )[-1]
    scale = 1e3 / seq.system.gamma
    limit = seq.system.max_grad * scale
    _, pns = safety.check_pns(
        seq,
        safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333),
        trace=True,
    )
    width = 20e-3
    spectrum = safety.mech_resonance_spectrum(seq, window_width=width)
    readout = abs(spectrum.amplitude[0])
    fundamental = float(spectrum.frequency[np.argmax(readout)])
    band = safety.ForbiddenBand(
        axis=None, f_min=fundamental - 150.0, f_max=fundamental + 150.0, tolerance=5.0
    )
    mech_ok, _ = safety.check_mech_resonance(seq, [band], window_width=width)

    figure, (grad, nerve, spec) = plt.subplots(
        3, 1, figsize=(PAGE_WIDTH, 7.6), layout="constrained"
    )
    for (times, amplitudes), name in zip(
        _played_waveforms(seq, "xy"), ("$G_x$", "$G_y$"), strict=True
    ):
        grad.plot(times * 1e3, amplitudes, lw=0.9, label=name)
    for sign in (1, -1):
        grad.axhline(
            sign * limit,
            color="C7",
            ls="--",
            lw=1.0,
            label="limit" if sign > 0 else None,
        )
    grad.set_ylabel("mT/m")
    grad.set_title("gradient amplitude against max_grad")
    grad.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    nerve.plot(pns.time * 1e3, pns.response, color=SERIES[0], lw=1.0, label="$R(t)$")
    nerve.axhline(1.0, color="C7", ls="--", lw=1.0, label="threshold")
    nerve.set_xlim(grad.get_xlim())
    nerve.set_xlabel("time (ms)")
    nerve.set_ylabel("fraction")
    nerve.set_title(f"nerve response against threshold: peak {pns.peak.value:.2f}")
    nerve.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    spec.plot(spectrum.frequency, readout, color=SERIES[0], lw=1.0, label="$G_x$")
    spec.axvspan(band.f_min, band.f_max, color="C7", alpha=0.2, lw=0, label="band")
    spec.hlines(
        band.tolerance, band.f_min, band.f_max, color="C7", ls="--", label="tolerance"
    )
    spec.set_xlim(0, 3 * fundamental)
    spec.set_xlabel("frequency (Hz)")
    spec.set_ylabel("mT/m")
    verdict = "passes" if mech_ok else "violated"
    spec.set_title(f"{width * 1e3:.0f} ms window spectrum against a band: {verdict}")
    spec.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return figure


# ---------------------------------------------------------------------------
#  Pulseq representation, shapes and storage, timing and rasters
# ---------------------------------------------------------------------------


def block_table_and_libraries():
    """The block table of a written file, the libraries it indexes, and the shapes.

    Every cell of the block table below the duration is a library id, and zero
    is the absence of an event on that channel. The counts are those of a
    written eight-line gradient-echo file: a library holds one row per
    distinct event, however many blocks reference it.
    """
    import re
    import tempfile

    from pypulseqpp import sequences

    plt = _pyplot()
    seq = sequences.gre2D_sequence(
        fov_x=220e-3, n_x=64, n_y=8, n_slices=1, n_dummy=0, tr=15e-3
    )
    path = Path(tempfile.mkdtemp()) / "measured.seq"
    seq.write(path)
    text = path.read_text()

    parts = re.split(r"^\[(\w+)\]\s*$", text, flags=re.M)
    body = dict(zip(parts[1::2], parts[2::2], strict=True))

    def rows(name):
        """Return the data lines of one section, without its comments."""
        lines = body.get(name, "").splitlines()
        return [line for line in lines if line.strip() and not line.startswith("#")]

    columns = ("NUM", "DUR", "RF", "GX", "GY", "GZ", "ADC", "EXT")
    table = [line.split() for line in rows("BLOCKS")[:4]]
    shapes = len(re.findall(r"^shape_id ", body.get("SHAPES", ""), flags=re.M))

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 5.0))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")

    def panel(x, y, width, height, colour, title, count, note):
        """Draw one outlined library box with its row count and its fields."""
        axis.add_patch(
            plt.Rectangle(
                (x, y), width, height, facecolor="none", edgecolor=colour, lw=1.0
            )
        )
        axis.text(
            x + width / 2,
            y + height - 3.5,
            title,
            ha="center",
            va="center",
            fontsize="small",
            color=colour,
        )
        axis.text(
            x + width / 2,
            y + height - 8.5,
            f"{count} rows",
            ha="center",
            va="center",
            fontsize="x-small",
            color=colour,
        )
        axis.text(
            x + width / 2,
            y + height - 12.0,
            note,
            ha="center",
            va="top",
            fontsize="x-small",
            color=MUTED,
            linespacing=1.3,
        )

    cell, line_height, left, top = 8.0, 6.0, 22.0, 96.0
    axis.text(
        left,
        top + 1.5,
        f"[BLOCKS] — {len(rows('BLOCKS'))} rows, the played order",
        fontsize="small",
        color=SERIES[0],
        va="bottom",
    )
    for index, name in enumerate(columns):
        axis.text(
            left + (index + 0.5) * cell,
            top - 1.4,
            name,
            ha="center",
            va="center",
            fontsize="x-small",
            color=MUTED,
            family="monospace",
        )
    for line, cells in enumerate(table):
        y = top - 3.5 - (line + 1) * line_height
        if line % 2 == 0:
            axis.add_patch(
                plt.Rectangle(
                    (left, y),
                    len(columns) * cell,
                    line_height,
                    facecolor=FAINT,
                    edgecolor="none",
                )
            )
        for index, value in enumerate(cells):
            axis.text(
                left + (index + 0.5) * cell,
                y + line_height / 2,
                value,
                ha="center",
                va="center",
                fontsize="x-small",
                color=MUTED,
                family="monospace",
            )

    libraries = (
        (
            SERIES[1],
            "[RF]",
            len(rows("RF")),
            "amplitude, shape ids,\ndelay, offsets, centre, use",
        ),
        (
            SERIES[2],
            "[TRAP], [GRADIENTS]",
            len(rows("TRAP")) + len(rows("GRADIENTS")),
            "amplitude with rise, flat\nand fall, or a shape id",
        ),
        (
            SERIES[3],
            "[ADC]",
            len(rows("ADC")),
            "samples, dwell, delay,\nfrequency and phase offsets",
        ),
        (
            SERIES[4],
            "[EXTENSIONS]",
            len(rows("EXTENSIONS")),
            "one linked list per block:\nlabels, rotations, triggers",
        ),
    )
    width, gap = 23.0, 2.6
    for index, (colour, title, count, note) in enumerate(libraries):
        x = index * (width + gap)
        panel(x, 26.0, width, 22.0, colour, title, count, note)
        axis.annotate(
            "",
            xy=(x + width / 2, 48.5),
            xytext=(x + width / 2, 67.0),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.0},
        )

    panel(
        14.0,
        0.0,
        72.0,
        21.0,
        SERIES[5],
        "[SHAPES]",
        shapes,
        "run-length encoded on the derivative, so a\nthousand-sample "
        "linear ramp is three numbers",
    )
    for index in (0, 1):
        x = index * (width + gap) + width / 2
        axis.annotate(
            "",
            xy=(x + 9.0, 21.5),
            xytext=(x, 25.5),
            arrowprops={
                "arrowstyle": "-|>",
                "color": MUTED,
                "lw": 1.0,
                "connectionstyle": "arc3,rad=0.15",
            },
        )
    figure.tight_layout()
    return figure


def gre_repetition_blocks():
    """One gradient-echo repetition, as the blocks it is written as.

    The events of a block play concurrently and the blocks play back to back,
    so the whole repetition is fixed by the block durations and the delay of
    each event within its block. A block whose duration exceeds the extent of
    its events is a delay, which is how the echo time and the repetition time
    are realized.
    """
    from pypulseqpp import sequences

    plt = _pyplot()
    seq = sequences.gre2D_sequence(
        fov_x=220e-3, n_x=64, n_y=8, n_slices=1, n_dummy=0, tr=15e-3
    )
    last = 6  # the blocks of one repetition, pulse to closing delay
    edges = np.cumsum([0.0, *(seq.block_durations[i] for i in range(1, last + 1))])
    span = edges[-1]
    channels = seq.waveforms_and_times(append_RF=True, block_range=(1, last))[0]
    t_adc = seq.adc_times()[0]
    t_adc = t_adc[t_adc <= span]

    figure, axes = plt.subplots(
        6,
        1,
        figsize=(PAGE_WIDTH, 4.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.5, 1.2, 1, 1, 1, 0.5]},
    )
    strip = axes[0]
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True), 1):
        strip.add_patch(
            plt.Rectangle(
                (left * 1e3, 0.15),
                (right - left) * 1e3,
                0.7,
                facecolor=FAINT,
                edgecolor=MUTED,
                lw=0.8,
            )
        )
        strip.text(
            0.5 * (left + right) * 1e3,
            0.5,
            str(index),
            ha="center",
            va="center",
            fontsize="x-small",
            color=MUTED,
        )
    strip.set_ylim(0, 1)
    strip.set_ylabel("block", rotation=0, ha="right", va="center")
    strip.set_yticks([])

    rows = (
        ("RF", channels[3], SERIES[0]),
        ("$G_x$", channels[0], SERIES[2]),
        ("$G_y$", channels[1], SERIES[3]),
        ("$G_z$", channels[2], SERIES[4]),
    )
    for axis, (name, channel, colour) in zip(axes[1:5], rows, strict=True):
        wave = np.asarray(channel)
        # The RF channel carries the complex envelope; the gradients are real.
        amplitude = np.abs(wave[1]) if name == "RF" else wave[1].real
        height = np.abs(amplitude).max() if wave.shape[1] else 1.0
        if wave.shape[1]:
            axis.plot(wave[0].real * 1e3, amplitude / height, lw=1.3, color=colour)
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        axis.set_yticks([])
        axis.set_ylim(-1.25, 1.25)

    adc = axes[5]
    adc.plot(t_adc * 1e3, np.zeros_like(t_adc), "|", ms=8, color=SERIES[1])
    adc.set_ylabel("ADC", rotation=0, ha="right", va="center")
    adc.set_yticks([])
    adc.set_ylim(-1, 1)
    adc.set_xlabel("time from the start of the repetition (ms)")

    for axis in axes:
        for edge in edges:
            axis.axvline(edge * 1e3, color=FAINT, lw=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    adc.set_xlim(-0.2, span * 1e3 + 0.2)
    figure.tight_layout()
    return figure


def rotation_against_materialised_shapes():
    """Shapes written and file size against interleaves, for the two representations.

    The same spiral acquisition is written twice: once as one interleaf with a
    ``ROTATIONS`` extension per block, and once with every shot's waveform
    rotated into the file. Both are deduplicated as they are written, which is
    why the materialized form shares shapes at the counts whose rotations map
    an axis onto an axis.
    """
    import re
    import tempfile

    import pypulseqpp as pp
    from pypulseqpp import sequences

    plt = _pyplot()
    system = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
    rf = pp.make_block_pulse(np.deg2rad(10.0), duration=0.2e-3, system=system)
    readout = sequences.SpiralReadout2D(
        system, rf, fov=220e-3, matrix=64, design_interleaves=16
    )
    folder = Path(tempfile.mkdtemp())

    def written(seq):
        """Return the shapes the written file holds and its size in kilobytes."""
        path = folder / "measured.seq"
        seq.write(path)
        shapes = len(re.findall(r"^shape_id ", path.read_text(), flags=re.M))
        return shapes, path.stat().st_size / 1024

    counts = [8, 16, 24, 32, 48, 64]
    measured = {"rotation extension": [], "rotated waveforms": []}
    for interleaves in counts:
        turned, materialised = pp.Sequence(system), pp.Sequence(system)
        for shot in range(interleaves):
            angle = 2 * np.pi * shot / interleaves
            turned.add_block(rf)
            turned.add_block(
                readout.gx, readout.gy, readout.adc, pp.make_rotation(angle)
            )
            materialised.add_block(rf)
            materialised.add_block(
                *pp.rotate(readout.gx, readout.gy, angle=angle, axis="z"),
                readout.adc,
            )
        measured["rotation extension"].append(written(turned))
        measured["rotated waveforms"].append(written(materialised))

    figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2), layout="constrained")
    for index, (label, values) in enumerate(measured.items()):
        shapes = [shape for shape, _ in values]
        sizes = [size for _, size in values]
        axes[0].plot(counts, shapes, "o-", ms=3.5, color=SERIES[index], label=label)
        axes[1].plot(counts, sizes, "o-", ms=3.5, color=SERIES[index], label=label)
    axes[0].set_ylabel("shapes in the file")
    axes[1].set_ylabel("file size (kB)")
    for axis in axes:
        axis.set_xlabel("interleaves")
        axis.set_xticks(counts)
        axis.margins(y=0.15)
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2
    )
    return figure


def bandwidth_against_sample_count():
    """The highest receiver bandwidth each sample count admits, against a request.

    The acquisition window has to be a whole number of gradient raster periods
    and the dwell a whole number of ADC raster periods, so the shortest dwell a
    sample count admits is ``a r / gcd(N, r)``, with ``a`` the ADC raster and
    ``r`` the ratio of the two. What :func:`~pypulseqpp.calc_adc_timing`
    returns is the first admissible dwell at or above the requested one, so a
    request above the ceiling is met at the ceiling and not at the request.
    """
    from math import gcd

    import pypulseqpp as pp

    plt = _pyplot()
    system = pp.Opts()
    adc_raster = system.adc_raster_time
    grad_raster = system.grad_raster_time
    ratio = round(grad_raster / adc_raster)
    requested = 250e3

    counts = np.arange(96, 161)
    ceiling = np.array(
        [1.0 / (adc_raster * ratio / gcd(int(n), ratio)) for n in counts]
    )

    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 2.7))
    axis.vlines(counts, 0.0, ceiling * 1e-3, lw=1.0, color=FAINT)
    axis.plot(counts, ceiling * 1e-3, "o", ms=3.5, color=SERIES[0])
    axis.axhline(requested * 1e-3, lw=1.0, ls="--", color=SERIES[1])
    axis.text(
        counts[-1],
        requested * 1e-3 + 14.0,
        "requested",
        ha="right",
        va="bottom",
        fontsize="x-small",
        color=SERIES[1],
    )
    axis.set_xlabel("ADC samples")
    axis.set_ylabel("highest receiver\nbandwidth (kHz)")
    axis.set_xlim(counts[0] - 1, counts[-1] + 1)
    axis.set_ylim(0.0, 1e-3 / adc_raster * 1.1)
    figure.tight_layout()
    return figure


#: Each figure's file name, without the extension, and the function that draws it.
def modules_in_one_train():
    """One inversion and the start of its radial train, shaded by the module that built each block.

    The sequence function adds ``inversion.blocks``, a delay placed from the
    modules' ``duration`` and ``center``, and then ``readout.blocks`` once per
    spoke; the loop owns the order and the delay, the modules own the blocks.
    """
    import pypulseqpp as pp
    from pypulseqpp import sequences

    import warnings

    warnings.filterwarnings("ignore", message="Specified RF delay")
    plt = _pyplot()
    system = pp.Opts(
        max_grad=32.0,
        grad_unit="mT/m",
        max_slew=130.0,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=20e-6,
        adc_dead_time=10e-6,
    )
    inversion = sequences.InversionPreparation(system, voxel_size_m=5e-3)
    excitation = sequences.SpatialSelectiveExcitation(
        system, 8.0, 80e-3, duration_s=1e-3, is_slab=True
    )
    readout = sequences.RadialStackReadout(
        system,
        excitation.rf,
        excitation.gz,
        fov=220e-3,
        matrix=128,
        fov_z=80e-3,
        matrix_z=16,
        readout_bandwidth_hz=100e3,
        spoiling_cycles=4.0,
    )
    ti = 60e-3  # shortened so the train is visible beside the inversion
    wait = pp.round_to_raster(
        ti - (inversion.duration - inversion.center) - excitation.center,
        system.block_duration_raster,
    )
    n_spokes = 3
    seq = pp.Sequence(system)
    owner = []  # (module, first block, last block)
    for block in inversion.blocks:
        seq.add_block(*block)
    owner.append(("InversionPreparation", 1, len(inversion.blocks)))
    seq.add_block(pp.make_delay(wait))
    owner.append(("delay", len(inversion.blocks) + 1, len(inversion.blocks) + 1))
    first = len(inversion.blocks) + 2
    for n in range(n_spokes):
        for block in readout.blocks:
            seq.add_block(*block, pp.make_rotation(n * 2.0))
        owner.append(("RadialStackReadout", first, first + len(readout.blocks) - 1))
        first += len(readout.blocks)
    # The readout opens with the excitation, whose blocks the readout lays out.
    edges = np.concatenate(
        [[0.0], np.cumsum([seq.block_durations[i] for i in range(1, first)])]
    )
    span = edges[-1]
    channels = seq.waveforms_and_times(append_RF=True)[0]

    colours = {"InversionPreparation": SERIES[0], "RadialStackReadout": SERIES[2]}
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(PAGE_WIDTH, 3.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1, 1, 1, 0.5]},
    )
    rows = (
        ("RF", channels[3], SERIES[1]),
        ("$G_x$", channels[0], MUTED),
        ("$G_y$", channels[1], MUTED),
        ("$G_z$", channels[2], MUTED),
    )
    for axis, (name, channel, colour) in zip(axes[:4], rows, strict=True):
        wave = np.asarray(channel)
        amp = np.abs(wave[1]) if name == "RF" else wave[1].real
        if name == "RF":
            # Each module's pulse to its own peak: the inversion is far stronger.
            t = wave[0].real
            amp = amp.astype(float).copy()
            for module, lo, hi in owner:
                inside = (t >= edges[lo - 1]) & (t <= edges[hi])
                if module != "delay" and inside.any():
                    amp[inside] /= amp[inside].max() or 1.0
            height = 1.0
        else:
            height = np.abs(amp).max() if wave.shape[1] else 1.0
        if wave.shape[1]:
            axis.plot(wave[0].real * 1e3, amp / height, lw=1.1, color=colour)
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        axis.set_yticks([])
        axis.set_ylim(-1.25, 1.25)
    adc = axes[4]
    t_adc = seq.adc_times()[0]
    adc.plot(t_adc * 1e3, np.zeros_like(t_adc), "|", ms=7, color=SERIES[1])
    adc.set_ylabel("ADC", rotation=0, ha="right", va="center")
    adc.set_yticks([])
    adc.set_ylim(-1, 1)
    adc.set_xlabel("time from the start of the train (ms)")

    for module, lo, hi in owner:
        left, right = edges[lo - 1] * 1e3, edges[hi] * 1e3
        colour = colours.get(module)
        for axis in axes:
            if colour:
                axis.axvspan(left, right, color=colour, alpha=0.14, lw=0)
            axis.axvline(left, color=FAINT, lw=0.6)
            axis.axvline(right, color=FAINT, lw=0.6)
    for module, colour in colours.items():
        axes[0].fill_between([], [], color=colour, alpha=0.3, label=module)
    gap = 0.5 * (edges[len(inversion.blocks)] + edges[len(inversion.blocks) + 1]) * 1e3
    axes[0].text(
        gap,
        0.0,
        "delay placed from\nduration and center",
        ha="center",
        va="center",
        fontsize="small",
        color=MUTED,
    )
    axes[0].set_ylabel("RF", rotation=0, ha="right", va="center")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
    axes[4].set_xlim(-0.5, span * 1e3 + 0.5)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    fig = modules_in_one_train()
    for bg, name in (("white", "w"), ("#121212", "d")):
        fig.savefig(f"/tmp/fig_sm_{name}.png", dpi=110, facecolor=bg, transparent=False)


def sampling_support_order_and_angles():
    """A support, the echo at which each view is played, and golden-angle spokes.

    Left: a Poisson-disc support with its calibration block, each view coloured
    by the echo of the adaptive radial order that plays it, the centre at echo 4
    of 16. Middle and right: the first 21 spokes of the golden angle and of the
    tiny golden angle of index 7, coloured by acquisition index.
    """
    import numpy as np

    import pypulseqpp as pp

    from matplotlib.colors import LinearSegmentedColormap

    plt = _pyplot()
    # Both ends of the ramp are series colours, which read on either background.
    ramp = LinearSegmentedColormap.from_list("order", [SERIES[0], SERIES[1]])
    n, etl, centre_echo = 64, 16, 4
    mask = pp.make_poisson_disc_mask((n, n), 6.0, calib=(8, 8), seed=1)
    views = np.argwhere(mask)
    trains = pp.make_radial_adaptive_order(
        views - (n // 2, n // 2), etl, center_echo=centre_echo, pad=True
    )
    echo = np.zeros(len(views), dtype=int)
    for train in trains:
        for e, row in enumerate(train):
            if row is not None:
                echo[row] = e

    figure, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH, 3.0), layout="constrained")
    scatter = axes[0].scatter(
        views[:, 1], views[:, 0], c=echo, cmap=ramp, s=4, marker="s"
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("kz index")
    axes[0].set_ylabel("ky index")
    axes[0].set_title("support by echo")
    figure.colorbar(scatter, ax=axes[0], label="echo", shrink=0.8)

    for axis, angles, title in (
        (axes[1], pp.calc_golden_angles(21), "golden angle"),
        (axes[2], pp.calc_tiny_golden_angles(21, index=7), "tiny golden, index 7"),
    ):
        colours = ramp(np.linspace(0, 1, len(angles)))
        for angle, colour in zip(angles, colours):
            axis.plot(
                [-np.cos(angle), np.cos(angle)],
                [-np.sin(angle), np.sin(angle)],
                color=colour,
                lw=1.0,
            )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title)
    figure.colorbar(
        plt.cm.ScalarMappable(cmap=ramp),
        ax=axes[2],
        label="spoke order",
        shrink=0.8,
        ticks=[0, 1],
    ).ax.set_yticklabels(["first", "last"])
    return figure


def designed_pulse_and_gradient():
    """Two SLR excitations through ``sim_bloch``, and a spiral path timed by ``traj_to_grad``.

    The envelopes of a linear-phase and a minimum-phase design of one
    time-bandwidth product, the transverse magnetisation each leaves across
    frequency, and the gradient magnitude the solver assigns to a spiral
    against ``max_grad``.
    """
    import pypulseqpp as pp

    plt = _pyplot()
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    figure, (env, prof, grad) = plt.subplots(
        3, 1, figsize=(PAGE_WIDTH, 7.2), layout="constrained"
    )
    duration = 3e-3
    offsets = np.linspace(-6e3, 6e3, 601)
    for label, kind, color in (
        ("filter_type='ls'", "ls", "C0"),
        ("filter_type='min'", "min", "C1"),
    ):
        rf = pp.make_slr_pulse(
            np.pi / 2,
            duration=duration,
            time_bw_product=4.0,
            pulse_type="ex",
            filter_type=kind,
            system=system,
        )
        rf = rf[0] if isinstance(rf, tuple) else rf
        b1 = np.asarray(rf.signal)
        dt = duration / b1.size
        t = (np.arange(b1.size) + 0.5) * dt
        env.plot(t * 1e3, abs(b1), color=color, lw=1.0, label=label)
        m = pp.sim_bloch(b1, offsets[:, None], dt)
        prof.plot(offsets * 1e-3, np.hypot(m[:, 0], m[:, 1]), color=color, lw=1.0)
    env.set_xlabel("time (ms)")
    env.set_ylabel("$|B_1|$ (Hz)")
    env.set_title("SLR excitation envelope")
    env.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    prof.set_xlabel("off-resonance (kHz)")
    prof.set_ylabel("$|M_{xy}|$")
    prof.set_title("profile from sim_bloch, no relaxation")

    theta = np.linspace(0, 8 * np.pi, 2000)
    radius = np.linspace(0, 250.0, 2000)
    k = np.stack([radius * np.cos(theta), radius * np.sin(theta)])
    limited = pp.Opts(max_grad=22, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    g, _ = pp.traj_to_grad(k, system=limited)
    scale = 1e3 / system.gamma
    t = np.arange(g.shape[1]) * system.grad_raster_time
    grad.plot(
        t * 1e3, np.linalg.norm(g, axis=0) * scale, color="C2", lw=1.0, label="$|G|$"
    )
    grad.axhline(
        limited.max_grad * scale, color="C7", ls="--", lw=1.0, label="max_grad"
    )
    grad.set_xlabel("time (ms)")
    grad.set_ylabel("mT/m")
    grad.set_title("spiral path timed by traj_to_grad")
    grad.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return figure


def kspace_reset_and_unbroken():
    """k_x of two spin-echo shots: the unbroken gradient integral and the excitation-aware trajectory.

    The unbroken integral is the reference of the RF and ADC shift phases;
    excitation resets the trajectory and a refocusing pulse inverts it, which
    places the echo.
    """
    import pypulseqpp as pp
    from pypulseqpp.plot._style import SERIES

    plt = _pyplot()
    system = pp.Opts()
    seq = pp.Sequence(system)
    readout = pp.make_trapezoid("x", flat_area=2000, flat_time=3.2e-3, system=system)
    adc = pp.make_adc(64, duration=3.2e-3, delay=readout.rise_time, system=system)
    for _ in range(2):
        seq.add_block(
            pp.make_block_pulse(
                np.pi / 2, duration=1e-3, use="excitation", system=system
            )
        )
        seq.add_block(
            pp.make_trapezoid("x", area=readout.area / 2, duration=2e-3, system=system)
        )
        seq.add_block(
            pp.make_block_pulse(np.pi, duration=1e-3, use="refocusing", system=system)
        )
        seq.add_block(readout, adc)
        seq.add_block(pp.make_delay(2e-3))
    k_adc, _, t_exc, t_ref, t_adc = seq.calculate_kspace()
    ((times, amplitude),) = _played_waveforms(seq, "x")
    grid = np.linspace(0, times[-1], 4000)
    g = np.interp(grid, times, amplitude * system.gamma * 1e-3)
    big_k = np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(grid))])
    at = lambda t: np.interp(t, grid, big_k)  # noqa: E731
    reset = np.empty_like(grid)
    for i, t in enumerate(grid):
        before = t_exc[t_exc <= t]
        e = before.max() if before.size else t_exc[0]
        r = t_ref[(t_ref <= t) & (t_ref > e)]
        reset[i] = (
            at(t) - at(e) if not r.size else -(at(r[0]) - at(e)) + at(t) - at(r[0])
        )

    figure, (grad, kx) = plt.subplots(
        2,
        1,
        figsize=(PAGE_WIDTH, 5.0),
        sharex=True,
        layout="constrained",
        height_ratios=(1, 2),
    )
    grad.plot(times * 1e3, amplitude, color=SERIES[0], lw=0.9, label="$G_x$")
    grad.set_ylabel("mT/m")
    kx.plot(grid * 1e3, big_k, color="C7", ls="--", lw=1.0, label="unbroken integral")
    kx.plot(grid * 1e3, reset, color=SERIES[0], lw=1.0, label="reset and inverted")
    kx.plot(t_adc * 1e3, k_adc[0], ".", color=SERIES[1], ms=2.5, label="ADC samples")
    for label, ts, style in (("excitation", t_exc, ":"), ("refocusing", t_ref, "-.")):
        for n, t in enumerate(ts):
            for ax in (grad, kx):
                ax.axvline(
                    t * 1e3,
                    color="C7",
                    ls=style,
                    lw=0.8,
                    label=label if n == 0 and ax is kx else None,
                )
    kx.set_xlabel("time (ms)")
    kx.set_ylabel("$k_x$ (1/m)")
    kx.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    grad.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return figure


if __name__ == "__main__":
    fig = kspace_reset_and_unbroken()
    for name, bg in (("light", "white"), ("dark", "#121212")):
        fig.savefig(
            f"/tmp/claude-0/-home-user/02270eae-9e71-5ef7-9016-93e6bf6d9eca/scratchpad/ks_{name}.png",
            facecolor=bg,
            dpi=90,
        )


def _loop_time(module, count):
    """Seconds to add `count` trapezoid blocks and one delay-free repetition."""
    import time

    system = module.Opts()
    gx = module.make_trapezoid("x", area=1000.0, duration=1e-3, system=system)
    sequence = module.Sequence(system)
    started = time.perf_counter()
    for _ in range(count):
        sequence.add_block(gx)
    return time.perf_counter() - started


def block_loop_time():
    """Time of an add_block loop against block count, pypulseqpp and PyPulseq."""
    import numpy as np

    import pypulseqpp

    plt = _pyplot()
    counts = np.array([250, 500, 1000, 2000])
    candidates = [("pypulseqpp", pypulseqpp, "C0")]
    try:
        import pypulseq

        candidates.append(("PyPulseq", pypulseq, "C1"))
    except ImportError:
        pass
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for label, module, colour in candidates:
        times = [min(_loop_time(module, int(n)) for _ in range(2)) for n in counts]
        axis.plot(counts, times, "o-", color=colour, lw=1.5, label=label)
    axis.set_xlabel("blocks added")
    axis.set_ylabel("loop time (s)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    figure.tight_layout()
    return figure


FIGURES = {
    "axis_peaks_against_vector": axis_peaks_against_vector,
    "rotation_against_per_axis_limit": rotation_against_per_axis_limit,
    "continuity_seam": continuity_seam,
    "strength_duration": strength_duration,
    "pns_response": pns_response,
    "gradient_spectra": gradient_spectra,
    "safety_checks_on_one_sequence": safety_checks_on_one_sequence,
    "block_table_and_libraries": block_table_and_libraries,
    "gre_repetition_blocks": gre_repetition_blocks,
    "rotation_against_materialised_shapes": rotation_against_materialised_shapes,
    "bandwidth_against_sample_count": bandwidth_against_sample_count,
    "modules_in_one_train": modules_in_one_train,
    "sampling_support_order_and_angles": sampling_support_order_and_angles,
    "designed_pulse_and_gradient": designed_pulse_and_gradient,
    "kspace_reset_and_unbroken": kspace_reset_and_unbroken,
    "block_loop_time": block_loop_time,
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
