"""An RF pulse beside the magnetisation profile it produces."""

from __future__ import annotations

import numpy as np

from . import _style

#: What each ``use`` asks of a pulse: which response answers for it, the
#: profile axis label, and a hue.
_RESPONSE = {
    "excitation": ("mz_xy", "$|M_{xy}|$", 0),
    "refocusing": ("ref_eff", "refocusing efficiency", 2),
    "inversion": ("mz_z", "$M_z$", 1),
    "saturation": ("mz_z", "$M_z$", 1),
    "preparation": ("mz_z", "$M_z$", 1),
    "other": ("mz_xy", "$|M_{xy}|$", 0),
}

#: Position in ``sim_rf``'s return of each response.
_SIM_RF = {"mz_z": 0, "mz_xy": 1, "ref_eff": 3}

#: What each letter of a ``plane`` draws against: the gradient channel that
#: makes the pulse selective along it, and the axis label.
_AXES = {
    "x": (0, "x [mm]"),
    "y": (1, "y [mm]"),
    "z": (2, "z [mm]"),
    "f": (None, "off-resonance [Hz]"),
}


def _use(event) -> str:
    return str(getattr(event, "use", "") or "undefined")


def _blocks_of(seq) -> list[tuple]:
    from .._block_to_events import block_to_events

    return [
        tuple(block_to_events(seq.get_block(number)))
        for number in range(1, seq.num_blocks + 1)
    ]


def _first_rf(blocks, use: str | None = None):
    """Return the first RF event played, optionally the first tagged with ``use``."""
    for block in blocks:
        for event in block:
            if getattr(event, "type", None) != "rf":
                continue
            if use is None or _use(event) == use:
                return event
    played = "RF pulse" if use is None else f"pulse used for {use!r}"
    raise ValueError(f"nothing here plays a {played}")


def _window(source, pulse, time_range, block_range):
    """Return the blocks a profile is simulated over, as a sequence, and the pulse's use.

    A sequence gives the block of its first pulse matching ``pulse`` unless a
    range is given; a module gives all its blocks; an RF event is played alone.
    """
    from .. import Opts, make_delay
    from .._block_to_events import block_to_events
    from .._sequence import Sequence
    from ._seqeyes import blocks_for

    if getattr(source, "type", None) == "rf":
        window = Sequence(system=Opts())
        window.add_block(source)
        return window, None

    if not hasattr(source, "_native"):
        module_blocks = getattr(source, "blocks", None)
        if module_blocks is None:
            raise TypeError(
                "plot_rf(): the source is a Sequence, a sequence module or an RF event"
            )
        window = Sequence(system=getattr(source, "system", None) or Opts())
        for block in module_blocks:
            window.add_block(*block)
        return window, pulse if isinstance(pulse, str) else None

    wanted = pulse if pulse is None or isinstance(pulse, str) else _use(pulse)
    if time_range is not None or block_range is not None:
        first, last = blocks_for(source, time_range, block_range)
    else:
        first = next(
            (
                number
                for number in range(1, source.num_blocks + 1)
                if (rf := getattr(source.get_block(number), "rf", None)) is not None
                and wanted in (None, _use(rf))
            ),
            None,
        )
        if first is None:
            played = "RF pulse" if wanted is None else f"pulse used for {wanted!r}"
            raise ValueError(f"plot_rf(): this sequence plays no {played}")
        last = first
    window = Sequence(system=source.system or Opts())
    for number in range(first, last + 1):
        block = source.get_block(number)
        # A decoded block also carries its duration; only events are replayed,
        # with the duration kept as a delay.
        events = [event for event in block_to_events(block) if hasattr(event, "type")]
        window.add_block(*events, make_delay(block.block_duration))
    return window, wanted


def _gradient_at(event, time: float) -> float:
    """Return the gradient amplitude at ``time`` on its block's clock, in Hz/m."""
    delay = float(event.delay)
    if event.type == "trap":
        rise, flat, fall = (
            float(event.rise_time),
            float(event.flat_time),
            float(event.fall_time),
        )
        times = [delay, delay + rise, delay + rise + flat, delay + rise + flat + fall]
        amplitudes = [0.0, float(event.amplitude), float(event.amplitude), 0.0]
    else:
        times = delay + np.asarray(event.tt, dtype=float)
        amplitudes = np.asarray(event.waveform, dtype=float)
    return float(np.interp(time, times, amplitudes, left=0.0, right=0.0))


def _selection(blocks, pulse) -> tuple[int | None, float]:
    """Return the channel a pulse is selective along, and the gradient at its centre."""
    centre = float(pulse.delay) + float(pulse.center)
    for block in blocks:
        if not any(event is pulse for event in block):
            continue
        for event in block:
            if getattr(event, "type", None) in ("trap", "grad"):
                amplitude = _gradient_at(event, centre)
                if abs(amplitude) > 1.0:
                    return "xyz".index(event.channel), amplitude
    return None, 0.0


def _on_one_raster(window, dt: float):
    """Resample the window's RF and gradients onto one uniform raster."""
    channels = window.waveforms_and_times(True, compat=False).waveforms
    parts = [channels.gx, channels.gy, channels.gz, channels.rf]
    parts = [np.atleast_2d(np.asarray(p)) if p is not None else None for p in parts]
    stop = max(float(p[0, -1].real) for p in parts if p is not None and p.size)
    times = np.arange(0.5 * dt, stop, dt)

    def resample(part, complex_values: bool):
        empty = np.zeros(times.size, dtype=complex if complex_values else float)
        if part is None or part.size == 0:
            return empty
        clock, values = part[0].real, part[1]
        sampled = np.interp(times, clock, values.real, left=0.0, right=0.0)
        if complex_values:
            sampled = sampled + 1j * np.interp(
                times, clock, values.imag, left=0.0, right=0.0
            )
        return sampled

    gradients = [resample(part, False) for part in parts[:3]]
    return times, resample(parts[3], True), gradients


def _field(axis_values, channel, gradients, times):
    """Return the longitudinal field each grid point sees over ``times``, in Hz."""
    if channel is None:
        return np.asarray(axis_values, dtype=float)[:, None] * np.ones_like(times)
    return np.outer(1e-3 * np.asarray(axis_values, dtype=float), gradients[channel])


def _responses(b1, field, dt: float) -> dict:
    """Every response a ``use`` can ask for, from three starting states."""
    from .._sim_rf import sim_bloch

    from_z, from_x, from_y = (
        sim_bloch(b1, field, dt, initial=start)
        for start in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    return {
        "mz_z": from_z[:, 2],
        "mz_xy": from_z[:, 0] + 1j * from_z[:, 1],
        "ref_eff": (
            (from_x[:, 0] + 1j * from_x[:, 1]) + 1j * (from_y[:, 0] + 1j * from_y[:, 1])
        )
        / 2.0,
    }


def _limits(extent):
    if extent is None:
        return None
    if np.ndim(extent) == 0:
        return -float(extent), float(extent)
    return float(extent[0]), float(extent[1])


def _default_span(letter, bandwidth: float, amplitude: float, centre: float):
    if letter == "f":
        # Wider than the passband: a subpulse train comes back well outside it.
        return -4.0 * bandwidth, 4.0 * bandwidth
    if abs(amplitude) > 1.0:
        width = 1e3 * (abs(bandwidth) + abs(centre)) / abs(amplitude)
        return -width, width
    return -20.0, 20.0


def plot_rf(
    source,
    pulse=None,
    *,
    time_range=None,
    block_range=None,
    plane: str | None = None,
    kind: str | None = None,
    extent=None,
    span=None,
    samples: int = 401,
    dt: float = 8e-6,
    whole: bool = False,
    title: str | None = None,
    plot_now: bool = True,
):
    """Draw an RF pulse's ``|B1|`` envelope beside the profile it produces.

    Parameters
    ----------
    source : Sequence, sequence module or RF event
        What plays the pulse. From a sequence only the pulse's own block is
        simulated unless a range is given.
    pulse : RF event or str, optional
        Which pulse of ``source``: an event, or the ``use`` it is tagged with
        (``"refocusing"`` finds a spin echo's refocusing pulse). The first
        pulse by default.
    time_range, block_range : sequence, optional
        Blocks of a sequence to simulate over, where the profile needs more
        than the pulse's own block.
    plane : str, optional
        One of ``x``, ``y``, ``z`` and ``f`` (off-resonance) for a profile, or
        two for a plane drawn as ``|Mxy|`` and ``Mz`` heatmaps. By default the
        axis the pulse is selective along, or ``f`` without a gradient.
    kind : {"excitation", "refocusing", "inversion", "saturation"}, optional
        Which response a profile draws; the pulse's ``use`` by default.
    extent, span : float or pair of float, optional
        Limits of the first and second axis: a half-width about zero or a
        ``(low, high)`` pair, in mm for a position and Hz for off-resonance.
    samples : int, default 401
        Points per axis; a plane caps at 91 a side.
    dt : float, default 8e-6
        Integration step in seconds, wherever the window is integrated rather
        than the pulse alone.
    whole : bool, default False
        Integrate everything in the window -- every pulse, gradient and the
        precession between them -- rather than the pulse alone. A plane is
        always integrated this way.
    title : str, optional
    plot_now : bool, default True
        Show the figure before returning.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    A single pulse against one axis is :func:`pypulseqpp.sim_rf`, which
    ignores the gradient: a position axis is its frequency axis over the
    selection gradient at the pulse centre. Everything else integrates
    :func:`pypulseqpp.sim_bloch` over the window without relaxation.
    """
    from matplotlib import pyplot as plt

    from .. import calc_rf_bandwidth, sim_rf

    window, wanted = _window(source, pulse, time_range, block_range)
    blocks = _blocks_of(window)
    pulse = _first_rf(blocks, wanted)
    channel, amplitude = _selection(blocks, pulse)
    if plane is None:
        plane = "xyz"[channel] if channel is not None else "f"
    if len(plane) not in (1, 2) or any(letter not in _AXES for letter in plane):
        raise ValueError(
            f"plot_rf(): plane must be one or two of {', '.join(_AXES)}, not {plane!r}"
        )
    if len(plane) == 1 and plane != "f" and _AXES[plane][0] != channel and not whole:
        raise ValueError(
            f"plot_rf(): the pulse is not selective along {plane!r}; pass whole=True "
            "to integrate the window along it"
        )

    response, label, hue = _RESPONSE.get(kind or _use(pulse), _RESPONSE["other"])
    bandwidth = float(calc_rf_bandwidth(pulse))
    centre = float(getattr(pulse, "freq_offset", 0.0))
    gamma = float(window.system.gamma)

    limits, grids = [], []
    for letter, asked in zip(plane, (extent, span), strict=False):
        low, high = _limits(asked) or _default_span(
            letter, bandwidth, amplitude, centre
        )
        limits.append((low, high))
        grids.append(
            np.linspace(low, high, samples if len(plane) == 1 else min(samples, 91))
        )

    responses, shape = None, None
    if len(plane) == 1 and not whole:
        simulated = sim_rf(pulse)
        values = np.asarray(simulated[_SIM_RF[response]])
        axis_values = np.asarray(simulated[2], dtype=float)
        if plane != "f":
            axis_values = 1e3 * axis_values / amplitude
        clock = 1e3 * np.asarray(pulse.t, dtype=float)
        envelope = 1e6 * np.abs(np.asarray(pulse.signal)) / gamma
    else:
        times, b1, gradients = _on_one_raster(window, dt)
        clock = 1e3 * times
        envelope = 1e6 * np.abs(b1) / gamma
        if len(plane) == 1:
            axis_values = grids[0]
            field = _field(axis_values, _AXES[plane][0], gradients, times)
            values = _responses(b1, field, dt)[response]
        else:
            first, second = np.meshgrid(grids[0], grids[1], indexing="ij")
            field = _field(
                first.ravel(), _AXES[plane[0]][0], gradients, times
            ) + _field(second.ravel(), _AXES[plane[1]][0], gradients, times)
            responses = _responses(b1, field, dt)
            shape = first.shape

    if len(plane) == 1:
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(8.4, 2.9), gridspec_kw={"width_ratios": (1.0, 1.35)}
        )
        _envelope_panel(left, clock, envelope)
        drawn = np.abs(values) if np.iscomplexobj(values) else np.asarray(values)
        order = np.argsort(axis_values)
        right.axhline(0.0, color=_style.FAINT, lw=0.8)
        right.plot(axis_values[order], drawn[order], color=_style.SERIES[hue], lw=1.8)
        right.set_xlabel(_AXES[plane][1])
        right.set_ylabel(label)
        right.set_xlim(*limits[0])
        _style.axis_style(right, "profile")
    else:
        figure, (left, mxy, mz) = plt.subplots(
            1, 3, figsize=(11.2, 3.1), gridspec_kw={"width_ratios": (1.0, 1.1, 1.1)}
        )
        _envelope_panel(left, clock, envelope)
        panels = (
            (mxy, np.abs(responses["mz_xy"]), r"$|M_{xy}|$", _style.MAGNITUDE, 0.0),
            (mz, np.real(responses["mz_z"]), "$M_z$", _style.SIGNED, -1.0),
        )
        for axis, grid, name, cmap, low in panels:
            image = axis.imshow(
                grid.reshape(shape).T,
                origin="lower",
                aspect="auto" if "f" in plane else "equal",
                cmap=cmap,
                vmin=low,
                vmax=1.0,
                interpolation="nearest",
                extent=(*limits[0], *limits[1]),
            )
            axis.set_xlabel(_AXES[plane[0]][1])
            axis.set_ylabel(_AXES[plane[1]][1])
            bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
            bar.outline.set_visible(False)
            bar.ax.tick_params(colors=_style.MUTED, labelsize=8, length=0)
            _style.image_style(axis, name)

    _style.figure_title(figure, title)
    figure.tight_layout(rect=(0, 0, 1, 0.92 if title else 1.0))
    if plot_now:
        plt.show()
    return figure


def _envelope_panel(axis, clock, envelope) -> None:
    axis.plot(clock, envelope, color=_style.SERIES[0], lw=1.6)
    axis.fill_between(clock, envelope, color=_style.SERIES[0], alpha=0.12, lw=0)
    axis.set_xlabel("time [ms]")
    axis.set_ylabel(r"$|B_1|$ [$\mu$T]")
    axis.set_xlim(clock[0], clock[-1])
    _style.axis_style(axis, "envelope")
