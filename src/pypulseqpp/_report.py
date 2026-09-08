"""What a sequence is, read back as a report on the scan it plays.

Everything here is already known: how many blocks there are and what they
carry, how long the scan lasts, where the gradients go and where the digitiser
samples them. The report is where those become a description of an experiment
-- the echo time, the repetition time, the flip angles, the resolution the
encoding reaches, how often k-space is revisited and whether it is sampled on
a grid.

Two of the answers are worked out rather than looked up, and both are compiled
passes. A flip angle is the integral of a pulse's envelope, so it belongs to
the RF library row rather than to a block: a pulse played ten thousand times is
integrated once. What the encoding covers is found by binning the sampled
trajectory onto a lattice fine enough to separate neighbouring positions,
which is a pass over every sample the scan takes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pypulseq import eps
from pypulseq.convert import convert as _in_units

from . import _ext as _cxx
from ._check_timing import describe

__all__ = ["report_data", "report_text"]

#: How finely k-space is divided when asking which samples land on the same
#: position: the extent of the trajectory over this many bins.
_K_BINS = 4e6

#: Added to every interval of the shared time base before a slew is divided
#: by it, so that two corners a fraction of a nanosecond apart do not report
#: an unbounded rate.
_STEADY = 1e-10


def report_data(seq) -> dict[str, Any]:
    """Return what the sequence is, as named statistics.

    Parameters
    ----------
    seq : Sequence
        The sequence to describe.

    Returns
    -------
    dict
        ``num_blocks``, ``event_count`` and ``libraries``; ``duration``,
        ``TE`` and ``TR`` in seconds; ``flip_angles_deg``;
        ``unique_k_positions``; ``max_gradient`` and ``max_slew_rate``, each
        per axis and as a vector magnitude, in the file's units and in the
        scanner's; and ``timing_ok`` with ``timing_error_report``.

        A sequence whose encoding visits more than one position also carries
        ``dimensions``, ``spatial_resolution_mm``, ``repetitions`` and
        ``is_cartesian``.
    """
    native = seq._native
    flip_angles_deg = np.unique(_cxx.flip_angles(native))
    duration, num_blocks, event_count = seq.duration()

    gw_data = seq.waveforms()
    sampled = seq._kspace(samples_only=True)
    k_traj_adc = sampled["k_traj_adc"]
    t_adc = sampled["t_adc"]
    t_excitation = sampled["t_excitation"]

    # An ADC window before the first pulse digitises whatever is there --
    # a noise scan -- rather than a signal the sequence excited.
    if t_excitation.size:
        acquired = t_adc > t_excitation[0]
        k_traj_adc = k_traj_adc[:, acquired]
        t_adc = t_adc[acquired]

    t_echo = _echo_time(k_traj_adc, t_adc)
    TE, TR = _te_and_tr(t_excitation, t_echo, duration)

    coverage = _coverage(k_traj_adc)
    ga, gs, ga_abs, gs_abs = _gradient_peaks(gw_data)
    timing_ok, timing_error_report = seq.check_timing()

    gamma = getattr(seq.system, "gamma", None) or 42576000.0
    data: dict[str, Any] = {
        "num_blocks": num_blocks,
        "event_count": {
            "rf": int(event_count[1]),
            "gx": int(event_count[2]),
            "gy": int(event_count[3]),
            "gz": int(event_count[4]),
            "adc": int(event_count[5]),
            "delay": int(event_count[0]),
            "extensions": int(event_count[6]),
        },
        "libraries": {
            "RF": native.num_rf(),
            "Gradient": native.num_gradients(),
            "Shape": native.num_shapes(),
            "ADC": native.num_adc(),
            "Extension": native.num_extensions(),
            "Trigger": native.num_triggers(),
            "Label set": native.num_label_set(),
            "Label inc": native.num_label_inc(),
            "RF shim": native.num_rf_shims(),
            "Rotation": native.num_rotations(),
            "Soft delay": native.num_soft_delays(),
        },
        "duration": duration,
        "TE": TE,
        "TR": TR,
        "flip_angles_deg": list(flip_angles_deg),
        "unique_k_positions": coverage["unique_positions"],
        "max_gradient": {
            "per_channel_Hz_m": list(ga),
            "per_channel_mT_m": list(
                _in_units(from_value=ga, from_unit="Hz/m", to_unit="mT/m", gamma=gamma)
            ),
            "absolute_Hz_m": ga_abs,
            "absolute_mT_m": _in_units(
                from_value=ga_abs, from_unit="Hz/m", to_unit="mT/m", gamma=gamma
            ),
        },
        "max_slew_rate": {
            "per_channel_Hz_m_s": list(gs),
            "per_channel_T_m_s": list(
                _in_units(
                    from_value=gs, from_unit="Hz/m/s", to_unit="T/m/s", gamma=gamma
                )
            ),
            "absolute_Hz_m_s": gs_abs,
            "absolute_T_m_s": _in_units(
                from_value=gs_abs, from_unit="Hz/m/s", to_unit="T/m/s", gamma=gamma
            ),
        },
        "timing_ok": timing_ok,
        "timing_error_report": timing_error_report,
    }

    if np.any(coverage["unique_positions"] > 1):
        k_extent = coverage["k_extent"]
        data["dimensions"] = len(k_extent)
        data["spatial_resolution_mm"] = list(0.5 / k_extent * 1e3)
        data["repetitions"] = {
            "median": coverage["repeats_median"],
            "min": coverage["repeats_min"],
            "max": coverage["repeats_max"],
        }
        data["is_cartesian"] = coverage["is_cartesian"]

    return data


def _echo_time(k_traj_adc, t_adc):
    """Return when the sequence passes closest to the centre of k-space.

    The sample nearest the origin, refined by projecting the way back to the
    origin onto the step to its neighbour: the echo falls between two samples
    unless one happens to land on it.
    """
    if t_adc.size == 0:
        return np.nan

    k_abs_adc = np.sqrt(np.sum(np.square(k_traj_adc), axis=0))
    index_echo = int(np.argmin(k_abs_adc))
    t_echo = t_adc[index_echo]
    if k_abs_adc[index_echo] <= eps:
        return t_echo

    neighbours = []
    if index_echo > 0:
        neighbours.append(index_echo - 1)
    if index_echo < len(k_abs_adc) - 1:
        neighbours.append(index_echo + 1)

    to_origin = -k_traj_adc[:, index_echo]
    for neighbour in neighbours:
        step = k_traj_adc[:, neighbour] - k_traj_adc[:, index_echo]
        along = np.matmul(to_origin, step) / np.square(np.linalg.norm(step))
        if along > 0:
            t_echo = t_adc[index_echo] * (1 - along) + t_adc[neighbour] * along
    return t_echo


def _te_and_tr(t_excitation, t_echo, duration):
    """Return the echo time and the repetition time, in seconds.

    TE is measured from the excitation the echo belongs to; TR from that
    excitation to the next one, or between the last two if the echo is in the
    last repetition. A sequence that never passes through the centre of
    k-space has no echo to measure to.
    """
    before = (
        t_excitation[t_excitation < t_echo]
        if t_excitation.size and np.isfinite(t_echo)
        else t_excitation[:0]
    )
    TE = t_echo - before[-1] if before.size else np.nan

    if t_excitation.size < 2:
        return TE, duration
    after = t_excitation[t_excitation > t_echo] if np.isfinite(t_echo) else before[:0]
    if after.size:
        return TE, after[0] - before[-1]
    if before.size > 1:
        return TE, before[-1] - before[-2]
    return TE, t_excitation[-1] - t_excitation[-2]


def _coverage(k_traj_adc) -> dict[str, Any]:
    """Return what the sampled trajectory covers.

    An axis the trajectory never moves along is not a dimension of the
    encoding, so it is dropped before the positions are counted.
    """
    idle = {
        "unique_positions": np.ones(1),
        "k_extent": np.zeros(0),
        "repeats_median": 0.0,
        "repeats_min": 0.0,
        "repeats_max": 0.0,
        "is_cartesian": False,
    }
    if k_traj_adc.shape[1] == 0:
        return idle

    k_extent = np.max(np.abs(k_traj_adc), axis=1)
    k_scale = np.max(k_extent)
    if k_scale == 0:
        return idle

    k_threshold = k_scale / _K_BINS
    moving = k_extent >= k_threshold
    if not np.all(moving):
        k_traj_adc = k_traj_adc[moving]
        k_extent = k_extent[moving]

    found = _cxx.kspace_coverage(np.ascontiguousarray(k_traj_adc), k_threshold)
    found["k_extent"] = k_extent
    return found


def _gradient_peaks(gw_data):
    """Return the strongest gradient and slew, per axis and as a magnitude.

    A per-axis peak is read off the waveform the axis plays. A magnitude is
    what a rotated sequence can ask of one amplifier, so the axes are first
    put on the time base they share.
    """
    axes = len(gw_data)
    ga = np.zeros(axes)
    gs = np.zeros(axes)

    silent = all(channel.size == 0 for channel in gw_data)
    common_time = np.unique(np.concatenate(gw_data, axis=1)[0])
    gw_ct = np.zeros(0) if silent else np.zeros((axes, len(common_time)))
    gs_ct = np.zeros(0) if silent else np.zeros((axes, len(common_time) - 1))

    for axis, channel in enumerate(gw_data):
        if channel.shape[1] == 0:
            continue
        slew = np.diff(channel[1]) / np.diff(channel[0])
        gw_ct[axis] = np.interp(
            x=common_time, xp=channel[0], fp=channel[1], left=0, right=0
        )
        gs_ct[axis] = np.diff(gw_ct[axis]) / (np.diff(common_time) + _STEADY)
        ga[axis] = np.max(np.abs(channel[1:]))
        gs[axis] = np.max(np.abs(slew))

    ga_abs = np.max(np.sqrt(np.sum(np.square(gw_ct), axis=0)))
    gs_abs = np.max(np.sqrt(np.sum(np.square(gs_ct), axis=0)))
    return ga, gs, ga_abs, gs_abs


def report_text(data: dict[str, Any]) -> str:
    """Return :func:`report_data`'s answer as the report a person reads."""
    event_count = data["event_count"]
    flip_angles_deg = data["flip_angles_deg"]
    unique_k_positions = data["unique_k_positions"]
    ga = data["max_gradient"]["per_channel_Hz_m"]
    ga_converted = data["max_gradient"]["per_channel_mT_m"]
    gs = data["max_slew_rate"]["per_channel_Hz_m_s"]
    gs_converted = data["max_slew_rate"]["per_channel_T_m_s"]

    report = (
        f"Number of blocks: {data['num_blocks']}\n"
        f"Number of events:\n"
        f"RF: {event_count['rf']:6.0f}\n"
        f"Gx: {event_count['gx']:6.0f}\n"
        f"Gy: {event_count['gy']:6.0f}\n"
        f"Gz: {event_count['gz']:6.0f}\n"
        f"ADC: {event_count['adc']:6.0f}\n"
        f"Delay: {event_count['delay']:6.0f}\n"
        f"Extensions: {event_count['extensions']:6.0f}\n"
        f"Sequence duration: {data['duration']:.6f} s\n"
        f"TE: {data['TE']:.6f} s\n"
        f"TR: {data['TR']:.6f} s\n"
    )
    report += (
        "Flip angle: " + ("{:.02f} " * len(flip_angles_deg)).format(*flip_angles_deg)
    ) + "deg\n"
    report += (
        "Unique k-space positions (aka cols, rows, etc.): "
        + ("{:.0f} " * len(unique_k_positions)).format(*unique_k_positions)
        + "\n"
    )

    if "dimensions" in data:
        resolution = data["spatial_resolution_mm"]
        repetitions = data["repetitions"]
        report += f"Dimensions: {data['dimensions']}\n"
        report += ("Spatial resolution: {:.02f} mm\n" * len(resolution)).format(
            *resolution
        )
        report += (
            f"Repetitions/slices/contrasts: {repetitions['median']}; "
            f"range: [{repetitions['min']}, {repetitions['max']}]\n"
        )
        report += (
            "Cartesian encoding trajectory detected\n"
            if data["is_cartesian"]
            else "Non-cartesian/irregular encoding trajectory detected "
            "(eg: EPI, spiral, radial, etc.)\n"
        )

    report += (
        "Max gradient: "
        + ("{:.0f} " * len(ga)).format(*ga)
        + "Hz/m == "
        + ("{:.02f} " * len(ga_converted)).format(*ga_converted)
        + "mT/m\n"
    )
    report += (
        "Max slew rate: "
        + ("{:.0f} " * len(gs)).format(*gs)
        + "Hz/m/s == "
        + ("{:.02f} " * len(gs_converted)).format(*gs_converted)
        + "T/m/s\n"
    )
    report += (
        f"Max absolute gradient: {data['max_gradient']['absolute_Hz_m']:.0f} Hz/m == "
        f"{data['max_gradient']['absolute_mT_m']:.2f} mT/m\n"
    )
    report += (
        f"Max absolute slew rate: {data['max_slew_rate']['absolute_Hz_m_s']:g} "
        f"Hz/m/s == {data['max_slew_rate']['absolute_T_m_s']:.2f} T/m/s"
    )

    report += "\nEvent library use:\n"
    for name, held in data["libraries"].items():
        report += f"{name}: {held:6.0f}\n"

    found = data["timing_error_report"]
    if data["timing_ok"]:
        return report + "\nEvent timing check passed successfully\n"

    report += f"\nEvent timing check failed with {len(found)} errors in total. \n"
    report += "Details of the first up to 20 timing errors:"
    shown = min(20, len(found))
    for problem in found[:shown]:
        report += f"\n{describe(problem)}"
    if len(found) > shown:
        report += "\n..."
    return report
