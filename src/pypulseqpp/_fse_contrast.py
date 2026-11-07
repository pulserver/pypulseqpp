"""Contrast-equivalent echo times for ideal CPMG refocusing trains."""

from functools import lru_cache

import numpy as np

from ._busse import _rotate, _shift, make_busse_schedule


def calc_fse_effective_te(
    flip_angles,
    echo_spacing: float,
    *,
    first_echo_time: float | None = None,
    reference_t1: float = 1.0,
    reference_t2: float = 0.1,
) -> np.ndarray:
    """Calculate contrast-equivalent TE by relaxation-normalized CPMG EPG.

    Parameters
    ----------
    flip_angles : array_like
        Refocusing flip angles (rad), shape ``(..., echoes)``. Startup pulses
        are included. Angles must lie in ``[0, pi]``.
    echo_spacing : float
        Time between successive echoes (s), positive.
    first_echo_time : float | None, default=None
        Excitation-to-first-echo time (s). ``None`` uses ``echo_spacing``.
        Each refocusing pulse bisects the preceding echo interval.
    reference_t1 : float, default=1.0
        Reference longitudinal relaxation time (s), positive and finite.
    reference_t2 : float, default=0.1
        Reference transverse relaxation time (s), positive and finite.

    Returns
    -------
    numpy.ndarray
        Contrast-equivalent echo times (s), shape matching ``flip_angles``.
        A zero-coherence echo is undefined and returned as ``nan``.

    Raises
    ------
    ValueError
        If angles, timing or reference relaxation times are invalid.

    Notes
    -----
    For a fully relaxed, ideal 90-degree excitation and instantaneous CPMG
    refocusing, EPG computes signal with reference relaxation and with no
    relaxation. The result is ``-reference_t2 * log(abs(S / S_no_relax))``
    [1]_. No RF-pulse duration, slice profile, off-resonance, transmit-field
    variation or inter-shot steady state is modeled. This is a reference
    contrast annotation, not a tissue-independent prediction. At 180-degree
    refocusing it equals physical echo time. No Torch dependency is required.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> np.allclose(pp.calc_fse_effective_te([np.pi] * 3, 0.004), [0.004, 0.008, 0.012])
    True

    References
    ----------
    .. [1] Busse RF et al. Fast spin echo sequences with very long echo
       trains: design of variable refocusing flip angle schedules and
       generation of clinical T2 contrast. Magn Reson Med. 2006;55:1030-1037.
       https://doi.org/10.1002/mrm.20863
    """
    angles = np.asarray(flip_angles, dtype=float)
    if (
        angles.ndim < 1
        or angles.shape[-1] < 1
        or not np.all(np.isfinite(angles))
        or np.any((angles < 0) | (angles > np.pi + 1e-8))
    ):
        raise ValueError(
            "flip_angles must be finite radians in [0, pi] with echoes on the last axis"
        )
    first = echo_spacing if first_echo_time is None else first_echo_time
    times = np.asarray([echo_spacing, first, reference_t1, reference_t2], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times <= 0):
        raise ValueError(
            "echo spacing, first echo time and reference T1/T2 must be positive and finite"
        )
    shape = angles.shape[:-1]
    count = angles.shape[-1]
    state = np.zeros((3, 2, *shape, count + 1), dtype=complex)
    state[:2, ..., 0] = 1.0
    result = np.empty_like(angles)
    for echo in range(count):
        half = (first if echo == 0 else echo_spacing) / 2
        e1 = np.exp(-half / reference_t1)
        e2 = np.exp(-half / reference_t2)
        relaxation1 = np.array([e1, 1.0]).reshape(2, *([1] * (len(shape) + 1)))
        relaxation2 = np.array([e2, 1.0]).reshape(2, *([1] * (len(shape) + 1)))
        for after_rf in (False, True):
            if after_rf:
                state = _rotate(state, angles[..., echo, None])
            state[:2] *= relaxation2
            state[2] *= relaxation1
            state[2, 0, ..., 0] += 1 - e1
            _shift(state)
        signal = abs(state[0, ..., 0])
        with np.errstate(invalid="ignore", divide="ignore"):
            equivalent = -reference_t2 * np.log(signal[0] / signal[1])
        result[..., echo] = np.where(signal[1] > 1e-12, equivalent, np.nan)
    return result


@lru_cache(maxsize=512)
def _busse_centre_te(length, startup, centre, controls, decay, esp, first):
    angles = make_busse_schedule(
        length,
        minimum_flip_angle=np.deg2rad(controls[0]),
        centre_flip_angle=np.deg2rad(controls[1]),
        maximum_flip_angle=np.deg2rad(controls[2]),
        centre_echo=startup + centre,
        startup_echoes=startup,
        initial_decay=decay,
    )
    return float(
        calc_fse_effective_te(
            angles[: startup + centre + 1], esp, first_echo_time=first
        )[-1]
    )


def _select_busse_centre(
    etl,
    peripheral_etl,
    startup,
    controls,
    peripheral_controls,
    decay,
    esp,
    first,
    target,
):
    """Locate the nearest feasible centre by its reference contrast-equivalent TE.

    Rebuild the prescribed train at each candidate centre; the centre-angle
    constraint therefore follows the k-space centre. Bisection uses the
    increasing centre-time relation of this fixed-control Busse family.
    Infeasible shooting candidates are excluded, not used as timing estimates.
    """
    if not np.isfinite(target) or target <= 0:
        raise ValueError("busse_effective_te must be positive and finite")
    low, high = max(0, 6 - startup), min(etl, peripheral_etl) - 2
    candidates = {}
    while low <= high:
        midpoint = (low + high) // 2
        # A failed local inverse solve must not discard the whole bracket.
        choices = sorted(range(low, high + 1), key=lambda c: abs(c - midpoint))
        for centre in choices:
            try:
                effective = _busse_centre_te(
                    etl + startup, startup, centre, tuple(controls), decay, esp, first
                )
                if etl != peripheral_etl or not np.array_equal(
                    controls, peripheral_controls
                ):
                    _busse_centre_te(
                        peripheral_etl + startup,
                        startup,
                        centre,
                        tuple(peripheral_controls),
                        decay,
                        esp,
                        first,
                    )
            except ValueError:
                continue
            if np.isfinite(effective):
                break
        else:
            break
        candidates[centre] = effective
        if effective < target:
            low = centre + 1
        else:
            high = centre - 1
    if not candidates:
        raise ValueError(
            "No feasible Busse centre echo for the requested effective TE and endpoint controls"
        )
    return min(candidates, key=lambda c: (abs(candidates[c] - target), c))
