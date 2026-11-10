"""Contrast-equivalent echo times for ideal CPMG refocusing trains."""

import numpy as np

from ._busse import _trains


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
    ImportError
        If blochsim is not installed.

    Notes
    -----
    For a fully relaxed, ideal 90-degree excitation and instantaneous CPMG
    refocusing, EPG computes signal with reference relaxation and with no
    relaxation. The result is ``-reference_t2 * log(abs(S / S_no_relax))``
    [1]_. No RF-pulse duration, slice profile, off-resonance, transmit-field
    variation or inter-shot steady state is modeled. This is a reference
    contrast annotation, not a tissue-independent prediction. At 180-degree
    refocusing it equals physical echo time. The echoes are computed by
    ``blochsim.trains.effective_te``, which the ``design`` extra installs;
    PyTorch is not imported.

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
    return 1e-3 * _trains().effective_te(
        np.rad2deg(angles),
        1e3 * echo_spacing,
        first_ESP=1e3 * first,
        T1=1e3 * reference_t1,
        T2=1e3 * reference_t2,
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

    ``blochsim.trains.busse_centre_echo`` rebuilds the prescribed train at
    every candidate centre in one batch; the centre-angle constraint therefore
    follows the k-space centre. A centre is feasible when both the central and
    the peripheral train can be designed for it. Of two centres equally near,
    the earlier is returned.
    """
    if not np.isfinite(target) or target <= 0:
        raise ValueError("busse_effective_te must be positive and finite")
    prescriptions = np.asarray([controls, peripheral_controls], dtype=float)
    try:
        centre, _ = _trains().busse_centre_echo(
            1e3 * target,
            [etl + startup, peripheral_etl + startup],
            1e3 * esp,
            first_ESP=1e3 * first,
            min_flip=prescriptions[:, 0],
            centre_flip=prescriptions[:, 1],
            max_flip=prescriptions[:, 2],
            startup_echoes=startup,
            initial_decay=decay,
        )
    except ValueError as error:
        raise ValueError(
            "No feasible Busse centre echo for the requested effective TE and endpoint controls"
        ) from error
    return centre - startup
