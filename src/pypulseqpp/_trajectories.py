"""K-space paths stated as geometry, for :func:`pypulseqpp.traj_to_grad`.

Each returns one base interleave as ``(n, 2)`` coordinates in cycles/m: where
the path goes, not when. The samples are a polyline dense enough to describe
the curve, and :func:`pypulseqpp.traj_to_grad` re-parameterises it against the
gradient and slew limits to find the time. How many rotated copies a scan
plays, and at which angles, is :func:`pypulseqpp.calc_golden_angles` and
:func:`pypulseqpp.make_rotation`'s business rather than the path's.
"""

from __future__ import annotations

__all__ = [
    "calc_radial_trajectory",
    "calc_rosette_trajectory",
    "calc_spiral_trajectory",
]

import math

import numpy as np

SPIRAL_DENSITIES = ("constant", "variable", "dual")


def _as_scalar(value, name, cast=float):
    array = np.asarray(value)
    if array.ndim == 0:
        return cast(array)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    first = cast(array.flat[0])
    if not np.all(array == array.flat[0]):
        raise ValueError(
            f"{name} must be isotropic for a rotationally symmetric trajectory"
        )
    return first


def _validate_common(fov, matrix, oversamp, bandwidth_hz_px):
    fov_m = _as_scalar(fov, "fov")
    n = _as_scalar(matrix, "matrix", int)
    if fov_m <= 0 or n < 2:
        raise ValueError("fov must be positive and matrix must be >= 2")
    if oversamp < 1:
        raise ValueError("oversamp must be >= 1")
    if bandwidth_hz_px <= 0:
        raise ValueError("bandwidth_hz_px must be positive")
    return fov_m, n


def _cumtrapz(values, x):
    out = np.zeros_like(values, dtype=float)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    return out


def calc_radial_trajectory(fov, matrix, *, num_points=None):
    """Return one full radial spoke through the centre of k-space.

    Parameters
    ----------
    fov : float
        Field of view, in m. Isotropic: a sequence of equal values is
        accepted, unequal ones are not.
    matrix : int
        Matrix size, which with ``fov`` sets the reach ``matrix / (2 fov)``.
    num_points : int, optional
        Samples along the spoke; ``matrix`` by default.

    Returns
    -------
    numpy.ndarray
        ``(num_points, 2)``, in cycles/m, from ``-kmax`` to ``+kmax`` along x.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> spoke = pp.calc_radial_trajectory(0.256, 128)
    >>> spoke.shape, float(spoke[-1, 0])
    ((128, 2), 250.0)
    """
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    count = int(num_points or n)
    if count < 2:
        raise ValueError("num_points must be >= 2")
    kmax = n / (2.0 * fov_m)
    return np.column_stack((np.linspace(-kmax, kmax, count), np.zeros(count)))


def calc_spiral_trajectory(
    fov,
    matrix,
    design_interleaves,
    *,
    density="constant",
    inner_design_interleaves=None,
    outer_design_interleaves=None,
    variable_density_power=2.0,
    transition_radius=0.5,
    transition_speed=12.0,
    num_points=1024,
):
    """Return one spiral-out interleave, from the centre to ``kmax``.

    The pitch between neighbouring turns is what sets the field of view a
    set of interleaves supports: ``design_interleaves`` rotated copies of this
    path sample the disc at the Nyquist rate for ``fov``. It does not say how
    many copies the caller acquires.

    Parameters
    ----------
    fov : float
        Field of view, in m. Isotropic.
    matrix : int
        Matrix size, which with ``fov`` sets the reach ``matrix / (2 fov)``.
    design_interleaves : int
        Interleave count the pitch is designed for. One is a single-shot
        spiral of ``matrix / 2`` turns.
    density : {"constant", "variable", "dual"}, optional
        How the local pitch changes with radius: not at all, as
        ``radius ** variable_density_power``, or between two plateaus joined
        by a logistic transition.
    inner_design_interleaves, outer_design_interleaves : float, optional
        Local interleave count at the centre and at the edge. The inner one
        defaults to ``design_interleaves``; the outer one to twice the inner
        for a variable-density spiral, and is required for a dual-density
        one. A larger count is a coarser pitch, so a variable-density
        spiral undersamples the edge.
    variable_density_power : float, optional
        Exponent of the variable-density ramp.
    transition_radius, transition_speed : float, optional
        Where between the centre (0) and the edge (1) a dual-density spiral
        changes pitch, and how abruptly.
    num_points : int, optional
        Samples along the polyline. Sets how finely the curve is described,
        not how many ADC samples the readout takes.

    Returns
    -------
    numpy.ndarray
        ``(num_points, 2)``, in cycles/m, starting at the origin.

    Raises
    ------
    ValueError
        If ``density`` is not one of the three, a dual-density spiral has no
        outer interleave count, or any count or shape parameter is out of
        range.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> arm = pp.calc_spiral_trajectory(0.256, 128, 8)
    >>> bool(np.allclose(arm[0], 0.0)), round(float(np.hypot(*arm[-1])), 6)
    (True, 250.0)
    """
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    design_interleaves = int(design_interleaves)
    num_points = int(num_points)
    if design_interleaves < 1 or num_points < 4:
        raise ValueError("design_interleaves must be >= 1 and num_points must be >= 4")
    if density not in SPIRAL_DENSITIES:
        raise ValueError(f"density must be one of {SPIRAL_DENSITIES}, got {density!r}")

    inner = float(
        design_interleaves
        if inner_design_interleaves is None
        else inner_design_interleaves
    )
    if outer_design_interleaves is None:
        if density == "variable":
            outer = 2.0 * inner
        elif density == "dual":
            raise ValueError(
                "outer_design_interleaves is required for dual-density spirals"
            )
        else:
            outer = inner
    else:
        outer = float(outer_design_interleaves)
    if inner <= 0 or outer <= 0:
        raise ValueError(
            "inner_design_interleaves and outer_design_interleaves must be positive"
        )

    radius = np.linspace(0.0, 1.0, num_points)
    if density == "constant":
        local_interleaves = np.full_like(radius, inner)
    elif density == "variable":
        if variable_density_power <= 0:
            raise ValueError("variable_density_power must be positive")
        local_interleaves = inner + (outer - inner) * radius ** float(
            variable_density_power
        )
    else:
        if not 0.0 < transition_radius < 1.0 or transition_speed <= 0:
            raise ValueError(
                "transition_radius must be in (0, 1) and transition_speed must be positive"
            )
        blend = 1.0 / (
            1.0 + np.exp(-float(transition_speed) * (radius - float(transition_radius)))
        )
        blend = (blend - blend[0]) / (blend[-1] - blend[0])
        local_interleaves = inner + (outer - inner) * blend

    # For a square matrix, dphi/dr = pi*N/n_interleaves gives N/2 turns
    # for a single-shot constant-density spiral and the corresponding local
    # pitch for multi-shot / variable-density paths.
    phi = _cumtrapz(np.pi * n / local_interleaves, radius)
    kmax = n / (2.0 * fov_m)
    rho = kmax * radius
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi)))


def calc_rosette_trajectory(
    fov,
    matrix,
    *,
    petals=5,
    angular_frequency_ratio=3.0 / 5.0,
    num_points=2049,
):
    """Return one rosette interleave, petals through the centre of k-space.

    The path is ``rho(u) = kmax sin(pi petals u)`` at angle
    ``theta(u) = pi petals angular_frequency_ratio u`` for ``u`` in
    ``[0, 1]``, so it crosses the centre ``petals + 1`` times.

    Parameters
    ----------
    fov : float
        Field of view, in m. Isotropic.
    matrix : int
        Matrix size, which with ``fov`` sets the reach ``matrix / (2 fov)``.
    petals : int, optional
        Centre-to-centre lobes within this one interleave. More petals cross
        the centre more often and lengthen the readout.
    angular_frequency_ratio : float, optional
        Angular over radial frequency. Below one the petals are open; one is
        the circular limit; above one they wind more tightly.
    num_points : int, optional
        Samples along the polyline. Sets how finely the curve is described,
        not how many ADC samples the readout takes.

    Returns
    -------
    numpy.ndarray
        ``(num_points, 2)``, in cycles/m, starting and ending at the origin.

    Raises
    ------
    ValueError
        If ``petals`` or ``angular_frequency_ratio`` is not positive, or
        ``num_points`` is below five.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> path = pp.calc_rosette_trajectory(0.256, 128, petals=4)
    >>> bool(np.allclose(path[[0, -1]], 0.0, atol=1e-9))
    True
    """
    fov_m, n = _validate_common(fov, matrix, 1.0, 1.0)
    petals, num_points = int(petals), int(num_points)
    angular_frequency_ratio = float(angular_frequency_ratio)
    if (
        petals < 1
        or not math.isfinite(angular_frequency_ratio)
        or angular_frequency_ratio <= 0
        or num_points < 5
    ):
        raise ValueError(
            "petals and angular_frequency_ratio must be positive and num_points must be >= 5"
        )
    u = np.linspace(0.0, 1.0, num_points)
    rho = (n / (2.0 * fov_m)) * np.sin(np.pi * petals * u)
    theta = np.pi * petals * angular_frequency_ratio * u
    return np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
