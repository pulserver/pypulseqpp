"""PROPELLER readouts: a narrow EPI blade, turned about the centre of k-space."""

from __future__ import annotations

__all__ = ["PropellerReadout2D", "PropellerStackReadout"]

import math
from typing import Any

import numpy as np

import pypulseqpp as pp

from ..._angles import calc_golden_angles, calc_uniform_angles
from .epi import _EpiReadout

_SCHEMES = ("uniform", "golden")


class _PropellerReadout(_EpiReadout):
    """Centred EPI blade with an in-plane rotation schedule.

    Each blade is ``blade_width`` phase-encode lines wide and spans the
    readout matrix. The caller scales ``gy_pre`` by ``blade_start`` and applies
    the blade's rotation about z to its encoding gradients. The slice and
    partition axes remain along z. Further keyword arguments are those of the
    EPI readouts.

    Parameters
    ----------
    system : pypulseqpp.Opts
        System limits.
    rf : RfEvent
        Pulse that opens the repetition.
    gz : GradEvent, default=None
        Selection gradient played with ``rf``.
    gz_reph : GradEvent, default=None
        Rephaser of ``gz``.
    fov : float
        Isotropic in-plane field of view (m).
    matrix : int
        Isotropic in-plane matrix size.
    blade_width : int, default=16
        Phase-encode lines per blade.
    n_blades : int, default=None
        Blades in the set. ``None`` is the smallest number that samples the
        edge of the disc at Nyquist, ``ceil(pi * matrix / (2 * blade_width))``.
    scheme : {'uniform', 'golden'}, default='uniform'
        Distribution of the generated angles. ``'uniform'`` divides half a turn
        evenly between the blades, uniform only for the complete set;
        ``'golden'`` keeps any prefix of the set near-uniform.
    angles : array_like, default=None
        In-plane blade angles (rad) instead of a generated set; excludes
        ``n_blades``.
    fov_z, matrix_z : float, int, default=None
        Slab field of view (m) and partition count; required by
        :class:`PropellerStackReadout` only.

    Attributes
    ----------
    blade_angles : numpy.ndarray
        In-plane blade angles (rad), one per blade, about z.
    blade_start : float
        Factor by which to scale ``gy_pre`` to centre the blade on k = 0; the
        same for every blade.
    blade_width : int
        Phase-encode lines in one blade.
    n_blades : int
        Blades in the set.

    Raises
    ------
    ValueError
        If a count is out of range, ``scheme`` is unknown, or ``angles`` is
        given together with ``n_blades``.
    """

    def init_module(
        self,
        system: pp.Opts,
        rf: Any,
        gz: Any = None,
        gz_reph: Any = None,
        *,
        fov: float,
        matrix: int,
        blade_width: int = 16,
        n_blades: int | None = None,
        scheme: str = "uniform",
        angles: Any = None,
        fov_z: float | None = None,
        matrix_z: int | None = None,
        **kwargs: Any,
    ) -> None:
        blade_width = int(blade_width)
        if blade_width < 1:
            raise ValueError("blade_width must be positive")
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")

        if angles is None:
            if n_blades is None:
                # Adjacent blades have to meet at the rim of the disc, where
                # the gap between them grows with radius: half a turn divided
                # by the angle a blade subtends there.
                n_blades = max(1, math.ceil(np.pi * int(matrix) / (2.0 * blade_width)))
            n_blades = int(n_blades)
            if n_blades < 1:
                raise ValueError("n_blades must be positive")
            blade_angles = (
                np.asarray(calc_uniform_angles(n_blades, span=np.pi))
                if scheme == "uniform"
                else np.asarray(calc_golden_angles(n_blades))
            )
        else:
            if n_blades is not None:
                raise ValueError(
                    "angles and n_blades are alternatives; pass the angles or ask for a count"
                )
            blade_angles = np.asarray(angles, dtype=float).ravel()
            if not len(blade_angles):
                raise ValueError("angles must hold at least one blade")

        # Every blade must SAMPLE k = 0, not merely straddle it: the centre
        # line is what makes a blade self-navigating, and it is the line the
        # echo sits on. So the lines run from -floor(w / 2) upward in steps
        # of one, the same convention an EPI blade of w lines follows, which
        # puts line floor(w / 2) exactly on the centre. Against a prewinder
        # template built at w / 2 lines, that is the fraction below.
        blade_start = -(blade_width // 2) / (blade_width / 2)

        ndim = self._ndim
        if ndim == 3:
            if fov_z is None or matrix_z is None:
                raise ValueError("a stack of propellers needs fov_z and matrix_z")
            shape: Any = (float(fov), float(fov), float(fov_z))
            counts: Any = (int(matrix), blade_width, int(matrix_z))
        else:
            shape, counts = (float(fov), float(fov)), (int(matrix), blade_width)

        super().init_module(
            system,
            rf,
            gz,
            gz_reph,
            fov=shape,
            matrix=counts,
            order=np.arange(blade_width),
            **kwargs,
        )
        self.register(blade_angles=blade_angles)
        self.blade_start = blade_start
        self.blade_width = blade_width
        self.n_blades = len(blade_angles)


class PropellerReadout2D(_PropellerReadout):
    """EPI blades turned about the centre of k-space, as a 2D PROPELLER set.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> excitation = design.SpatialSelectiveExcitation(system, 60.0, 3e-3)
    >>> blade = design.PropellerReadout2D(
    ...     system, excitation.rf, excitation.gz, excitation.gz_reph,
    ...     fov=0.22, matrix=128, blade_width=16,
    ... )
    >>> blade.etl, blade.n_blades
    (16, 13)

    >>> float(round(blade.blade_angles[1] - blade.blade_angles[0], 6))
    0.241661

    One blade, and the strip of k-space its echo train covers before the
    loop turns it:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       excitation = sequences.SpatialSelectiveExcitation(system, 15.0, 5e-3)
       readout = sequences.PropellerReadout2D(
           system, excitation.rf, excitation.gz, excitation.gz_reph,
           fov=0.22, matrix=64, blade_width=8,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _ndim = 2


class PropellerStackReadout(_PropellerReadout):
    """A stack of 2D PROPELLER blade sets, partition-encoded along z.

    The blade turns about z, which the rotation leaves unchanged, so the
    partition encode is unaffected by it: scale ``gz_pre`` per partition and
    rotate the blocks per blade.

    Examples
    --------
    >>> import pypulseqpp.sequences as design
    >>> import pypulseqpp as pp
    >>> system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=180, slew_unit="T/m/s")
    >>> slab = design.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
    >>> blade = design.PropellerStackReadout(
    ...     system, slab.rf, slab.gz, fov=0.22, matrix=128, blade_width=16,
    ...     fov_z=0.12, matrix_z=24,
    ... )
    >>> blade.n_blades, blade.gz_pre.channel
    (13, 'z')

    The partition does not move inside a blade, so nothing is blipped on z:

    >>> set(blade.gz_blips)
    {None}

    One blade of one partition, with the partition encode on z:

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pypulseqpp as pp
       from pypulseqpp import sequences

       system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       slab = sequences.SpatialSelectiveExcitation(system, 15.0, 0.12, is_slab=True)
       readout = sequences.PropellerStackReadout(
           system, slab.rf, slab.gz,
           fov=0.22, matrix=64, blade_width=8, fov_z=0.12, matrix_z=16,
       )
       readout.seq.paper_plot()
       pp.plot.plot_kspace(readout.seq, plane="xy", show_trajectory=True)
       plt.show()
    """

    _ndim = 3
