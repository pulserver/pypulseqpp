"""Isochromats whose magnetisation the Bloch equation carries from one block to the next."""

from __future__ import annotations

import numpy as np

from ._ext import sim as _kernels

__all__ = ["Isochromats"]


def _per_isochromat(value, count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    try:
        return np.ascontiguousarray(np.broadcast_to(values, (count,)))
    except ValueError:
        raise ValueError(
            f"{name} must be one value or one per isochromat, got shape {values.shape}"
        ) from None


def _sensitivities(value, count: int, name: str):
    if value is None:
        return None
    values = np.asarray(value, dtype=complex)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != count:
        raise ValueError(
            f"{name} must be (isochromats,) or (isochromats, channels), got shape {values.shape}"
        )
    return np.ascontiguousarray(values)


class Isochromats:
    """Isochromats, and the magnetisation the Bloch equation carries from one block to the next.

    The magnetisation turns about the field ``b = (Re b1, Im b1, bz)``, in Hz,
    in the frame rotating at the reference frequency, as the magnetic moment
    of a nucleus of positive gyromagnetic ratio precesses: ``dM/dt = 2 pi M x
    b``, clockwise seen from the tip of ``b``. A 90 degree pulse along ``+x``
    takes ``+z`` to ``+y``, and an isochromat at a positive off-resonance or
    along a positive gradient accrues ``Mx + i My`` a negative phase.

    Free precession under a piecewise-linear gradient, with relaxation, is
    integrated exactly and applied when the magnetisation is next needed. An
    RF pulse is a sequence of steps, each a rotation about the step's mean
    field between two half steps of relaxation; isochromats that see the same
    field during a pulse share its computation.

    Parameters
    ----------
    positions : array_like
        ``(n, 3)`` positions, in m, along the axes the gradients are played on.
    proton_density : float or array_like, default=1.0
        Equilibrium longitudinal magnetisation, per isochromat.
    t1, t2 : float or array_like, default=inf
        Relaxation times, in s; ``inf`` for none.
    off_resonance : float or array_like, default=0.0
        Precession frequency at rest, in Hz from the reference frequency:
        field inhomogeneity and chemical shift together.
    transmit : array_like, default=None
        Complex transmit sensitivities, ``(n,)`` or ``(n, channels)``, scaling
        the field of each RF channel. By default one channel of unit
        sensitivity, onto which the channels of a pTx pulse are summed.
    receive : array_like, default=None
        Complex receive sensitivities, ``(n,)`` or ``(n, coils)``. By default
        one coil of unit sensitivity.
    threads : int, default=0
        Worker threads; 0 for every core.

    Raises
    ------
    ValueError
        If a property has the wrong shape or is not finite, or a relaxation
        time is not positive.

    Notes
    -----
    Every isochromat starts at equilibrium, along ``+z``. The engine computes
    what the sequence does to these isochromats; it does not model the
    scanner's hardware, diffusion, flow or motion.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseqpp as pp
    >>> spins = pp.Isochromats([[0.0, 0.0, 0.0]], t2=0.1)

    A 25 kHz field along ``+x`` held for 10 us turns ``+z`` by 90 degrees,
    to ``+y``; the transverse magnetisation then decays with ``t2``:

    >>> pulse = (0.0, 10e-6, [25e3])
    >>> signal = spins.play(10.01e-3, rf=pulse, adc=[10e-6, 10.01e-3])
    >>> np.round(signal, 4)
    array([[0.+1.j    , 0.+0.9048j]])
    """

    def __init__(
        self,
        positions,
        *,
        proton_density=1.0,
        t1=np.inf,
        t2=np.inf,
        off_resonance=0.0,
        transmit=None,
        receive=None,
        threads: int = 0,
    ):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must be (n, 3), got shape {positions.shape}")
        count = positions.shape[0]
        self._native = _kernels.Isochromats(
            np.ascontiguousarray(positions),
            _per_isochromat(proton_density, count, "proton_density"),
            _per_isochromat(t1, count, "t1"),
            _per_isochromat(t2, count, "t2"),
            _per_isochromat(off_resonance, count, "off_resonance"),
            _sensitivities(transmit, count, "transmit"),
            _sensitivities(receive, count, "receive"),
            int(threads),
        )

    def __len__(self) -> int:
        return self._native.size

    @property
    def coils(self) -> int:
        """Number of receive coils; one without receive sensitivities."""
        return self._native.coils

    @property
    def elapsed(self) -> float:
        """Time played since construction or the last :meth:`reset`, in s."""
        return self._native.elapsed

    @property
    def magnetization(self) -> np.ndarray:
        """``(n, 3)`` magnetisation, a copy; assigning replaces it."""
        return self._native.magnetization()

    @magnetization.setter
    def magnetization(self, value) -> None:
        values = np.broadcast_to(np.asarray(value, dtype=float), (len(self), 3))
        self._native.set_magnetization(np.ascontiguousarray(values))

    def reset(self) -> None:
        """Return every isochromat to equilibrium, along ``+z``, and the clock to zero."""
        self._native.reset()

    def play(self, duration: float, *, gradients=None, rf=None, adc=None) -> np.ndarray:
        """Play one block's events and return what each coil receives at each ADC sample.

        Times are in s from the block's start.

        Parameters
        ----------
        duration : float
            Block duration, in s.
        gradients : sequence, default=None
            Three entries, for the x, y and z axes, each ``None`` or a
            ``(2, m)`` array of corner times over gradient amplitude, in Hz/m.
            The gradient is linear between the corners and zero outside them.
        rf : tuple, default=None
            ``(start, step, samples)``: the transverse field ``b1``, in Hz, as
            ``(steps,)`` or ``(channels, steps)`` complex samples, each held for
            ``step`` from ``start`` on. Without transmit sensitivities the
            channels are summed; with them there must be one channel each.
        adc : array_like, default=None
            Increasing sample times, none inside the RF pulse.

        Returns
        -------
        NDArray[np.complex128]
            ``(coils, samples)``: at each sample, the sum over the isochromats of
            the receive sensitivity times ``Mx + i My``, not demodulated.

        Raises
        ------
        ValueError
            If an event does not fit in the block, an ADC sample falls inside
            the RF pulse, or the RF channels do not match the transmit
            sensitivities.
        """
        axes = [None, None, None] if gradients is None else list(gradients)
        if len(axes) != 3:
            raise ValueError("gradients must hold the three axes, x, y and z")
        axes = [
            None
            if axis is None or np.size(axis) == 0
            else np.ascontiguousarray(axis, dtype=float)
            for axis in axes
        ]
        start = step = 0.0
        samples = None
        if rf is not None:
            start, step, samples = rf
            samples = np.asarray(samples, dtype=complex)
            if samples.ndim == 1:
                samples = samples[None, :]
            samples = np.ascontiguousarray(samples)
        times = None if adc is None else np.ascontiguousarray(adc, dtype=float)
        return self._native.play(
            float(duration), axes, float(start), float(step), samples, times
        )
