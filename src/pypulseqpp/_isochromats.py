"""Isochromats whose magnetisation the Bloch equation carries from one block to the next."""

from __future__ import annotations

import numpy as np

from ._ext import sim as _kernels
from ._offsets import calc_absolute_offsets
from ._opts import Opts as _Opts

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


def _axes(gradients) -> list:
    axes = [None, None, None] if gradients is None else list(gradients)
    if len(axes) != 3:
        raise ValueError("gradients must hold the three axes, x, y and z")
    return [
        None
        if axis is None or np.size(axis) == 0
        else np.ascontiguousarray(axis, dtype=float)
        for axis in axes
    ]


def _rotation(rotation):
    if rotation is None:
        return None
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"rotation must be a finite (3, 3) matrix, got shape {matrix.shape}"
        )
    return np.ascontiguousarray(matrix)


def _field(rf, system) -> tuple:
    """Return the field ``rf`` plays as ``(start, step, samples)``, samples ``(channels, steps)``."""
    if rf is None:
        return 0.0, 0.0, None
    if hasattr(rf, "signal"):
        frequency, phase = calc_absolute_offsets(rf, system=system)
        return _kernels.pulse_steps(
            np.ascontiguousarray(rf.t, dtype=float).ravel(),
            np.ascontiguousarray(rf.signal, dtype=complex).ravel(),
            float(rf.delay),
            phase,
            frequency,
            float((_Opts.default if system is None else system).rf_raster_time),
        )
    start, step, samples = rf
    samples = np.asarray(samples, dtype=complex)
    if samples.ndim == 1:
        samples = samples[None, :]
    return float(start), float(step), np.ascontiguousarray(samples)


def _window(adc, system) -> tuple:
    """Return the sample times and the phase each is demodulated by, None for plain times."""
    if adc is None:
        return None, None
    if not hasattr(adc, "num_samples"):
        return np.ascontiguousarray(adc, dtype=float), None
    count = int(adc.num_samples)
    dwell = float(adc.dwell)
    if count < 0 or not dwell > 0.0:
        raise ValueError("an ADC needs a count of samples and a positive dwell time")
    modulation = getattr(adc, "phase_modulation", None)
    modulation = (
        np.zeros(0)
        if modulation is None
        else np.asarray(modulation, dtype=float).ravel()
    )
    if modulation.size not in (0, count):
        raise ValueError(
            f"an ADC's phase modulation must hold one value per sample, {count}, got {modulation.size}"
        )
    frequency, phase = calc_absolute_offsets(adc, system=system)
    return _kernels.adc_window(
        count,
        dwell,
        float(adc.delay),
        phase,
        frequency,
        np.ascontiguousarray(modulation),
    )


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

    def play(
        self,
        duration: float,
        *,
        gradients=None,
        rotation=None,
        rf=None,
        adc=None,
        system=None,
    ) -> np.ndarray:
        """Play one block's events and return what each coil receives at each ADC sample.

        Times are in s from the block's start. An RF or ADC event plays as in
        :meth:`Sequence.simulate`, which :doc:`/explanations/simulation`
        describes; the field and sample times the events amount to can be
        given instead.

        Parameters
        ----------
        duration : float
            Block duration, in s.
        gradients : sequence, default=None
            Three entries, for the x, y and z axes, each ``None`` or a
            ``(2, m)`` array of corner times over gradient amplitude, in Hz/m.
            The gradient is linear between the corners and zero outside them.
        rotation : array_like, default=None
            ``(3, 3)`` matrix from the axes ``gradients`` are given along to
            the axes of the positions, a reflection allowed. Each axis plays
            the sum its row weights, exact on the union of the corners, with
            a gradient's step from or to zero at its first or last corner
            kept as a step.
        rf : RF event or tuple, default=None
            An RF event: ``signal``, in Hz, at the times ``t`` from the pulse's
            start, which is ``delay`` into the block, with its phase and
            frequency offsets; a dynamic pTx pulse holds its channels one
            after another over one time base. Or the transverse field ``b1``,
            as ``(start, step, samples)``: ``(steps,)`` or
            ``(channels, steps)`` complex samples, in Hz, each held for
            ``step`` from ``start`` on. Without transmit sensitivities the
            channels are summed; with them there must be one channel each.
        adc : ADC event or array_like, default=None
            An ADC event, sampled at the middle of each dwell from its delay
            on, whose samples are returned demodulated by its phase offset,
            frequency offset and phase modulation. Or increasing sample
            times, whose samples are returned as received. No sample may fall
            inside the RF pulse.
        system : Opts, default=None
            The RF raster, over whose steps an RF event with a time shape is
            held, and the gamma and B0 the events' ppm offsets are resolved
            at; the default system when None.

        Returns
        -------
        NDArray[np.complex128]
            ``(coils, samples)``: at each sample, the sum over the isochromats of
            the receive sensitivity times ``Mx + i My``, demodulated for an ADC
            event.

        Raises
        ------
        ValueError
            If an event does not fit in the block, an ADC sample falls inside
            the RF pulse, the RF channels do not match the transmit
            sensitivities, or an event or the rotation is malformed.

        Examples
        --------
        >>> import numpy as np
        >>> import pypulseqpp as pp
        >>> spins = pp.Isochromats([[0.0, 0.0, 0.0]])
        >>> rf = pp.make_block_pulse(np.pi / 2, duration=1e-3)
        >>> adc = pp.make_adc(1, dwell=10e-6, delay=1e-3)
        >>> np.round(spins.play(1.01e-3, rf=rf, adc=adc), 6)
        array([[0.+1.j]])
        """
        start, step, samples = _field(rf, system)
        times, receiver = _window(adc, system)
        signal = self._native.play(
            float(duration),
            _axes(gradients),
            _rotation(rotation),
            start,
            step,
            samples,
            times,
        )
        return signal if receiver is None else signal * np.exp(1j * receiver)
