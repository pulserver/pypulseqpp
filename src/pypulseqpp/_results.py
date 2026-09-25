"""Named waveform and timing results returned under ``compat=False``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "AdcEchoes",
    "AdcTimes",
    "RfBandwidth",
    "RfGradients",
    "RfTimes",
    "Waveforms",
    "WaveformsAndTimes",
]

#: Pulseq's RF uses, in the order `supported_rf_uses` lists them.
RF_USES = (
    "excitation",
    "refocusing",
    "inversion",
    "saturation",
    "preparation",
    "other",
    "undefined",
)

#: What a `use` reads as in a library row, which is its first letter.
_BY_CODE = {use[0]: use for use in RF_USES}


def use_of(code: str) -> str:
    """Return the RF use a stored code denotes; `undefined` for any other code.

    Parameters
    ----------
    code : str
        A use as a library row records it, which is its first letter.

    Returns
    -------
    str
        One of `RF_USES`.
    """
    return _BY_CODE.get(code, "undefined")


@dataclass(frozen=True)
class Waveforms:
    """Gradient corner waveforms and an optional complex RF envelope.

    Each channel has shape ``(2, n)``: time in seconds, then amplitude.
    Gradient amplitudes are in Hz/m; RF amplitudes are in Hz.
    ``rf`` is None unless RF expansion was requested.
    """

    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray
    rf: np.ndarray | None = None

    @property
    def channels(self) -> list[np.ndarray]:
        """The gradient channels as upstream returns them: a list, with RF appended."""
        channels = [self.gx, self.gy, self.gz]
        if self.rf is not None:
            channels.append(self.rf)
        return channels

    @property
    def duration(self) -> float:
        """Latest time present on any channel, in seconds."""
        ends = [channel[0, -1] for channel in self.channels if channel.shape[1]]
        return float(max(ends)) if ends else 0.0


@dataclass(frozen=True)
class RfTimes:
    """RF centres in play order, including all Pulseq use tags.

    ``t`` is in seconds; ``block`` contains 1-based indices. Frequency
    offsets are in Hz and phase offsets in radians, including ppm terms.
    Phase includes ``2*pi*frequency*centre`` relative to the RF event.
    """

    t: np.ndarray
    freq_offset: np.ndarray
    phase_offset: np.ndarray
    use: tuple[str, ...]
    block: np.ndarray

    def of(self, *uses: str) -> RfTimes:
        """Select pulses with any of the specified use tags.

        An undefined use is not implicitly included with excitation; request
        both tags to reproduce the PyPulseq excitation group.

        Parameters
        ----------
        *uses : str
            The uses to keep, each one of `RF_USES`.

        Returns
        -------
        RfTimes
            The pulses carrying one of ``uses``, in play order.

        Raises
        ------
        ValueError
            If a use is not one of `RF_USES`.
        """
        unknown = set(uses) - set(RF_USES)
        if unknown:
            raise ValueError(
                f"unknown RF use(s) {sorted(unknown)}; expected one of {RF_USES}"
            )
        keep = np.array([tag in set(uses) for tag in self.use], dtype=bool)
        return RfTimes(
            t=self.t[keep],
            freq_offset=self.freq_offset[keep],
            phase_offset=self.phase_offset[keep],
            use=tuple(tag for tag, taken in zip(self.use, keep, strict=True) if taken),
            block=self.block[keep],
        )

    @property
    def tfp(self) -> np.ndarray:
        """``(3, n)``: time over frequency over phase, upstream's packing."""
        return np.vstack((self.t, self.freq_offset, self.phase_offset))

    def __len__(self) -> int:
        return int(self.t.size)


@dataclass(frozen=True)
class AdcTimes:
    """ADC sampling times and offsets at window and sample resolution.

    Per window, ``freq_offset`` (Hz) and ``phase_offset`` (rad) are stored
    offsets without ppm corrections; ``block`` is 1-based and
    ``num_samples`` gives the window length.

    Per sample, ``t`` is in seconds, ``phase_modulation`` and
    ``sample_phase`` are in radians, and ``sample_frequency`` is in Hz.
    Sample phase includes ppm terms, modulation and accumulated
    ``2*pi*frequency*time`` within the ADC event.
    """

    t: np.ndarray
    freq_offset: np.ndarray
    phase_offset: np.ndarray
    phase_modulation: np.ndarray
    sample_phase: np.ndarray
    sample_frequency: np.ndarray
    block: np.ndarray
    num_samples: np.ndarray

    @property
    def fp(self) -> np.ndarray:
        """``(n_windows, 2)``: frequency over phase, upstream's ``fp_adc``."""
        return np.stack((self.freq_offset, self.phase_offset), axis=-1)

    def __len__(self) -> int:
        return int(self.t.size)


@dataclass(frozen=True)
class WaveformsAndTimes:
    """Waveforms, RF centres and ADC timing, including all RF use tags.

    The PyPulseq-compatible tuple is ``(waveforms.channels,
    rf.of("excitation", "undefined").tfp, rf.of("refocusing").tfp,
    adc.t, adc.fp)``.
    """

    waveforms: Waveforms
    rf: RfTimes
    adc: AdcTimes


@dataclass(frozen=True)
class RfBandwidth:
    """RF bandwidth, the spectrum it was measured on, and the bands in it.

    Frequencies are in Hz. ``frequency`` is centred on where the pulse is
    tuned; ``band_offsets`` are relative to that tuning, the pulse's own
    frequency offset, so a single-band pulse has one band at zero. The
    PyPulseq-compatible tuple is ``(bandwidth, spectrum, frequency)``.

    Attributes
    ----------
    bandwidth : float
        Width between the outermost crossings at ``cutoff`` of the peak, over
        every band.
    spectrum : numpy.ndarray
        Complex Fourier transform of the envelope, one value per ``frequency``.
    frequency : numpy.ndarray
        Frequency of each bin of ``spectrum``. Upstream's axis, which the
        compatible tuple returns, labels every bin one bin low.
    band_offsets : numpy.ndarray
        Magnitude-weighted centre of each band, lowest first.
    band_bandwidths : numpy.ndarray
        Width of each band at ``cutoff`` of that band's own peak.
    """

    bandwidth: float
    spectrum: np.ndarray
    frequency: np.ndarray
    band_offsets: np.ndarray
    band_bandwidths: np.ndarray

    @property
    def num_bands(self) -> int:
        """Number of bands; one for a pulse with a single passband."""
        return int(self.band_offsets.size)


@dataclass(frozen=True)
class AdcEchoes:
    """Per readout, the axes its k-space moves along and the samples nearest the centre.

    One entry per block that acquires, in play order; see
    :meth:`pypulseqpp.Sequence.adc_echoes`.

    Attributes
    ----------
    block : NDArray[np.int32]
        ``(n,)``: the 1-based block.
    num_samples : NDArray[np.int32]
        ``(n,)``: the samples it acquires.
    first_sample : NDArray[np.int64]
        ``(n,)``: the column of its first sample in
        :meth:`pypulseqpp.Sequence.adc_kspace`.
    moving : NDArray[np.bool_]
        ``(n, 3)``: whether its k-space moves along x, y and z.
    echo : NDArray[np.int32]
        ``(n, 2)``: the first and last 0-based sample no further from the
        centre of k-space, over the moving axes, than the nearest sample plus
        1% of the larger k step beside it; -1 for a readout that does not move
        or has fewer than two samples.
    """

    block: np.ndarray
    num_samples: np.ndarray
    first_sample: np.ndarray
    moving: np.ndarray
    echo: np.ndarray


@dataclass(frozen=True)
class RfGradients:
    """The gradient each RF pulse plays under, along the channel axes.

    One entry per block with RF, in play order; see
    :meth:`pypulseqpp.Sequence.rf_gradients`.

    Attributes
    ----------
    block : NDArray[np.int32]
        ``(n,)``: the 1-based block.
    steady : NDArray[np.bool_]
        ``(n, 3)``: whether the gradient along x, y and z holds one value from
        the pulse's first sample to its last; an axis without a gradient is
        steady at zero.
    gradient : NDArray[np.float64]
        ``(n, 3)``: the gradient along x, y and z at the pulse's centre, in
        Hz/m.
    """

    block: np.ndarray
    steady: np.ndarray
    gradient: np.ndarray
