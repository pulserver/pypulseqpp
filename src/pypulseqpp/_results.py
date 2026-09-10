"""Named waveform and timing results for ``compat=False``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AdcTimes", "RfTimes", "Waveforms", "WaveformsAndTimes"]

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
    """Return the use a stored code stands for; `undefined` for anything else."""
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
        """The gradients as upstream returns them: a list, RF appended."""
        channels = [self.gx, self.gy, self.gz]
        if self.rf is not None:
            channels.append(self.rf)
        return channels

    @property
    def duration(self) -> float:
        """The last time any channel carries, in seconds."""
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
