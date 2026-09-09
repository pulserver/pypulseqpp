"""What an expansion knows, said in names rather than in tuple positions.

Upstream PyPulseq's `waveforms_and_times` returns five values, and a script
written against it unpacks five. That is what `compat=True` is for, and it is
the default everywhere: a drop-in replacement has to hand back what the thing
it replaces hands back.

What that tuple cannot say is the rest of what the pass already worked out.
Pulseq has seven RF uses; the tuple carries two, and an inversion pulse does
not appear in it at all. The ADC offsets come at two granularities and the
tuple carries one of them. So `compat=False` returns these instead, which
carry all of it and say what each part is.

The shapes are the ones `pulserver` settled on, because a script written
against that reads the same way here.
"""

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
    """The gradient waveforms, and the RF envelope if it was asked for.

    Each channel is a ``(2, n)`` array of times in seconds over amplitudes in
    Hz/m, which is upstream's layout. ``rf`` is complex, and is None unless
    ``append_RF`` was set.
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
    """Every RF pulse, in play order, with its use.

    One flat table rather than upstream's two buckets, so nothing is dropped
    and a caller who wants the buckets asks for them by name.

    ``freq_offset`` and ``phase_offset`` carry the ppm terms, and the phase
    carries the ``2*pi*f*t_centre`` term that takes it to the pulse's centre,
    which is the convention both toolboxes use.
    """

    t: np.ndarray
    freq_offset: np.ndarray
    phase_offset: np.ndarray
    use: tuple[str, ...]
    block: np.ndarray

    def of(self, *uses: str) -> RfTimes:
        """Return the pulses carrying any of ``uses``, as another `RfTimes`.

        More than one, because upstream's buckets are not one use each: a
        pulse whose row records no use reads back as ``undefined``, and
        upstream counts it as an excitation. Reproducing upstream is asking
        for both.
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
    """Every ADC sample, and the offsets that apply to it.

    Two granularities, because the two toolboxes disagree about which one
    ``fp_adc`` means and both are worth having:

    - **Per window**: ``freq_offset``, ``phase_offset``, ``block``,
      ``num_samples``. This is upstream's ``fp_adc`` -- the offsets as the
      event records them, with no ppm term folded in.
    - **Per sample**: ``t``, ``phase_modulation``, ``sample_phase``. The last
      two are the toolbox's ``pm_adc`` and the second row of its ``fp_adc``:
      the phase each sample is actually acquired with, ppm and phase
      modulation and the accumulated ``2*pi*f*t`` all in. That is the number a
      simulation or a demodulator wants.
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
    """Everything `waveforms_and_times` knows.

    Upstream's five values are ``waveforms.channels``,
    ``rf.of("excitation", "undefined").tfp``, ``rf.of("refocusing").tfp``,
    ``adc.t`` and ``adc.fp``. What it cannot express is the other five RF
    uses, the per-sample ADC phase, and which block each of them is in.
    """

    waveforms: Waveforms
    rf: RfTimes
    adc: AdcTimes
