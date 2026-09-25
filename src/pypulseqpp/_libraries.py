"""A sequence's block table and libraries as arrays: the tables its Pulseq file holds."""

from __future__ import annotations

__all__ = ["SequenceLibraries", "Shape", "libraries"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import _ext as _cxx
from ._offsets import _hz_per_ppm

#: What :attr:`SequenceLibraries.layout` holds.
_LAYOUT = 1

#: The tables the native export hands back as arrays, and those it hands back
#: as lists of names.
_ARRAYS = (
    "rf",
    "trapezoid_ids",
    "trapezoids",
    "arbitrary_gradient_ids",
    "arbitrary_gradients",
    "adc",
    "extensions",
    "triggers",
    "rotations",
    "label_set_values",
    "label_inc_values",
    "soft_delays",
)
_NAMES = ("rf_use", "label_set_labels", "label_inc_labels", "soft_delay_hints")


@dataclass(frozen=True, eq=False)
class Shape:
    """One row of the shape library, as a file writes it.

    The fields are those of the namespace upstream's ``decompress_shape``
    reads, so that function accepts a shape too.

    Attributes
    ----------
    num_samples : int
        Samples the shape stands for.
    data : NDArray[np.float64]
        The values stored: the samples run-length encoded on their derivative
        when there are fewer of them than ``num_samples``, the samples
        themselves when there are as many.
    """

    num_samples: int
    data: NDArray[np.float64]

    def decompressed(self) -> NDArray[np.float64]:
        """Return the ``num_samples`` samples the shape stands for."""
        return np.asarray(_cxx.decompress_shape(self.data, self.num_samples))


@dataclass(frozen=True, eq=False)
class SequenceLibraries:
    """A sequence's block table and every library it holds, as read-only arrays.

    Each table is a section of a Pulseq file of the sequence, with the file's
    columns in the file's order and without its id column. Times are in
    seconds; every other column is in the file's unit. Ids are 1-based and 0
    names no event: row ``i`` holds id ``i + 1``, except in the two gradient
    tables, which share one numbering and list the id each row holds. Columns
    holding ids hold whole numbers.

    Attributes
    ----------
    layout : int
        The layout of these tables, 1 for the one described here. It rises
        when a table gains, loses or reorders a column, or a column changes
        unit.
    block_durations : NDArray[np.float64]
        ``(N,)``: each block's duration.
    blocks : NDArray[np.int32]
        ``(N, 6)``: each block's RF, gx, gy, gz, ADC and extension ids.
    rf : NDArray[np.float64]
        ``(R, 10)``: amplitude in Hz; the magnitude, phase and time shape ids,
        0 for a time shape on the RF raster; centre; delay; frequency ppm
        offset in ppm; phase ppm offset in rad/MHz; frequency offset in Hz;
        phase offset in rad.
    rf_use : tuple[str, ...]
        Per RF id, the event's use: ``'excitation'``, ``'refocusing'``,
        ``'inversion'``, ``'saturation'``, ``'preparation'``, ``'other'`` or
        ``'undefined'``.
    trapezoid_ids : NDArray[np.int32]
        ``(T,)``: the gradient id each row of :attr:`trapezoids` holds.
    trapezoids : NDArray[np.float64]
        ``(T, 5)``: amplitude in Hz/m; rise, flat and fall times; delay.
    arbitrary_gradient_ids : NDArray[np.int32]
        ``(A,)``: the gradient id each row of :attr:`arbitrary_gradients`
        holds.
    arbitrary_gradients : NDArray[np.float64]
        ``(A, 6)``: amplitude, first and last values in Hz/m; the waveform
        shape id; the time shape id, 0 for samples at the centres of the
        gradient raster and -1 for samples every half raster; delay.
    adc : NDArray[np.float64]
        ``(D, 8)``: sample count; dwell; delay; frequency ppm offset in ppm;
        phase ppm offset in rad/MHz; frequency offset in Hz; phase offset in
        rad; the phase-modulation shape id, 0 for none.
    shapes : tuple[Shape, ...]
        By shape id. An RF magnitude is normalised to its peak, an RF phase is
        in cycles, a gradient waveform is normalised to its amplitude, a time
        shape counts rasters and a phase modulation is in rad.
    extensions : NDArray[np.int32]
        ``(E, 3)``: each link's extension type id, the id of the row it names
        in that type's table, and the id of the next link, 0 ending the chain.
    extension_types : dict[str, int]
        The type id of each extension the sequence numbers, by name.
    triggers : NDArray[np.float64]
        ``(G, 4)``: control code, 1 for an output and 2 for an input; channel
        code; delay; duration.
    rotations : NDArray[np.float64]
        ``(Q, 4)``: a unit quaternion, scalar first.
    label_set_values, label_inc_values : NDArray[np.int32]
        ``(L,)``: the value each row sets its label to or increments it by.
    label_set_labels, label_inc_labels : tuple[str, ...]
        The label each row names.
    rf_shims : tuple[NDArray[np.float64], ...]
        Per row, one magnitude and one phase in rad for each transmit channel,
        alternating.
    soft_delays : NDArray[np.float64]
        ``(S, 3)``: the delay's number, offset and factor.
    soft_delay_hints : tuple[str, ...]
        The hint each soft-delay row names.
    """

    layout: int
    block_durations: NDArray[np.float64]
    blocks: NDArray[np.int32]
    rf: NDArray[np.float64]
    rf_use: tuple[str, ...]
    trapezoid_ids: NDArray[np.int32]
    trapezoids: NDArray[np.float64]
    arbitrary_gradient_ids: NDArray[np.int32]
    arbitrary_gradients: NDArray[np.float64]
    adc: NDArray[np.float64]
    shapes: tuple[Shape, ...]
    extensions: NDArray[np.int32]
    extension_types: dict[str, int]
    triggers: NDArray[np.float64]
    rotations: NDArray[np.float64]
    label_set_values: NDArray[np.int32]
    label_set_labels: tuple[str, ...]
    label_inc_values: NDArray[np.int32]
    label_inc_labels: tuple[str, ...]
    rf_shims: tuple[NDArray[np.float64], ...]
    soft_delays: NDArray[np.float64]
    soft_delay_hints: tuple[str, ...]

    def absolute_offsets(
        self, system
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return every RF and ADC row's offsets with its ppm offsets resolved.

        Parameters
        ----------
        system : Opts
            The gamma (Hz/T) and B0 (T) the ppm offsets are resolved at.

        Returns
        -------
        rf : NDArray[np.float64]
            ``(R, 2)``: each RF row's frequency offset in Hz and phase offset
            in rad, as :func:`pypulseqpp.calc_absolute_offsets` gives them for
            its event.
        adc : NDArray[np.float64]
            ``(D, 2)``: the same for each ADC row.

        Examples
        --------
        >>> import pypulseqpp as pp
        >>> seq = pp.Sequence(pp.Opts())
        >>> seq.add_block(pp.make_block_pulse(1.0, duration=4e-3, freq_ppm=-3.45))
        1
        >>> rf, adc = seq.libraries().absolute_offsets(pp.Opts(B0=3.0))
        >>> rf.round(2).tolist(), adc.shape
        ([[-440.66, 0.0]], (0, 2))
        """
        f = _hz_per_ppm(system)
        return (
            self.rf[:, 8:10] + self.rf[:, 6:8] * f,
            self.adc[:, 5:7] + self.adc[:, 3:5] * f,
        )


def _frozen(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


def libraries(native) -> SequenceLibraries:
    """Take a native sequence's tables; see :meth:`pypulseqpp.Sequence.libraries`."""
    tables = native.libraries()
    return SequenceLibraries(
        layout=_LAYOUT,
        # Views of the block table's snapshot, which no later edit reaches;
        # read-only, so that nothing writes through them into the table.
        block_durations=_frozen(native.block_durations()),
        blocks=_frozen(native.block_events()),
        shapes=tuple(
            Shape(int(count), _frozen(data)) for count, data in tables["shapes"]
        ),
        extension_types=dict(tables["extension_types"]),
        rf_shims=tuple(_frozen(row) for row in tables["rf_shims"]),
        **{name: _frozen(tables[name]) for name in _ARRAYS},
        **{name: tuple(tables[name]) for name in _NAMES},
    )
