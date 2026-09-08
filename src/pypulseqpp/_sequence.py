"""The sequence a design script builds, over the compiled core.

What a caller holds is this; what does the work is a `pypulseqpp._ext`
sequence underneath. The methods here are the ones a design loop and a
writer need, and each is a single call into the core rather than a loop in
Python: `add_block` unpacks its events and registers them inside C++, and
reading and writing are compiled passes.
"""

from __future__ import annotations

from pathlib import Path

from . import _ext as _cxx

__all__ = ["Sequence"]


class Sequence:
    """A Pulseq sequence: event libraries, a block table, and definitions."""

    def __init__(self, system=None) -> None:
        """Start an empty sequence.

        Parameters
        ----------
        system : pypulseq.Opts, optional
            The system the sequence is designed against. Its rasters are
            recorded, because a block's duration is stored as a count of
            them and a reader cannot recover the seconds without them.
        """
        self._native = _cxx.Sequence()
        self.system = system
        if system is not None:
            self._native.set_rasters(
                system.rf_raster_time,
                system.grad_raster_time,
                system.adc_raster_time,
                system.block_duration_raster,
            )

    # -- blocks --------------------------------------------------------

    def add_block(self, *events) -> int:
        """Append one block playing ``events``.

        Parameters
        ----------
        *events
            Any mixture of RF, gradient, ADC, delay, label, trigger,
            rotation, RF shim and soft delay events. A bare delay sets a
            floor on the block's duration.

        Returns
        -------
        int
            The new block's 1-based index.

        Notes
        -----
        The block lasts as long as the longest thing in it, rounded up onto
        the block duration raster, which is what a caller passing a bare
        delay is asking for directly.
        """
        return self._native.add_block_events(*events)

    def set_block(self, index: int, *events) -> None:
        """Write ``events`` over the block at ``index``, 1-based."""
        self._native.set_block_events(index, *events)

    def get_block(self, index: int):
        """Return the block at ``index``, 1-based, and what it plays."""
        return self._native.get_block(index)

    @property
    def block_durations(self):
        """Every block's duration in seconds, as a view over the table."""
        return self._native.block_durations()

    def duration(self) -> float:
        """How long the sequence plays for, in seconds."""
        return self._native.duration()

    def __len__(self) -> int:
        return self._native.num_blocks()

    # -- definitions ---------------------------------------------------

    def set_definition(self, key: str, value) -> None:
        """Record ``key`` in `[DEFINITIONS]`."""
        self._native.set_definition(key, value)

    def get_definition(self, key: str):
        """Return what ``key`` says, or None if it is not defined."""
        return self._native.definitions().get(key)

    @property
    def definitions(self) -> dict:
        """Everything `[DEFINITIONS]` will carry."""
        return self._native.definitions()

    # -- files ---------------------------------------------------------

    def write(self, name, create_signature: bool = True) -> None:
        """Write a Pulseq 1.5.1 `.seq` file."""
        Path(name).write_bytes(_cxx.write_text(self._native, create_signature))

    def write_binary(self, name) -> None:
        """Write the binary form, which a scanner parses faster."""
        Path(name).write_bytes(_cxx.write_binary(self._native))

    def write_v141(
        self, name, create_signature: bool = True, gamma=42576000.0, field=1.5
    ) -> None:
        """Write a Pulseq 1.4.1 file, for an interpreter that predates 1.5."""
        Path(name).write_bytes(
            _cxx.write_text_v141(self._native, create_signature, gamma, field)
        )

    def read(self, name, verify: bool = False) -> None:
        """Replace this sequence with the one in ``name``.

        Text or binary, told apart by what is in the bytes. Every revision
        from 1.2.0 is read, older ones by converting them.
        """
        self._native = _cxx.read(Path(name).read_bytes(), verify)

    # -- collapsing ----------------------------------------------------

    def remove_duplicates(self) -> None:
        """Collapse identical library rows and renumber what points at them."""
        self._native.remove_duplicates()
