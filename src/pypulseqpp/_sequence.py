"""The sequence a design script builds, over the compiled core.

What a caller holds is this; what does the work is a `pypulseqpp._ext`
sequence underneath. The methods here are the ones a design loop and a
writer need, and each is a single call into the core rather than a loop in
Python: `add_block` unpacks its events and registers them inside C++, and
reading and writing are compiled passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from warnings import warn

import numpy as np

from . import _ext as _cxx
from ._check_timing import check_timing as _check_timing
from ._check_timing import print_error_report

__all__ = ["Sequence"]

#: Where a written file records its own hash.
_SIGNATURE = re.compile(r"^Hash (\w+)$", re.MULTILINE)

#: The event columns upstream's block table carries. The core's rows do not
#: hold the leading delay, because a pure delay in Pulseq 1.5 is a block
#: duration rather than an event, so a count reported per column pads it back
#: on and a caller's column indices are upstream's.
_UPSTREAM_BLOCK_WIDTH = 7


class Sequence:
    """A Pulseq sequence: event libraries, a block table, and definitions."""

    def __init__(self, system=None, use_block_cache: bool = True) -> None:
        """Start an empty sequence.

        Parameters
        ----------
        system : pypulseq.Opts, optional
            The system the sequence is designed against. Its rasters are
            recorded, because a block's duration is stored as a count of
            them and a reader cannot recover the seconds without them.
        use_block_cache : bool, default True
            Accepted and remembered; nothing here caches. See the no-ops at
            the end of this class.
        """
        self._native = _cxx.Sequence()
        self.system = system
        self._use_block_cache = use_block_cache
        self._use_event_cache = True
        self._trid_names: list[str] = []
        self.signature_type: str | None = None
        self.signature_file: str | None = None
        self.signature_value: str | None = None
        if system is not None:
            self._native.set_rasters(
                system.rf_raster_time,
                system.grad_raster_time,
                system.adc_raster_time,
                system.block_duration_raster,
            )

    def __str__(self) -> str:
        native = self._native
        return "\n".join(
            [
                "Sequence:",
                f"blocks: {native.num_blocks()}",
                f"rf_library: {native.num_rf()}",
                f"grad_library: {native.num_gradients()}",
                f"adc_library: {native.num_adc()}",
                f"extensions_library: {native.num_extensions()}",
                f"shape_library: {native.num_shapes()}",
            ]
        )

    def __len__(self) -> int:
        return self._native.num_blocks()

    def __enter__(self) -> Sequence:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.clear_caches()
        return False

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

    def duration(self) -> tuple[float, int, np.ndarray]:
        """Return how long the sequence plays for, and what it is made of.

        Returns
        -------
        duration : float
            Total duration in seconds.
        num_blocks : int
            How many blocks there are.
        event_count : np.ndarray
            How many blocks carry an event in each column of the block
            table, in upstream's column order: delay, RF, the three
            gradient axes, ADC, extension.
        """
        native = self._native
        events = native.block_events()
        counts = np.zeros(_UPSTREAM_BLOCK_WIDTH)
        # Column 0 is upstream's delay library, which Pulseq 1.5 does not
        # have: a block with nothing in it lasts for its stored duration.
        counts[1:] = (events > 0).sum(axis=0)
        return native.duration(), native.num_blocks(), counts

    def find_block_by_time(self, t: float):
        """Return the 1-based index of the block playing at ``t`` seconds.

        Parameters
        ----------
        t : float
            A time from the start of the sequence, in seconds.

        Returns
        -------
        int or None
            The block playing then, or None if ``t`` is past the end.
        """
        durations = self._native.block_durations()
        index = int(np.searchsorted(np.cumsum(durations), t, side="right"))
        if index >= len(durations):
            return None
        if durations[index] <= 0:
            raise ValueError("Block duration cannot be negative")
        return index + 1

    # -- timing --------------------------------------------------------

    def check_timing(self, print_errors: bool = False):
        """Return whether the sequence is playable, and every problem found.

        Parameters
        ----------
        print_errors : bool, default False
            Print the report as well as returning it.

        Returns
        -------
        is_ok : bool
            True when nothing was found.
        error_report : list of SimpleNamespace
            One entry per problem, in block order.
        """
        is_ok, error_report = _check_timing(self)
        if not is_ok and print_errors:
            print_error_report(self, error_report)
        return is_ok, error_report

    # -- definitions ---------------------------------------------------

    def set_definition(self, key: str, value) -> None:
        """Record ``key`` in `[DEFINITIONS]`."""
        self._native.set_definition(key, value)

    def get_definition(self, key: str):
        """Return what ``key`` says, or ``''`` if it is not defined."""
        return self._native.definitions().get(key, "")

    @property
    def definitions(self) -> dict:
        """Everything `[DEFINITIONS]` will carry."""
        return self._native.definitions()

    def copy_definitions(self, other_seq: Sequence) -> None:
        """Take every definition ``other_seq`` carries."""
        for key, value in other_seq.definitions.items():
            self.set_definition(key, value)

    # -- extensions ----------------------------------------------------

    def get_extension_type_ID(self, extension_string: str) -> int:
        """Return the numeric id for ``extension_string``, minting one if new."""
        return self._native.extension_type_id(extension_string)

    def get_extension_type_string(self, extension_id: int) -> str:
        """Return the name ``extension_id`` stands for.

        Raises
        ------
        ValueError
            If no extension has been given that id.
        """
        name = self._native.extension_type_name(extension_id)
        if not name:
            raise ValueError(
                f"Extension for the given ID - {extension_id} - is unknown."
            )
        return name

    def set_extension_string_ID(self, extension_str: str, extension_id: int) -> None:
        """Pin ``extension_str`` to ``extension_id``."""
        self._native.set_extension_type_id(extension_str, extension_id)

    # -- TR ids --------------------------------------------------------

    def get_or_create_trid_id(self, label_name: str) -> int:
        """Return the TRID number ``label_name`` is known by, naming it if new.

        Parameters
        ----------
        label_name : str
            What this repetition is called.

        Returns
        -------
        int
            Its number, counting from 1 in the order names were first seen.
        """
        label_name = str(label_name)
        if not label_name:
            raise ValueError("TRID label_name must be a non-empty char/string.")
        if label_name not in self._trid_names:
            self._trid_names.append(label_name)
        return self._trid_names.index(label_name) + 1

    def add_trid(self, label_name: str) -> None:
        """Append a block setting TRID to the number ``label_name`` is known by."""
        if not getattr(self.system, "flag_trid", True):
            return
        from ._make_label import make_label

        self.add_block(
            make_label("TRID", "SET", float(self.get_or_create_trid_id(label_name)))
        )

    # -- files ---------------------------------------------------------

    def write(
        self,
        name,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = False,
        v141_compat: bool = False,
    ) -> str | None:
        """Write a Pulseq `.seq` file.

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Sign the file, so a reader can tell it has not been edited.
        remove_duplicates : bool, default True
            Collapse identical library rows first. The sequence held here is
            left as it is; what is written is the collapsed copy.
        check_timing : bool, default False
            Judge the timing first, and warn if anything is wrong.
        v141_compat : bool, default False
            Write 1.4.1 instead, for an interpreter that predates 1.5.

        Returns
        -------
        str or None
            The signature written, or None if the file is unsigned. It is
            the signature of what was written, so with ``remove_duplicates``
            it belongs to the collapsed copy rather than to this sequence.
        """
        if check_timing:
            is_ok, error_report = self.check_timing()
            if not is_ok:
                warn(
                    f"write(): {len(error_report)} timing errors found in the sequence",
                    stacklevel=2,
                )

        source = self.remove_duplicates() if remove_duplicates else self
        if v141_compat:
            return source.write_v141(name, create_signature)

        written = _cxx.write_text(source._native, create_signature)
        Path(name).write_bytes(written)
        return self._note_signature(written, "text")

    def write_binary(
        self,
        name,
        create_signature: bool = True,  # noqa: ARG002 -- the binary form is unsigned
    ) -> None:
        """Write the binary form, which a scanner parses faster.

        Parameters
        ----------
        name : str or Path
            Where to write it.
        create_signature : bool, default True
            Accepted for the text writer's signature; the binary format has
            no signature section, so a binary file is never signed.

        Returns
        -------
        None
            Always, since there is no signature to hand back.
        """
        Path(name).write_bytes(_cxx.write_binary(self._native))
        return None

    def write_v141(
        self, name, create_signature: bool = True, gamma=42576000.0, field=1.5
    ) -> str | None:
        """Write a Pulseq 1.4.1 file, for an interpreter that predates 1.5."""
        written = _cxx.write_text_v141(self._native, create_signature, gamma, field)
        Path(name).write_bytes(written)
        return self._note_signature(written, "text")

    def _note_signature(self, written: bytes, kind: str) -> str | None:
        """Record and return the signature ``written`` carries, if any."""
        found = _SIGNATURE.search(written.decode("ascii", "replace"))
        if found is None:
            self.signature_type = self.signature_file = self.signature_value = None
            return None
        self.signature_type = "md5"
        self.signature_file = kind
        self.signature_value = found.group(1)
        return self.signature_value

    def read(
        self,
        file_path,
        detect_rf_use: bool = False,
        remove_duplicates: bool = True,
        verify: bool = False,
    ) -> None:
        """Replace this sequence with the one in ``file_path``.

        Text or binary, told apart by what is in the bytes. Every revision
        from 1.2.0 is read, older ones by converting them.

        Parameters
        ----------
        file_path : str or Path
            The file to read.
        detect_rf_use : bool, default False
            Guess what each pulse is for in a file that does not say. Files
            from 1.5.0 on record it, and nothing older is guessed at here.
        remove_duplicates : bool, default True
            Collapse identical library rows after reading.
        verify : bool, default False
            Check the file against the signature it carries.
        """
        if detect_rf_use:
            warn(
                "read(): detect_rf_use is not supported; a pulse in a file "
                "that does not record what it is for is read as undefined",
                stacklevel=2,
            )
        self._native = _cxx.read(Path(file_path).read_bytes(), verify)
        if remove_duplicates:
            self._native.remove_duplicates()

    # -- collapsing ----------------------------------------------------

    def remove_duplicates(self, in_place: bool = False) -> Sequence:
        """Collapse identical library rows and renumber what points at them.

        Parameters
        ----------
        in_place : bool, default False
            Collapse this sequence. Otherwise a copy is collapsed and this
            one is left as it is.

        Returns
        -------
        Sequence
            The collapsed sequence: this one, or the copy.
        """
        target = self if in_place else self._copy()
        target._native.remove_duplicates()
        return target

    def _copy(self) -> Sequence:
        """Return a sequence holding everything this one holds.

        The core has no copy of its own, so the copy goes through the binary
        form: it carries every library, the block table and the definitions,
        and reading it back is a compiled pass.
        """
        other = Sequence(self.system)
        other._native = _cxx.read(_cxx.write_binary(self._native), False)
        other._use_block_cache = self._use_block_cache
        other._use_event_cache = self._use_event_cache
        other._trid_names = list(self._trid_names)
        return other

    # -- aliases -------------------------------------------------------
    #
    # Names the toolboxes carry for what is above, so a design script that
    # spells a call the other way round still runs.

    def write_file(self, filename) -> None:
        """`write(filename, create_signature=False)`."""
        self.write(filename, create_signature=False)

    def read_binary(self, filename) -> None:
        """`read(filename)`, which tells text and binary apart by itself."""
        self.read(filename)

    # -- no-ops --------------------------------------------------------
    #
    # The toolboxes cache decompressed blocks and registration results in
    # Python, and a design script turns that off or clears it to bound its
    # memory. Nothing here caches: a block is decoded in C++ on the way out
    # and an event carries the ids it was registered under. So the flags are
    # remembered and never read, the sizes are zero, and clearing is
    # nothing to do.

    @property
    def use_block_cache(self) -> bool:
        """What was asked for. Nothing here caches decompressed blocks."""
        return self._use_block_cache

    @use_block_cache.setter
    def use_block_cache(self, enabled: bool) -> None:
        self._use_block_cache = enabled

    @property
    def use_event_cache(self) -> bool:
        """What was asked for. Nothing here caches registration results."""
        return self._use_event_cache

    @use_event_cache.setter
    def use_event_cache(self, enabled: bool) -> None:
        self._use_event_cache = enabled

    @property
    def block_cache_size(self) -> int:
        """Zero: no block is held decompressed."""
        return 0

    @property
    def event_cache_size(self) -> int:
        """Zero: no registration result is held."""
        return 0

    def clear_block_cache(self) -> None:
        """Nothing to clear."""

    def clear_event_cache(self) -> None:
        """Nothing to clear."""

    def clear_caches(self) -> None:
        """Nothing to clear."""
