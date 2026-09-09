"""Moving, turning and resizing the imaging volume of a designed sequence.

Two callers want this and they want different halves of it.

A **design module** uses it to put itself somewhere: a fat-saturation band on
an oblique slab is built at its own position and orientation and then flagged
`NOPOS`/`NOROT`, so the prescription that arrives later moves everything else
and leaves it alone.

An **interpreter** uses it to apply the prescription: the offsets it hands the
scanner alongside the block table, the rotation the scanner is told to apply,
and the trajectory the reconstructor is given so it can form the receive phase
itself -- and form it again, for a pose update, without the sequence being
touched.

Everything below is a thin shell. The arithmetic is in `pulseq/fov.cpp`, which
is where it has to be: a shift is a walk over every block, and a prescription
arrives once per scan.

**A shift is a phase, and the phase is `dr . k`.** Written in the logical
frame -- the frame the gradients were designed in, which is what `(1, 0, 0)`
means when a scanner prescribes a unit offset along the reconstructed image's
x. Rotating `dr` and `k` together leaves their product alone, so this needs to
know nothing about the orientation the scanner applies, nor about the motion
correction composed into it.

**A rotation is an annotation.** It is attached to each block as a `ROTATIONS`
extension, never baked into new waveforms: baking costs one waveform per
orientation, where attaching costs four numbers and lets a thousand
orientations share the trajectory they were designed from.

**A scale is one multiplication**, because a gradient is a normalised shape
beside a single amplitude.
"""

from __future__ import annotations

import numpy as np

from . import _ext as _cxx

__all__ = ["TransformFOV"]


def _quaternion_of(rotation):
    """Return a rotation as four numbers, scalar first, however it arrives."""
    if hasattr(rotation, "as_quat"):
        return np.asarray(rotation.as_quat(canonical=True, scalar_first=True), float)
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"TransformFOV(): `rotation` must be (3, 3) or a rotation object, "
            f"got {matrix.shape}"
        )
    from scipy.spatial.transform import Rotation

    return np.asarray(
        Rotation.from_matrix(matrix).as_quat(canonical=True, scalar_first=True), float
    )


def _runs_not_exempt(seq, label, first, last):
    """Return the stretches of ``first..last`` that ``label`` does not exempt.

    Pulseq's exemption flags are sticky: a block that sets one is itself
    exempt, and so is every block after it until another block clears it. A
    module that places itself -- a saturation band on its own slab -- sets one
    on its way in and clears it on its way out, and what falls between is what
    a later prescription must leave alone.

    The walk is the compiled one: a block carrying no label costs a column
    read, which is nearly all of them.
    """
    found = _cxx.evaluate_labels(
        seq._native, evolution="blocks", first_block=first, last_block=last
    )
    stop = len(seq) if last == 0 else last
    exempt = found.get(label)
    if exempt is None:
        return [(first, stop)]

    runs = []
    at = None
    for offset, flag in enumerate(np.atleast_1d(exempt)):
        block = first + offset
        if flag:
            if at is not None:
                runs.append((at, block - 1))
                at = None
        elif at is None:
            at = block
    if at is not None:
        runs.append((at, stop))
    return runs


class TransformFOV:
    """A prescription: where the volume sits, which way it faces, how big it is.

    Parameters
    ----------
    rotation : array_like or Rotation, optional
        A 3x3 matrix or a SciPy rotation. Applied *after* whatever a block
        already carries, so a module that placed itself keeps its own
        orientation inside the prescription's.
    translation : sequence of float, optional
        The offset in logical metres -- the frame the gradients were designed
        in. A prescribed physical offset is turned once by the caller,
        ``dr_logical = R.T @ dr_physical``, and never again.
    scale : sequence of float, optional
        Per logical axis. The field of view is divided by these, so a
        gradient is multiplied by them; zero silences an axis.
    transform : array_like, optional
        A 4x4 homogeneous matrix, instead of ``rotation`` and ``translation``.
    use_rotation_extension : bool, default True
        False would bake the rotation into new waveforms, which is not
        offered.
    system : Opts, optional
        What the result is checked against.

    Attributes
    ----------
    block_k_origin : tuple of float
        Where the trajectory stands entering the next range this transforms,
        carried so a scan too large to hold at once can be moved a piece at a
        time. Nothing says a repeating unit begins with an excitation, so a
        piece cannot work this out for itself.
    """

    def __init__(
        self,
        *,
        rotation=None,
        translation=None,
        scale=None,
        transform=None,
        use_rotation_extension: bool = True,
        system=None,
    ) -> None:
        if transform is not None:
            if rotation is not None or translation is not None:
                raise ValueError(
                    "TransformFOV(): `transform` already carries a rotation and a "
                    "translation; give it or give those two, not both"
                )
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(
                    f"TransformFOV(): `transform` must be (4, 4), got {matrix.shape}"
                )
            rotation = matrix[:3, :3]
            # A homogeneous matrix states its offset in the frame it rotates
            # into; everything here works in the logical one.
            translation = tuple(np.asarray(rotation, dtype=float).T @ matrix[:3, 3])
        elif rotation is None and translation is None and scale is None:
            raise ValueError(
                "TransformFOV(): give at least one of `rotation`, `translation`, "
                "`scale` or `transform`"
            )

        if not use_rotation_extension:
            raise NotImplementedError(
                "TransformFOV(use_rotation_extension=False) would bake the rotation "
                "into new gradient waveforms, one per orientation, which is not "
                "offered here: a rotation is an annotation, and the point of it is "
                "that a thousand orientations share one trajectory"
            )

        self.quaternion = None if rotation is None else _quaternion_of(rotation)
        self.translation = (
            None if translation is None else tuple(float(v) for v in translation)
        )
        self.scale = None if scale is None else tuple(float(v) for v in scale)
        self.use_rotation_extension = use_rotation_extension
        self.system = system
        self.block_k_origin = (0.0, 0.0, 0.0)

        for name, value in (("translation", self.translation), ("scale", self.scale)):
            if value is not None and len(value) != 3:
                raise ValueError(
                    f"TransformFOV(): `{name}` must be three numbers, got {len(value)}"
                )

    def apply_to_sequence(
        self, seq, *, time_range=None, block_range=None, in_place: bool = False
    ):
        """Apply this prescription to ``seq``.

        Parameters
        ----------
        seq : Sequence
            The sequence to transform.
        time_range : list of float, optional
            Two times in seconds; the blocks they touch are the ones changed.
        block_range : sequence of int, optional
            Two 1-based block indices, inclusive. Not with ``time_range``.

            Either way, only the blocks named are changed -- but everything
            before them still counts towards where the trajectory stands
            inside them, which is the only way a partial shift means
            anything.
        in_place : bool, default False
            Transform ``seq`` itself rather than a copy.

        Returns
        -------
        Sequence
            The transformed sequence: ``seq`` itself when ``in_place``.

        Notes
        -----
        Applied as scale, then rotate, then translate. Scaling changes the
        gradient amplitudes the translation's phase is read from, so it has
        to be in place before the translation runs.
        """
        target = seq if in_place else seq._copy()
        first, last = target._range_for(time_range, block_range)

        if self.scale is not None:
            for begins, ends in _runs_not_exempt(target, "NOSCL", first, last):
                _cxx.apply_fov_scale(
                    target._native, scale=self.scale, first=begins, last=ends
                )
        if self.quaternion is not None:
            for begins, ends in _runs_not_exempt(target, "NOROT", first, last):
                _cxx.apply_fov_rotation(
                    target._native,
                    quaternion=tuple(self.quaternion),
                    first=begins,
                    last=ends,
                )
        if self.translation is not None:
            # Where the trajectory stands is a fact about everything played
            # before it, exempt or not, so the walk runs over the whole range
            # and only what it writes is gated.
            for begins, ends in _runs_not_exempt(target, "NOPOS", first, last):
                self.block_k_origin = _cxx.apply_fov_shift(
                    target._native,
                    shift=self.translation,
                    first=begins,
                    last=ends,
                    carry=self.block_k_origin,
                )
        return target

    #: The reference toolbox's name for the same thing.
    apply_to_seq = apply_to_sequence

    def trajectories(self, seq, *, block_range=None):
        """Where each readout samples k, per axis, in 1/m.

        What a reconstructor is handed instead of a phase: given the
        trajectory it forms ``dr . k`` itself, so a prescription can change --
        a new offset, a pose update from motion correction -- without the
        sequence being built again. It is also the array the metadata an
        acquisition is enriched with wants, so it is one thing serving two.

        In the logical frame; a block's own rotation is the caller's to apply,
        and a caller that turns the trajectory usually turns the shift with
        it, which changes nothing.

        Returns
        -------
        list of (int, np.ndarray)
            The 1-based block, and a 3-by-n array of where its samples sit.
            Blocks that do not acquire are not in the list.

        Notes
        -----
        Where the trajectory stands is accumulated from the start of the
        range in one walk, not worked out per readout: a readout's origin is
        a fact about everything played before it.
        """
        first, last = seq._range_for(None, block_range)
        origins = _cxx.block_k_origins(
            seq._native, first=first, last=last, carry=self.block_k_origin
        )["origins"]
        stop = len(seq) if last == 0 else last

        found = []
        for block in range(first, stop + 1):
            sampled = _cxx.absolute_trajectory(
                seq._native, block, origin=tuple(origins[block - first])
            )
            if sampled.shape[1]:
                found.append((block, sampled))
        return found
