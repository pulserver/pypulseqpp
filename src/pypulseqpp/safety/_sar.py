"""Time-averaged SAR from virtual observation points, one repetition at a time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from .. import _ext as _cxx


class VopModel(NamedTuple):
    """Virtual observation points, and optional global SAR matrices.

    ``vops`` is ``(N, Nc, Nc)``, complex Hermitian, in W/kg per unit channel
    drive squared: SAR is ``v^H Q v`` for the channel drive phasors ``v`` at
    peak amplitude. ``global_matrix`` is ``(Nc, Nc)``, or ``(B, Nc, Nc)`` with
    one matrix per body model as a population's file carries; the global SAR of
    a window is the largest over the models. ``metadata`` is what the file says
    about itself, such as the coil, the drive unit, the channel order and the
    ``safety_factor`` every local SAR is multiplied by. ``cores`` is
    ``(N, Nc, Nc)``: the averaged SAR matrix each point was compressed from,
    which the point dominates. The cores bound local SAR from below and the
    points from above, so a ratio of SAR against a reference needs both; for
    matrices that were never compressed, the cores are the points themselves.
    """

    vops: np.ndarray
    global_matrix: np.ndarray | None = None
    metadata: dict | None = None
    cores: np.ndarray | None = None


#: Variable names a VOP file may use, and a global SAR matrix's.
_VOP_NAMES = ("VOP", "VOPm", "vops", "VOPs", "VOP_matrices", "Q10g", "Q")
_GLOBAL_NAMES = ("Sglobal", "global_matrix", "Qglobal", "Q_global")


def _hermitian(matrices: np.ndarray, what: str) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=complex)
    conjugate = np.conj(np.swapaxes(matrices, -1, -2))
    scale = max(float(np.abs(matrices).max(initial=0.0)), 1e-300)
    if not np.allclose(matrices, conjugate, rtol=0.0, atol=1e-6 * scale):
        raise ValueError(f"{what} must be Hermitian")
    return 0.5 * (matrices + conjugate)


def _stacked(matrices: np.ndarray, what: str, *, column_major: bool) -> np.ndarray:
    """Read a stack of square matrices as ``(K, Nc, Nc)``, whichever axis counts them.

    A single matrix becomes a stack of one. Of a three-axis array, the two
    axes of equal length are the channels and the odd one counts the matrices,
    so both ``(K, Nc, Nc)`` and MATLAB's ``(Nc, Nc, K)`` are read. Where all
    three are equal the layout is ambiguous, and ``column_major`` -- true for a
    ``.mat`` file, false for an ``.npz`` -- decides.
    """
    matrices = np.asarray(matrices)
    if matrices.ndim == 2 and matrices.shape[0] == matrices.shape[1]:
        return matrices[None]
    if matrices.ndim == 3:
        count, rows, columns = matrices.shape
        if rows == columns and not (column_major and count == rows):
            return matrices
        if count == rows:
            return np.transpose(matrices, (2, 0, 1))
    raise ValueError(f"{what} are a stack of square matrices, not {matrices.shape}")


def _validated(model: VopModel) -> VopModel:
    vops = np.asarray(model.vops)
    if vops.ndim != 3 or vops.shape[1] != vops.shape[2] or vops.shape[0] == 0:
        raise ValueError(f"VOPs are (N, Nc, Nc), not {vops.shape}")
    vops = _hermitian(vops, "every VOP")
    cores = model.cores
    if cores is not None:
        cores = np.asarray(cores)
        if cores.shape != vops.shape:
            raise ValueError(f"the cores are {cores.shape} and the VOPs {vops.shape}")
        cores = _hermitian(cores, "every core")
        scale = max(float(np.abs(vops).max(initial=0.0)), 1e-300)
        if float(np.linalg.eigvalsh(vops - cores)[:, 0].min()) < -1e-9 * scale:
            raise ValueError("a VOP does not dominate its core")
    _safety_factor(model.metadata, None)
    global_matrix = model.global_matrix
    if global_matrix is not None:
        one = np.asarray(global_matrix).ndim == 2
        global_matrix = _stacked(
            global_matrix, "the global SAR matrices", column_major=False
        )
        if global_matrix.shape[1:] != vops.shape[1:]:
            raise ValueError(
                f"a global SAR matrix is {global_matrix.shape[1:]}, and the VOPs "
                f"describe {vops.shape[1]} channels"
            )
        global_matrix = _hermitian(global_matrix, "every global SAR matrix")
        global_matrix = global_matrix[0] if one else global_matrix
    return VopModel(vops, global_matrix, model.metadata, cores)


def _safety_factor(metadata: dict | None, given: float | None) -> float:
    """Return the factor on local SAR: the one given, else the file's, else 1."""
    factor = given
    if factor is None:
        factor = (metadata or {}).get("safety_factor", 1.0)
    factor = float(factor)
    if not (np.isfinite(factor) and factor >= 1.0):
        raise ValueError(f"a safety factor is at least 1, not {factor}")
    return factor


def read_vops(path: str | os.PathLike) -> VopModel:
    """Read VOPs and the global SAR matrices from a ``.mat`` or ``.npz`` file.

    The VOPs are the array named ``VOP``, ``VOPm``, ``vops``, ``VOPs``,
    ``VOP_matrices``, ``Q10g`` or ``Q``, a stack of one ``Nc`` by ``Nc`` matrix
    per point: MARIE's ``(N, Nc, Nc)`` and MATLAB's ``(Nc, Nc, N)`` are both
    read, and a lone matrix is read as one point. The global SAR matrices are
    the array named ``Sglobal``, ``global_matrix``, ``Qglobal`` or
    ``Q_global``, a single matrix or one per body model as mariepy's
    ``vop.write`` stores a population's. The cores are the array named
    ``cores``, laid out as the VOPs are. An ``.npz`` file's ``metadata``, a
    JSON string, is read onto the model; its ``safety_factor`` multiplies every
    local SAR. MATLAB v7.3 (HDF5) files are not read.

    Parameters
    ----------
    path : str | os.PathLike
        The ``.mat`` or ``.npz`` file to read.

    Returns
    -------
    VopModel
        The VOPs and, where the file carries them, the global SAR matrices and
        the metadata, for :func:`~pypulseqpp.safety.check_sar` to take as its
        ``model``.

    Raises
    ------
    ValueError
        If the file holds no recognised VOP array, or one whose shape is not a
        stack of square matrices, a VOP does not dominate its core, or the
        safety factor is below 1.
    OSError
        If the file cannot be read, a MATLAB v7.3 file among them.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as held:
            names = set(held.files)
            name = next((name for name in _VOP_NAMES if name in names), None)
            if name is None:
                raise ValueError(f"{path} holds none of {', '.join(_VOP_NAMES)}")
            vops = _stacked(held[name], "the VOPs", column_major=False)
            global_name = next((one for one in _GLOBAL_NAMES if one in names), None)
            global_matrix = None if global_name is None else held[global_name]
            cores = (
                _stacked(held["cores"], "the cores", column_major=False)
                if "cores" in names
                else None
            )
            metadata = (
                json.loads(str(held["metadata"])) if "metadata" in names else None
            )
        return _validated(VopModel(vops, global_matrix, metadata, cores))

    from scipy.io import loadmat

    try:
        held = loadmat(path)
    except NotImplementedError as exc:
        raise ValueError(
            f"{path} is a MATLAB v7.3 file; save it with -v7, or as .npz"
        ) from exc
    name = next((name for name in _VOP_NAMES if name in held), None)
    if name is None:
        raise ValueError(f"{path} holds none of {', '.join(_VOP_NAMES)}")
    vops = _stacked(held[name], "the VOPs", column_major=True)
    global_name = next((name for name in _GLOBAL_NAMES if name in held), None)
    global_matrix = None
    if global_name is not None:
        whole = np.asarray(held[global_name])
        global_matrix = (
            whole
            if whole.ndim == 2
            else _stacked(whole, "the global SAR matrices", column_major=True)
        )
    cores = (
        _stacked(held["cores"], "the cores", column_major=True)
        if "cores" in held
        else None
    )
    return _validated(VopModel(vops, global_matrix, cores=cores))


def example_vops(num_channels: int = 8) -> SimpleNamespace:
    """Build a synthetic VOP model, for demonstrations and tests only.

    An ``num_channels``-loop array around a uniform cylinder at 3 T, in a 2D
    quasi-static model: each loop is a pair of opposed axial wires, and the
    electric field is ``-j omega A_z`` of their vector potential. Local SAR
    matrices are averaged over 12 mm disks on a coarse grid and used as the
    VOPs uncompressed; the global matrix is averaged over the whole cylinder.

    No tissue, no coil coupling and no conservative field are modelled, so the
    numbers are plausible in scale and nothing more.

    Parameters
    ----------
    num_channels : int, default=8
        Transmit channels the synthetic array has.

    Returns
    -------
    SimpleNamespace
        ``model``, a :class:`VopModel` in W/kg per V^2, uncompressed so that
        its cores are its points; ``drive_per_hz``, the
        drive in volts per channel that gives 1 Hz of B1+ at the centre in
        circular polarisation; and ``cp_shim``, that polarisation's weights.
    """
    mu0, gamma = 4e-7 * np.pi, 42.576e6
    omega = 2 * np.pi * 128e6
    radius, coil_radius = 0.1, 0.14
    conductivity, density, impedance = 0.6, 1000.0, 50.0
    grid, average_radius = 4e-3, 12e-3

    angles = 2 * np.pi * np.arange(num_channels) / num_channels
    half = 0.6 * np.pi / num_channels
    plus = coil_radius * np.exp(1j * (angles - half))
    minus = coil_radius * np.exp(1j * (angles + half))

    axis = np.arange(-radius, radius + grid / 2, grid)
    xx, yy = np.meshgrid(axis, axis)
    points = (xx + 1j * yy)[np.abs(xx + 1j * yy) < radius]

    # E_z per volt on each channel, (Nc, P).
    potential = -(mu0 / (2 * np.pi)) * (
        np.log(np.abs(points[None, :] - plus[:, None]))
        - np.log(np.abs(points[None, :] - minus[:, None]))
    )
    field = -1j * omega * potential / impedance
    factor = conductivity / (2 * density)

    centres = points[
        (
            np.abs(
                np.round(points.real / average_radius) * average_radius - points.real
            )
            < grid / 2
        )
        & (
            np.abs(
                np.round(points.imag / average_radius) * average_radius - points.imag
            )
            < grid / 2
        )
    ]
    vops = []
    for centre in centres:
        near = field[:, np.abs(points - centre) <= average_radius]
        vops.append(factor * (near @ near.conj().T) / near.shape[1])
    global_matrix = factor * (field @ field.conj().T) / field.shape[1]

    # B1+ at the centre per volt: B of each wire is mu0 I / (2 pi |p|^2) (p_y, -p_x).
    def b1_plus(wire, sign):
        current = sign / impedance
        b = (
            mu0
            * current
            / (2 * np.pi * np.abs(wire) ** 2)
            * (wire.imag - 1j * wire.real)
        )
        return 0.5 * b  # (Bx + i By) / 2 with B written as Bx + i By

    per_channel = b1_plus(plus, 1.0) + b1_plus(minus, -1.0)
    candidates = [np.exp(1j * angles), np.exp(-1j * angles)]
    cp_shim = max(candidates, key=lambda shim: abs(np.sum(shim * per_channel)))
    b1_per_volt = abs(np.sum(cp_shim * per_channel))

    return SimpleNamespace(
        model=_validated(
            VopModel(np.asarray(vops), global_matrix, cores=np.asarray(vops))
        ),
        drive_per_hz=1.0 / (gamma * b1_per_volt),
        cp_shim=cp_shim,
    )


def _evaluate(seq, matrices, global_matrix, drive, shim, reference=None):
    size, start = seq.repetition() if seq.num_blocks else (0, 0)
    found = _cxx.vop_sar(
        seq._native,
        vops=np.ascontiguousarray(matrices),
        global_matrix=None
        if global_matrix is None
        else np.ascontiguousarray(global_matrix),
        drive=np.ascontiguousarray(drive),
        default_shim=np.ascontiguousarray(shim),
        size=size,
        start=start,
        reference=None if reference is None else np.ascontiguousarray(reference),
    )
    return size, start, found


def _bodies(global_matrix: np.ndarray) -> np.ndarray:
    """Return the global matrices as a stack, one per body model."""
    return global_matrix[None] if global_matrix.ndim == 2 else global_matrix


def _per_body(seq, model, drive, shim, found) -> np.ndarray:
    """Return each body model's global SAR in the window of largest global SAR."""
    if model.global_matrix.ndim == 2:
        k = int(np.argmax(found["global"]))
        return np.array([found["global"][k]])
    _, _, each = _evaluate(seq, model.global_matrix, None, drive, shim)
    return np.asarray(each["worst"], dtype=float)


_NO_FLOOR = (
    "a ratio against a reference needs a lower bound on the reference's local "
    "SAR: a model with the cores its VOPs were compressed from, or matrices "
    "that were never compressed"
)


def _reference_of(reference, model, drive, shim) -> SimpleNamespace:
    """Return what a sequence is compared with: the reference's SAR and durations.

    ``peak`` is the cores' local SAR, a lower bound on the reference's true peak
    local SAR, in the window where it is largest, of ``duration``.
    ``global_sar`` is each body model's global SAR in the window of largest
    global SAR, of ``global_duration``, or None without global matrices.
    """
    if hasattr(reference, "worst_local"):
        worst = reference.worst_local
        if worst is None:
            raise ValueError("the reference report has no window to compare with")
        if worst.floor is None:
            raise ValueError(_NO_FLOOR + "; pass the reference sequence instead")
        whole = reference.worst_global
        return SimpleNamespace(
            peak=float(worst.floor),
            duration=float(reference.windows.duration[worst.window]),
            global_sar=None if whole is None else np.asarray(whole.per_body),
            global_duration=None
            if whole is None
            else float(reference.windows.duration[whole.window]),
        )
    if model.cores is None:
        raise ValueError(_NO_FLOOR)
    _, _, found = _evaluate(reference, model.cores, model.global_matrix, drive, shim)
    if not found["first"].size:
        raise ValueError("the reference sequence plays nothing to compare with")
    window = int(np.argmax(found["local"]))
    against = SimpleNamespace(
        peak=float(found["local"][window]),
        duration=float(found["duration"][window]),
        global_sar=None,
        global_duration=None,
    )
    if model.global_matrix is not None:
        against.global_sar = _per_body(reference, model, drive, shim, found)
        against.global_duration = float(
            found["duration"][int(np.argmax(found["global"]))]
        )
    return against


def check_sar(
    seq,
    model: VopModel,
    *,
    drive_per_hz,
    local_limit: float = 10.0,
    global_limit: float = 3.2,
    default_shim=None,
    reference=None,
    safety_factor: float | None = None,
) -> tuple[bool, SimpleNamespace]:
    """Check window-averaged local and global SAR against their limits.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    model : VopModel
        VOPs and optional global matrices, in W/kg per unit drive squared; see
        :func:`read_vops` and :func:`example_vops`. With one global matrix per
        body model, a window's global SAR is the largest over the models.
    drive_per_hz : float or array_like
        Channel drive per Hz of RF amplitude, in the VOPs' drive unit: one
        value, or one per channel.
    local_limit : float, default=10.0
        W/kg; IEC 60601-2-33 normal-mode head limits by default.
    global_limit : float, default=3.2
        W/kg; IEC 60601-2-33 normal-mode head limits by default.
    default_shim : array_like, default=None
        Channel weights of a single-channel pulse played without an RF shim;
        all ones by default.
    reference : Sequence or report, default=None
        What to compare with, in the same model and calibration: a sequence,
        such as a CP-mode FID, evaluated here through the model's cores with
        the same drive and default shim; or the report of an earlier call made
        with a model whose cores are its VOPs. Its window of largest local SAR
        gives the reference's peak local SAR, global SAR and duration.
    safety_factor : float, default=None
        Factor, at least 1, every local SAR is multiplied by; the model's
        ``metadata["safety_factor"]`` when None, and 1 when it has none.

    Returns
    -------
    is_ok : bool
        True when every window's largest VOP SAR is within ``local_limit`` and,
        with a global matrix, its global SAR within ``global_limit``.
    report : SimpleNamespace
        The limits and ``safety_factor``; ``tr_size`` and ``tr_start``
        (blocks); ``windows``, arrays ``first``, ``last`` (1-based blocks),
        ``duration`` (s), ``local_sar``, ``vop``, ``global_sar`` (W/kg) and
        ``global_body``, and with a reference ``reference_ratio``;
        ``worst_local`` and ``worst_global``, each the ``sar``, ``window`` and
        its ``first`` and ``last`` block, or None (``worst_local`` also its
        ``vop``, ``per_vop``, the SAR of every VOP in that window, and
        ``floor``, a lower bound on that window's true peak local SAR, or None
        when the model's cores are not its VOPs; ``worst_global`` its
        ``body`` and ``per_body``, every body model's global SAR in that
        window); and ``reference``, or None without one.

        ``reference`` carries ``sar_ratio``, the largest over windows of the
        window's local SAR over the reference's peak local SAR, with its
        ``vop`` and ``window``; ``energy_ratio``, the largest of that ratio
        times the window's duration over the reference's, with its
        ``energy_window``; and ``global_sar_ratio`` and ``global_energy_ratio``
        alike for the global matrices, each body model's SAR over the
        reference's in the same body model, or None. Every window counts, a
        shortened last window included.

    Raises
    ------
    ValueError
        If a reference sequence plays nothing to compare with, the model has
        no cores to bound the reference's local SAR from below, a shim weighs
        a different number of channels than the model has, or the safety
        factor is below 1.

    Notes
    -----
    SAR is averaged over each window: each repetition
    :meth:`~pypulseqpp.Sequence.repetition` returns, from block 1, and the
    blocks after the last full repetition as a shorter last window; or the
    whole sequence when it does not repeat.
    Every window is compared with the limits. A pulse drives channel ``c`` with
    ``drive_c * s_c * b_c(t)``: ``b`` its waveform in Hz, resampled every
    microsecond as :func:`pypulseqpp.calc_rf_power` does (one channel's waveform
    on every channel for a single-channel pulse), and ``s`` its block's RF shim,
    ``default_shim`` or ones.

    A window's local SAR is the largest over the VOPs, an upper bound on its
    peak local SAR, times the safety factor. The reference's is the largest
    over the cores, a lower bound, so ``sar_ratio`` is never below the ratio of
    the two true peaks. The scale of ``drive_per_hz`` and of the matrices
    cancels in the reference ratios; relative channel gains do not, and the
    safety factor, which multiplies only the sequence's side, does not. With a
    reference lasting its minimum TR, ``energy_ratio`` scales that minimum TR
    to this sequence's repetition, at the energy each repetition deposits.
    """
    model = _validated(model)
    factor = _safety_factor(model.metadata, safety_factor)
    channels = model.vops.shape[1]
    drive = np.broadcast_to(np.asarray(drive_per_hz, dtype=float), (channels,))
    shim = (
        np.ones(channels, dtype=complex)
        if default_shim is None
        else np.asarray(default_shim, dtype=complex).ravel()
    )
    if shim.size != channels:
        raise ValueError(f"default_shim weighs {shim.size} channels, not {channels}")

    against = (
        None if reference is None else _reference_of(reference, model, drive, shim)
    )
    size, start, found = _evaluate(
        seq, factor * model.vops, model.global_matrix, drive, shim
    )
    windows = SimpleNamespace(
        first=found["first"],
        last=found["last"],
        duration=found["duration"],
        local_sar=found["local"],
        vop=found["vop"],
        global_sar=found["global"] if model.global_matrix is not None else None,
        global_body=found["global_body"] if model.global_matrix is not None else None,
        reference_ratio=None if against is None else found["local"] / against.peak,
    )

    exact = model.cores is not None and np.array_equal(model.cores, model.vops)
    worst_local = worst_global = compared = None
    if windows.first.size:
        k = int(np.argmax(windows.local_sar))
        worst_local = SimpleNamespace(
            sar=float(windows.local_sar[k]),
            vop=int(windows.vop[k]),
            window=k,
            first=int(windows.first[k]),
            last=int(windows.last[k]),
            per_vop=np.asarray(found["worst"], dtype=float),
            floor=float(windows.local_sar[k]) / factor if exact else None,
        )
        if windows.global_sar is not None:
            k = int(np.argmax(windows.global_sar))
            worst_global = SimpleNamespace(
                sar=float(windows.global_sar[k]),
                body=int(windows.global_body[k]),
                window=k,
                first=int(windows.first[k]),
                last=int(windows.last[k]),
                per_body=_per_body(seq, model, drive, shim, found),
            )
        if against is not None:
            whole = None
            if windows.global_sar is not None and against.global_sar is not None:
                _, _, each = _evaluate(
                    seq,
                    _bodies(model.global_matrix),
                    None,
                    drive,
                    shim,
                    reference=against.global_sar,
                )
                whole = each["ratio"]
            compared = _compared(windows, against, whole)

    report = SimpleNamespace(
        local_limit=local_limit,
        global_limit=global_limit,
        safety_factor=factor,
        tr_size=size,
        tr_start=start,
        windows=windows,
        worst_local=worst_local,
        worst_global=worst_global,
        reference=compared,
    )
    is_ok = (worst_local is None or worst_local.sar <= local_limit) and (
        worst_global is None or worst_global.sar <= global_limit
    )
    return is_ok, report


def _compared(windows, against, whole) -> SimpleNamespace:
    ratio = windows.reference_ratio
    k = int(np.argmax(ratio))
    energy = ratio * windows.duration / against.duration
    j = int(np.argmax(energy))
    global_sar_ratio = global_energy_ratio = None
    if whole is not None:
        global_sar_ratio = float(whole.max())
        global_energy_ratio = float(
            (whole * windows.duration / against.global_duration).max()
        )
    return SimpleNamespace(
        sar_ratio=float(ratio[k]),
        vop=int(windows.vop[k]),
        window=k,
        energy_ratio=float(energy[j]),
        energy_window=j,
        global_sar_ratio=global_sar_ratio,
        global_energy_ratio=global_energy_ratio,
    )
