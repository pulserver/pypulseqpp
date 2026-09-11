"""Time-averaged SAR from virtual observation points, one repetition at a time."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from .. import _ext as _cxx


class VopModel(NamedTuple):
    """Virtual observation points, and an optional global SAR matrix.

    ``vops`` is ``(N, Nc, Nc)`` and ``global_matrix`` ``(Nc, Nc)``, complex
    Hermitian, in W/kg per unit channel drive squared: SAR is ``v^H Q v`` for
    the channel drive phasors ``v`` at peak amplitude.
    """

    vops: np.ndarray
    global_matrix: np.ndarray | None = None


#: Variable names a VOP file may use, and a global SAR matrix's.
_VOP_NAMES = ("VOP", "vops", "VOPs", "Q10g", "Q")
_GLOBAL_NAMES = ("Sglobal", "global_matrix", "Qglobal", "Q_global")


def _hermitian(matrices: np.ndarray, what: str) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=complex)
    conjugate = np.conj(np.swapaxes(matrices, -1, -2))
    scale = max(float(np.abs(matrices).max(initial=0.0)), 1e-300)
    if not np.allclose(matrices, conjugate, rtol=0.0, atol=1e-6 * scale):
        raise ValueError(f"{what} must be Hermitian")
    return 0.5 * (matrices + conjugate)


def _validated(model: VopModel) -> VopModel:
    vops = np.asarray(model.vops)
    if vops.ndim != 3 or vops.shape[1] != vops.shape[2] or vops.shape[0] == 0:
        raise ValueError(f"VOPs are (N, Nc, Nc), not {vops.shape}")
    vops = _hermitian(vops, "every VOP")
    global_matrix = model.global_matrix
    if global_matrix is not None:
        global_matrix = np.asarray(global_matrix)
        if global_matrix.shape != vops.shape[1:]:
            raise ValueError(
                f"the global SAR matrix is {global_matrix.shape}, and the VOPs "
                f"describe {vops.shape[1]} channels"
            )
        global_matrix = _hermitian(global_matrix, "the global SAR matrix")
    return VopModel(vops, global_matrix)


def read_vops(path: str | os.PathLike) -> VopModel:
    """Read VOPs and a global SAR matrix from a ``.mat`` or ``.npz`` file.

    A ``.mat`` file holds MATLAB's ``(Nc, Nc, N)`` stack, named ``VOP``,
    ``vops``, ``Q10g`` or ``Q``, and optionally ``Sglobal`` or
    ``global_matrix``; the stack is read into ``(N, Nc, Nc)``. MATLAB v7.3
    (HDF5) files are not read. An ``.npz`` file holds ``vops`` as
    ``(N, Nc, Nc)`` and optionally ``global_matrix``.
    """
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as held:
            vops = held["vops"]
            global_matrix = held.get("global_matrix")
        return _validated(VopModel(vops, global_matrix))

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
    stack = np.asarray(held[name])
    if stack.ndim == 2:
        stack = stack[:, :, None]
    global_name = next((name for name in _GLOBAL_NAMES if name in held), None)
    global_matrix = None if global_name is None else np.asarray(held[global_name])
    return _validated(VopModel(np.transpose(stack, (2, 0, 1)), global_matrix))


def example_vops(num_channels: int = 8) -> SimpleNamespace:
    """Build a synthetic VOP model, for demonstrations and tests only.

    An ``num_channels``-loop array around a uniform cylinder at 3 T, in a 2D
    quasi-static model: each loop is a pair of opposed axial wires, and the
    electric field is ``-j omega A_z`` of their vector potential. Local SAR
    matrices are averaged over 12 mm disks on a coarse grid and used as the
    VOPs uncompressed; the global matrix is averaged over the whole cylinder.
    No tissue, no coil coupling and no conservative field are modelled, so the
    numbers are plausible in scale and nothing more.

    Returns
    -------
    SimpleNamespace
        ``model``, a :class:`VopModel` in W/kg per V^2; ``drive_per_hz``, the
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
        model=_validated(VopModel(np.asarray(vops), global_matrix)),
        drive_per_hz=1.0 / (gamma * b1_per_volt),
        cp_shim=cp_shim,
    )


def check_sar(
    seq,
    model: VopModel,
    *,
    drive_per_hz,
    local_limit: float = 10.0,
    global_limit: float = 3.2,
    default_shim=None,
) -> tuple[bool, SimpleNamespace]:
    """Return whether no repetition exceeds the local and global SAR limits.

    Parameters
    ----------
    seq : Sequence
        Sequence to check.
    model : VopModel
        VOPs and optional global matrix, in W/kg per unit drive squared; see
        :func:`read_vops` and :func:`example_vops`.
    drive_per_hz : float or array_like
        Channel drive per Hz of RF amplitude, in the VOPs' drive unit: one
        value, or one per channel.
    local_limit, global_limit : float
        W/kg; IEC 60601-2-33 normal-mode head limits by default.
    default_shim : array_like, optional
        Channel weights of a single-channel pulse played without an RF shim;
        all ones by default.

    Returns
    -------
    is_ok : bool
        True when every window's largest VOP SAR is within ``local_limit`` and,
        with a global matrix, its global SAR within ``global_limit``.
    report : SimpleNamespace
        The limits; ``tr_size`` and ``tr_start`` (blocks); ``windows``, arrays
        ``first``, ``last`` (1-based blocks), ``duration`` (s), ``local_sar``,
        ``vop`` and ``global_sar`` (W/kg); and ``worst_local`` and
        ``worst_global``, each the ``sar``, ``window`` and its ``first`` and
        ``last`` block (``worst_local`` also its ``vop``), or None.

    Notes
    -----
    SAR is averaged over each window: the blocks before the first full
    repetition, each repetition the block definitions repeat with, and any
    blocks after the last; or the whole sequence when it does not repeat. The
    worst real repetition decides. A pulse drives channel ``c`` with
    ``drive_c * s_c * b_c(t)``: ``b`` its waveform in Hz, resampled every
    microsecond as :func:`pypulseqpp.calc_rf_power` does (one channel's waveform
    on every channel for a single-channel pulse), and ``s`` its block's RF shim,
    ``default_shim`` or ones.
    """
    model = _validated(model)
    channels = model.vops.shape[1]
    drive = np.broadcast_to(np.asarray(drive_per_hz, dtype=float), (channels,))
    shim = (
        np.ones(channels, dtype=complex)
        if default_shim is None
        else np.asarray(default_shim, dtype=complex).ravel()
    )
    if shim.size != channels:
        raise ValueError(f"default_shim weighs {shim.size} channels, not {channels}")

    size, start = seq._detect_tr() if seq.num_blocks else (0, 0)
    found = _cxx.vop_sar(
        seq._native,
        vops=np.ascontiguousarray(model.vops),
        global_matrix=None
        if model.global_matrix is None
        else np.ascontiguousarray(model.global_matrix),
        drive=np.ascontiguousarray(drive),
        default_shim=np.ascontiguousarray(shim),
        size=size,
        start=start,
    )
    windows = SimpleNamespace(
        first=found["first"],
        last=found["last"],
        duration=found["duration"],
        local_sar=found["local"],
        vop=found["vop"],
        global_sar=found["global"] if model.global_matrix is not None else None,
    )

    worst_local = worst_global = None
    if windows.first.size:
        k = int(np.argmax(windows.local_sar))
        worst_local = SimpleNamespace(
            sar=float(windows.local_sar[k]),
            vop=int(windows.vop[k]),
            window=k,
            first=int(windows.first[k]),
            last=int(windows.last[k]),
        )
        if windows.global_sar is not None:
            k = int(np.argmax(windows.global_sar))
            worst_global = SimpleNamespace(
                sar=float(windows.global_sar[k]),
                window=k,
                first=int(windows.first[k]),
                last=int(windows.last[k]),
            )

    report = SimpleNamespace(
        local_limit=local_limit,
        global_limit=global_limit,
        tr_size=size,
        tr_start=start,
        windows=windows,
        worst_local=worst_local,
        worst_global=worst_global,
    )
    is_ok = (worst_local is None or worst_local.sar <= local_limit) and (
        worst_global is None or worst_global.sar <= global_limit
    )
    return is_ok, report
