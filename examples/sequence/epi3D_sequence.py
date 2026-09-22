"""3D gradient-echo EPI with skipped-CAIPI sampling."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


def caipi_shift(ry: int, rz: int) -> int:
    """Return the CAIPI shift whose lattice keeps its aliases furthest apart.

    The sampled lattice is spanned by ``(0, rz)`` and ``(ry, shift)``, in
    lines and partitions. Its aliases form the dual lattice, which in two
    dimensions is the same lattice turned and scaled, so the shift that
    lengthens the sampling vectors also spreads the aliases most. Shifts are
    compared on the lengths of their two shortest primitive vectors, so a tie
    on the shortest is settled by the next, and then by the smaller shift. The rule
    reproduces the patterns Stirnberg and Stöcker (Magn Reson Med 2021,
    doi:10.1002/mrm.28486) found best, such as 2x2z1, 3x2z1, 2x4z2 and 1x6z2.

    Parameters
    ----------
    ry : int
        Acceleration along the phase-encode axis.
    rz : int
        Acceleration along the partition axis, which is also the shifts to
        choose among.

    Returns
    -------
    int
        The shift, in partitions per ``ry`` lines.
    """

    def shortest(shift: int) -> list[int]:
        # One of each pair of opposite vectors, b > 0 or b == 0 < a, and no
        # multiple of a shorter one.
        return sorted(
            (b * ry) ** 2 + (a * rz + b * shift) ** 2
            for a in range(-rz, rz + 1)
            for b in range(rz + 1)
            if (b or a > 0) and math.gcd(a, b) == 1
        )[:2]

    return max(range(rz), key=lambda shift: (shortest(shift), -shift))


def train_lines(
    n: int, ry: int, n_shots: int, partial_fourier: float
) -> tuple[int, int]:
    """Return the first line of shot 0 and the lines per shot.

    Shot ``s`` reads lines ``start + (s + i * n_shots) * ry``. The lattice keeps
    every line ``i`` with ``(i - n // 2) % ry == 0``, so the centre line is
    always read. The shots end on the lattice's last line and cover the lines
    from ``n - round(partial_fourier * n)`` up; when the shots cannot share
    them evenly, the extra lines extend below that, or are dropped where
    there is no room.

    Parameters
    ----------
    n : int
        Phase-encode lines across the full field of view.
    ry : int
        Acceleration along the phase-encode axis.
    n_shots : int
        Shots the train is segmented into.
    partial_fourier : float
        Fraction of k-space acquired, from the far side of the centre.

    Returns
    -------
    start : int
        The first line of shot 0.
    etl : int
        Lines each shot reads.

    Raises
    ------
    ValueError
        If the lines cannot be shared among ``n_shots`` shots.
    """
    first = n - round(partial_fourier * n)
    lattice = [i for i in range(n) if (i - n // 2) % ry == 0]
    count = sum(1 for i in lattice if i >= first)
    etl = -(-count // n_shots)
    if lattice[-1] - (etl * n_shots - 1) * ry < 0:
        etl = count // n_shots
    if etl < 1:
        raise ValueError(
            f"{count} lines cannot be shared among {n_shots} shots; lower n_shots"
        )
    return lattice[-1] - (etl * n_shots - 1) * ry, etl


def shell_bases(n: int, rz: int, partial_fourier: float) -> list[int]:
    """Return the first partition of every shell, the centre one first.

    Shells of ``rz`` partitions end on the last partition and cover those
    from ``n - round(partial_fourier * n)`` up; a shell that would start below
    partition zero is dropped. They are played centre-out.

    Parameters
    ----------
    n : int
        Partitions across the full field of view.
    rz : int
        Partitions one shell spans, which is the partition acceleration.
    partial_fourier : float
        Fraction of k-space acquired, from the far side of the centre.

    Returns
    -------
    list of int
        The first partition of each shell, centre-out.

    Raises
    ------
    ValueError
        If the partitions cannot hold one shell.
    """
    kept = round(partial_fourier * n)
    count = -(-kept // rz)
    if n - count * rz < 0:
        count = kept // rz
    if count < 1:
        raise ValueError(f"{kept} partitions cannot hold one shell of {rz}")
    bases = [n - (m + 1) * rz for m in range(count)]
    return sorted(bases, key=lambda b: (abs(b + (rz - 1) / 2 - n // 2), b))


class Epi3DApp(sequences.SequenceApp):
    """3D gradient-echo EPI: one train per ``(shot, shell)``, skipped-CAIPI sampled.

    The views sampled form a CAIPIRINHA lattice holding the centre of
    k-space: line ``y`` is read when ``(y - n_y // 2) % ry == 0``, at the
    partitions ``z`` with ``(z - n_z // 2 - d * j) % rz == 0``, ``j`` being the
    line's lattice index and ``d`` the shift :func:`caipi_shift` picks. A
    shell is ``rz`` consecutive partitions, and each of its lines holds one
    lattice partition. Shot ``s`` of a shell reads every ``n_shots``-th
    lattice line from the ``s``-th, blipping to that line's partition
    (skipped-CAIPI, Stirnberg and Stöcker, Magn Reson Med 2021,
    doi:10.1002/mrm.28486); one shot per shell is blipped-CAIPI. A volume is
    every shot of every shell, shots outer and shells centre-out inner.

    Every shot reads :attr:`NAVIGATOR_LINES` centre lines without blips
    (``NAV``) before its train, every line carries ``REV`` for its read
    polarity, and shots are delayed by successive fractions of the echo
    spacing so the echo time grows smoothly across the lines; ``te`` is the
    echo time of the centre line. Acquisitions carry ``LIN``, ``PAR``,
    ``SEG`` (the shot) and ``REP``.

    Two prescans are linked ahead of the imaging (:meth:`prescans`): the
    ``calibration``, when undersampled, a Cartesian gradient echo over the
    central ``n_acs_y x n_acs_z`` rectangle (``REF``); and the ``reference``,
    one volume with the phase encode reversed (``SET = 1``), for distortion
    correction.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.epi3D_sequence(n_x=32, n_y=16, n_z=4, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    >>> seq.definitions["Matrix"], seq.definitions["Name"]
    ([32.0, 16.0, 4.0], 'epi_3d')
    """

    NAME = "epi_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: SLR design of the slab-selective pulse.
    PULSE_DURATION = 3e-3
    TIME_BW_PRODUCT = 4.0
    #: Duration of the nonselective hard pulse (s).
    HARD_PULSE_DURATION = 0.5e-3
    #: Fat methylene shift from water (ppm), converted against ``system.B0``
    #: when the spectral-spatial pulse is built.
    FAT_SHIFT_PPM = -3.4
    #: Centre lines read without blips at the start of every shot.
    NAVIGATOR_LINES = 3
    #: Dephasing left on the readout axis at the end of each shot, in cycles
    #: across one voxel.
    SPOILING_CYCLES = 4.0
    #: Digital output marking every volume under ``volume_output``, and how
    #: long it lasts (s); it never outlasts the excitation block it rides.
    OUTPUT_CHANNEL = "ext1"
    OUTPUT_DURATION = 1e-3

    def init_sequence(
        self,
        fov_x: float = 220e-3,
        fov_y: float = 220e-3,
        fov_z: float = 96e-3,
        n_x: int = 128,
        n_y: int = 128,
        n_z: int = 32,
        flip_angle_deg: float = 20.0,
        te: float | None = None,
        tr: float | None = None,
        n_frames: int = 1,
        readout_bandwidth_hz: float = 500e3,
        ry: int = 1,
        rz: int = 1,
        partial_fourier_y: float = 1.0,
        partial_fourier_z: float = 1.0,
        n_shots: int = 1,
        *,
        n_dummy: int = 2,
        excitation: str = "slab",
        readout_oversampling: float = 1.0,
        n_acs_y: int = 24,
        n_acs_z: int = 16,
        volume_output: bool = False,
    ) -> None:
        """Design the excitation, the trains, the shells and the shot order.

        Parameters
        ----------
        fov_x : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        fov_y : float, default=0.22
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        fov_z : float, default=0.096
            Field of view along the readout, the phase encode and the
            partition encode (m). The slab excited is ``fov_z`` thick.
        n_x : int, default=128
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_y : int, default=128
            Matrix size along the readout, the phase encode and the partition
            encode.
        n_z : int, default=32
            Matrix size along the readout, the phase encode and the partition
            encode.
        flip_angle_deg : float, default=20.0
            Excitation flip angle (degrees).
        te : float | None, default=None
            Echo time of the centre line (s). ``None`` is as short as the
            navigator and the train admit.
        tr : float | None, default=None
            Volume repetition time (s): every shot of every shell. ``None`` is
            as short as possible.
        n_frames : int, default=1
            Volumes in the time series, each carrying its ``REP`` counter.
        readout_bandwidth_hz : float, default=500000.0
            Requested receiver bandwidth (Hz).
        ry, rz : int, default=1
            Undersampling along the phase and the partition encode; ``rz`` is
            also the shell height.
        partial_fourier_y, partial_fourier_z : float, default=1.0
            Fraction of the phase- and partition-encode extent read, in
            ``[0.5, 1]``. Truncates the lines and the shells before the centre.
        n_shots : int, default=1
            Interleaved shots each shell's lines are split into.
        n_dummy : int, default=2
            Non-acquiring volumes before a time series; with one frame,
            non-acquiring shots.
        excitation : {'slab', 'nonselective', 'spsp'}, default='slab'
            A slab-selective SLR pulse, a hard pulse, or a slab- and
            water-selective spectral-spatial pulse.
        readout_oversampling : float, default=1.0
            Readout oversampling factor, at least one.
        n_acs_y : int, default=24
            Extent of the gradient-echo calibration rectangle along the phase
            and the partition encode.
        n_acs_z : int, default=16
            Extent of the gradient-echo calibration rectangle along the phase
            and the partition encode.
        volume_output : bool, default=False
            Play a digital output on :attr:`OUTPUT_CHANNEL` at the first
            excitation of every volume, dummy volumes included.

        Raises
        ------
        ValueError
            If ``excitation`` is unknown, a partial Fourier fraction is outside
            ``[0.5, 1]``, a count is below one, the shots cannot share the lines
            or the partitions hold no shell, or the TE or TR is shorter than
            the shots take.
        """
        if excitation not in sequences.EXCITATIONS:
            raise ValueError(
                f"excitation must be one of {sequences.EXCITATIONS}, got {excitation!r}"
            )
        for name, fraction in (
            ("partial_fourier_y", partial_fourier_y),
            ("partial_fourier_z", partial_fourier_z),
        ):
            if not 0.5 <= fraction <= 1.0:
                raise ValueError(f"{name} must lie in [0.5, 1], got {fraction}")
        for name, count in (
            ("ry", ry),
            ("rz", rz),
            ("n_shots", n_shots),
            ("n_frames", n_frames),
        ):
            if count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")

        system = self.system
        self.fov = (fov_x, fov_y, fov_z)
        self.matrix = (n_x, n_y, n_z)
        self.excitation = excitation
        self.n_frames, self.n_shots = n_frames, n_shots
        self.n_dummy, self.volume_output = n_dummy, volume_output
        self.raster = system.block_duration_raster
        self.exc = sequences.make_excitation(
            system,
            excitation,
            flip_angle_deg,
            fov_z,
            duration_s=self.PULSE_DURATION,
            time_bw_product=self.TIME_BW_PRODUCT,
            hard_duration_s=self.HARD_PULSE_DURATION,
            fat_shift_ppm=self.FAT_SHIFT_PPM,
        )
        self.gz = getattr(self.exc, "gz", None)

        # Shot s reads lattice lines s, s + n_shots, ...; the centre line is the
        # c-th, echoed at the (c / n_shots)-th line of the unshifted train.
        start, etl = train_lines(n_y, ry, n_shots, partial_fourier_y)
        self.origins = [start + s * ry for s in range(n_shots)]
        self.shells = shell_bases(n_z, rz, partial_fourier_z)
        self.shift = caipi_shift(ry, rz)
        steps = np.arange(etl) * n_shots * ry
        # Every shell starts at the same residue modulo rz, so a shot's offsets
        # within its shell depend on the shot alone.
        trains = {}
        self.shot_trains = []
        for s in range(n_shots):
            lines = start + (s + np.arange(etl) * n_shots) * ry
            lattice = n_z // 2 + self.shift * ((lines - n_y // 2) // ry)
            offsets = (lattice - self.shells[0]) % rz
            key = tuple(offsets)
            if key not in trains:
                trains[key] = sequences.EpiReadout3D(
                    system,
                    self.exc.rf,
                    self.gz,
                    order=np.column_stack((steps, offsets)),
                    te=te,
                    te_line=(n_y // 2 - start) // ry / n_shots,
                    navigator_lines=self.NAVIGATOR_LINES,
                    echo_shifts=n_shots,
                    fov=self.fov,
                    matrix=self.matrix,
                    oversampling=readout_oversampling,
                    readout_bandwidth_hz=readout_bandwidth_hz,
                    spoiling_cycles=self.SPOILING_CYCLES,
                    labels=("LIN", "PAR"),
                )
            self.shot_trains.append(trains[key])
        self.epi = self.shot_trains[0]
        self.echo_time = self.epi.echo_time

        # A shot is the train and a closing delay of at least one raster.
        self.volume = [(s, b) for s in range(n_shots) for b in self.shells]
        shot = self.epi.duration + self.raster
        self.pad = self.raster
        if tr is not None:
            per_shot = tr / len(self.volume)
            if per_shot < shot - 1e-9:
                raise ValueError(
                    f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
                    f"{len(self.volume) * shot * 1e3:.3f} ms the "
                    f"{len(self.volume)} shots of a volume take"
                )
            self.pad += pp.round_to_raster(per_shot - shot, self.raster)
        self.shot_duration = shot - self.raster + self.pad
        self.repetition_time = len(self.volume) * self.shot_duration
        n_dummy_shots = n_dummy * (len(self.volume) if n_frames > 1 else 1)
        self.dummies = [self.volume[i % len(self.volume)] for i in range(n_dummy_shots)]
        self.output = pp.make_digital_output_pulse(
            self.OUTPUT_CHANNEL,
            duration=min(
                self.OUTPUT_DURATION,
                pp.calc_duration(self.exc.rf, *([] if self.gz is None else [self.gz])),
            ),
            system=system,
        )

        # A gradient echo keeps EPI distortion out of the coil maps.
        undersampled = (
            ry > 1 or rz > 1 or partial_fourier_y < 1 or partial_fourier_z < 1
        )
        self.calibration = []
        if undersampled:
            low_y, low_z = n_y // 2 - n_acs_y // 2, n_z // 2 - n_acs_z // 2
            self.calibration = [
                (y, z)
                for y in range(max(low_y, 0), min(low_y + n_acs_y, n_y))
                for z in range(max(low_z, 0), min(low_z + n_acs_z, n_z))
            ]
        self.gre = None
        if self.calibration:
            self.gre = sequences.LineReadout3D(
                system,
                self.exc.rf,
                self.gz,
                fov=self.fov,
                matrix=self.matrix,
                oversampling=readout_oversampling,
                readout_bandwidth_hz=readout_bandwidth_hz,
                spoiling_cycles=self.SPOILING_CYCLES,
                labels=("LIN", "PAR"),
            )

    def prescans(self) -> dict:
        """Return ``calibration`` (when undersampled) and ``reference``.

        Returns
        -------
        dict
            One loop per prescan, in play order.
        """
        chain = {"calibration": self.calibrate} if self.gre is not None else {}
        return {**chain, "reference": self.reference}

    def calibrate(self) -> None:
        """Play the gradient-echo calibration over the central rectangle."""
        for view in self.calibration:
            self.calibration_kernel(view)
        self._define(Name=f"{self.NAME}_calibration")

    def calibration_kernel(self, view: tuple[int, int]) -> None:
        """One gradient echo at ``(line, partition)``.

        Parameters
        ----------
        view : tuple of int
            The phase-encode line and the partition to acquire.
        """
        ro, seq = self.gre, self.seq
        n_y, n_z = self.matrix[1:]
        line, partition = view
        ro.adc_labels[0].value, ro.adc_labels[1].value = line, partition
        ky = (line - n_y // 2) / (n_y / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        seq.add_block(
            ro.rf, *([] if self.gz is None else [self.gz]), *self.labels(REF=1)
        )
        wait_te = getattr(ro, "wait_te", None)
        if wait_te is not None:
            seq.add_block(wait_te)
        seq.add_block(
            ro.gx_pre, pp.scale_grad(ro.gy_pre, ky), pp.scale_grad(ro.gz_pre, kz)
        )
        seq.add_block(ro.gx, ro.adc, *ro.adc_labels)
        seq.add_block(
            ro.gx_spoil, pp.scale_grad(ro.gy_rew, ky), pp.scale_grad(ro.gz_rew, kz)
        )

    def reference(self) -> None:
        """Play one volume with the phase encode reversed, after its dummies."""
        self.play(frames=[0], reversed_encode=True)
        self._define(Name=f"{self.NAME}_reference", EchoSpacing=self.epi.esp)

    def _define(self, **definitions) -> None:
        for key, value in {
            "FOV": list(self.fov),
            "Matrix": list(self.matrix),
            **definitions,
        }.items():
            self.seq.set_definition(key=key, value=value)

    def loop(self) -> None:
        """Play the dummies, then every frame, shot by shot."""
        self.play(frames=range(self.n_frames))

    def play(self, frames, reversed_encode: bool = False) -> None:
        """Play the dummies and then ``frames``.

        Parameters
        ----------
        frames : iterable of int
            The frames to acquire, in play order.
        reversed_encode : bool, default=False
            Negate every line encode, which is what the reference prescan
            plays.
        """
        shots = [(None, view) for view in self.dummies]
        shots += [(frame, view) for frame in frames for view in self.volume]
        for frame, (shot, base) in shots:
            output = (
                self.volume_output
                and not reversed_encode
                and (shot, base) == self.volume[0]
            )
            self.kernel(shot, base, frame, output, reversed_encode)

    def kernel(
        self,
        shot: int,
        base: int,
        frame: int | None,
        output: bool = False,
        reversed_encode: bool = False,
    ) -> None:
        """One shot of the shell starting at partition ``base``; ``frame=None`` plays a dummy.

        ``reversed_encode`` negates every line encode and keeps the labels of
        the forward shot.

        Parameters
        ----------
        shot : int
            Which segment of the phase-encode lattice this shot reads.
        base : int
            The shell's first partition.
        frame : int or None
            The frame to acquire, or None for a dummy.
        output : bool, default=False
            Play the digital output that marks the start of a volume.
        reversed_encode : bool, default=False
            Negate every line encode.
        """
        seq, epi = self.seq, self.shot_trains[shot]
        n_y, n_z = self.matrix[1:]
        sign = -1.0 if reversed_encode else 1.0
        line, partition = self.origins[shot], base + int(epi.order[0, 1])
        acquire = frame is not None

        if not acquire:
            flags = {"ONCE": 1}
        else:
            flags = {"REP": frame, "SEG": shot, "ONCE": 0}
            if reversed_encode:
                flags["SET"] = 1
            if not self.n_dummy:
                del flags["ONCE"]
        labels = self.labels(**flags)
        epi.shot_labels[0].value, epi.shot_labels[1].value = line, partition

        # The prewinders and the rewinders closing the shot are scaled alike.
        ky = sign * (line - n_y // 2) / (n_y / 2)
        kz = (partition - n_z // 2) / (n_z / 2)
        swap = {
            id(epi.gy_pre): pp.scale_grad(epi.gy_pre, ky),
            id(epi.gy_rew): pp.scale_grad(epi.gy_rew, ky),
            id(epi.gz_pre): pp.scale_grad(epi.gz_pre, kz),
            id(epi.gz_rew): pp.scale_grad(epi.gz_rew, kz),
        }
        if reversed_encode:
            swap.update(
                (id(blip), pp.scale_grad(blip, -1.0))
                for blip in epi.gy_blips
                if blip is not None
            )
        step = shot * epi.echo_shift_step
        wait_shift = getattr(epi, "wait_shift", None)
        if wait_shift is not None:
            swap[id(wait_shift)] = pp.make_delay(wait_shift.delay + step)
        wait_tr = getattr(epi, "wait_tr", None)
        closing = self.pad + (0.0 if wait_tr is None else wait_tr.delay) - step

        navigator = {id(event) for event in getattr(epi, "gx_navigator", ())}
        lines = {id(event) for event in epi.gx}
        polarity = 0
        for index, block in enumerate(epi.blocks):
            head = block[0]
            if head is wait_tr:
                continue
            events = [
                swap.get(id(event), event)
                for event in block
                if acquire or event.type != "adc"
            ]
            if index == 0:
                events += [*labels, *([self.output] if output else [])]
            elif id(head) in navigator or id(head) in lines:
                if acquire:
                    events += self.labels(NAV=int(id(head) in navigator), REV=polarity)
                polarity ^= 1
            seq.add_block(*events)
        seq.add_block(pp.make_delay(closing))

    def finalize(self) -> None:
        """Write the prescription and the train's timing as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV (compat=False: the lines are sampled on the ramps).
        n_x, n_y, n_z = self.matrix
        definitions = {
            "FOV": list(self.fov),
            "Matrix": [n_x, n_y, n_z],
            "Name": self.NAME,
            "TE": self.echo_time,
            "TR": self.repetition_time,
            "EchoSpacing": self.epi.esp,
            "EPIFactor": self.epi.etl,
            "Excitation": self.excitation,
            "CaipiShift": self.shift,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterPartition": n_z // 2,
            "SliceThickness": self.fov[2],
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Epi3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="epi_3d.seq"))
