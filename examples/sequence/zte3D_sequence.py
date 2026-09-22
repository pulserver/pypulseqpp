"""3D zero echo time, a koosh ball of spokes read on a gradient that stays on."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The shell shapes ``scheme`` selects from.
SCHEMES = ("spiral", "meridian")


class Zte3DApp(sequences.SequenceApp):
    """3D zero echo time: hard pulses on a readout gradient held on across each shell.

    One shell of views, written out as one continuous waveform that ramps up
    once at its first view and down once after its last, is replayed per shot
    turned about z by a rotation extension. The Nyquist set is
    ``ceil(pi * n ** 2)`` spokes over the sphere, dealt into ``n_shots``
    shells; every ``r``-th shell is played, in order, which is angular
    undersampling. The dead-time gap after each pulse leaves
    ``MissingSamples`` at the centre of k-space unacquired. Dummy shells
    precede the first shot, played as it is without the ADC. Acquisitions
    carry the view as ``LIN`` and the shot as ``SEG``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.zte3D_sequence(n=32, n_shots=2, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "zte_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0
    #: Duration of the nonselective hard pulse (s). It plays on the readout
    #: gradient, so its bandwidth has to cover the whole spoke, and the
    #: readout refuses a pulse the dead-time gap cannot hold.
    HARD_PULSE_DURATION = 10e-6

    def init_sequence(
        self,
        fov: float = 220e-3,
        n: int = 128,
        flip_angle_deg: float = 3.0,
        tr: float | None = None,
        readout_bandwidth_hz: float = 250e3,
        r: int = 1,
        n_shots: int | None = None,
        scheme: str = "spiral",
        *,
        n_dummy: int = 2,
        readout_oversampling: float = 2.0,
        n_gain_calibration_readouts: int = 1,
    ) -> None:
        """Design the hard pulse, the shell, and the shots the scan plays.

        Parameters
        ----------
        fov : float, default=0.22
            Isotropic field of view (m).
        n : int, default=128
            Isotropic matrix size; a spoke reads it from the centre out.
        flip_angle_deg : float, default=3.0
            Excitation flip angle (degrees); small, because the pulse plays on
            the readout gradient.
        tr : float | None, default=None
            Pulse centre to pulse centre (s), one per view. ``None`` is as
            short as the widest turn between views allows; a longer one is
            spent slewing more gently rather than waiting.
        readout_bandwidth_hz : float, default=250000.0
            Requested receiver bandwidth (Hz). It sets the gradient amplitude
            too, the spoke being traversed at one sample per ``delta_k``.
        r : int, default=1
            Angular undersampling: one shell in every ``r`` of the set is
            played.
        n_shots : int | None, default=None
            Shells the sphere is dealt into, each the same one turned about
            ``z``. ``None`` balances the spacing within a shell against the
            spacing between them.
        scheme : {'spiral', 'meridian'}, default='spiral'
            Shape of the shell, as :data:`SCHEMES` names them.
        n_dummy : int, default=2
            Whole shells played without acquiring before the first shot.
        readout_oversampling : float, default=2.0
            Radial oversampling: a finer ``delta_k`` along the same spoke.
        n_gain_calibration_readouts : int, default=1
            Written as the ``NumGainCalibrationReadouts`` definition, for the
            reconstruction to scale the gap at the centre against.

        Raises
        ------
        ValueError
            If ``scheme`` is unknown, ``r`` is below one, or the hard pulse is
            longer than the dead-time gap can hold.
        """
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if r < 1:
            raise ValueError(f"r must be at least 1, got {r}")

        system = self.system
        self.fov, self.matrix = fov, (n, n, n)
        self.scheme = scheme
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.exc = sequences.NonSelectiveExcitation(
            system, flip_angle_deg, duration_s=self.HARD_PULSE_DURATION
        )
        self.ro = sequences.ZteReadout(
            system,
            self.exc.rf,
            fov=fov,
            matrix=n,
            n_shots=n_shots,
            scheme=scheme,
            tr=tr,
            oversampling=readout_oversampling,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )

        # One rotation for the whole shell: the shell is written out view by
        # view, so this only places the shot.
        self.rotations = pp.make_rotation(np.asarray(self.ro.shot_rotations))
        self.shots = list(range(0, len(self.rotations), r))
        self.duration = (n_dummy + len(self.shots)) * self.ro.duration

    def loop(self) -> None:
        """Play the dummy shells, then every shot the undersampling keeps."""
        for _ in range(self.n_dummy):
            self.kernel(0, acquire=False)
        for shot in self.shots:
            self.kernel(shot)

    def kernel(self, shot: int, acquire: bool = True) -> None:
        """One shell at shot ``shot``; ``acquire=False`` leaves the ADC off.

        Parameters
        ----------
        shot : int
            Which rotation of the shell's directions this repetition plays.
        acquire : bool, default=True
            Play the ADC. False leaves it off, which is what a dummy plays.
        """
        ro, seq = self.ro, self.seq
        turn = self.rotations[shot]
        seq.add_block(*ro.g_ramp, turn)
        for view in range(len(ro.directions)):
            once = self.labels(ONCE=int(not acquire)) if self.n_dummy else []
            seq.add_block(ro.rf, *ro.g_hold[view], turn, *once)
            if acquire:
                labels = self.labels(LIN=view, SEG=shot)
                seq.add_block(ro.adc, *ro.g_read[view], turn, *labels)
            else:
                seq.add_block(*ro.g_read[view], turn)

    def finalize(self) -> None:
        """Write the prescription, the shell layout and the central gap as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov] * 3,
            "Matrix": list(self.matrix),
            "Name": self.NAME,
            "TE": 0.0,
            "TR": self.ro.tr,
            "Trajectory": "zte",
            "NumShots": len(self.shots),
            "ViewsPerShot": len(self.ro.directions),
            "MissingSamples": self.ro.n_missing,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Zte3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="zte_3d.seq"))
