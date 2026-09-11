"""3D zero-echo-time imaging: a continuous-gradient koosh ball of spokes."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences


class Zte3DApp(sequences.SequenceApp):
    """3D zero echo time: hard pulses on a readout gradient that never returns to zero.

    One shell of views, written out as one continuous waveform that ramps up
    once at its first view and down once after its last, is replayed per shot
    turned about z by a rotation extension. The dead-time gap after each pulse
    leaves ``MissingSamples`` at the centre of k-space unacquired. Dummy
    shells precede the first shot, played as it is without the ADC.
    Acquisitions carry the view as ``LIN`` and the shot as ``SEG``.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.zte3D_sequence(n_x=32, n_views=24, n_shots=2, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    """

    NAME = "zte_3d"
    MAX_GRAD = 80.0
    MAX_SLEW = 200.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_views: int | None = None,
        n_shots: int | None = None,
        flip_angle_deg: float = 3.0,
        pulse_duration: float = 10e-6,
        readout_bandwidth_hz: float = 250e3,
        n_dummy: int = 2,
        n_gain_calibration_readouts: int = 1,
    ) -> None:
        """Design the hard pulse, the shell and the rotation of every shot.

        Parameters
        ----------
        fov : float, optional
            Isotropic field of view, in metres.
        n_x : int, optional
            Isotropic matrix size.
        n_views : int or None, optional
            Views per shell. ``None`` is the Nyquist-matched count.
        n_shots : int or None, optional
            Shells the sphere is dealt into. ``None`` balances the spacing
            within and between shells.
        flip_angle_deg : float, optional
            Hard-pulse flip angle, in degrees; small, because the pulse plays
            on the readout gradient.
        pulse_duration : float, optional
            Hard-pulse duration, in seconds, short enough to excite the whole
            spoke.
        readout_bandwidth_hz : float, optional
            Requested receiver bandwidth, in Hz.
        n_dummy : int, optional
            Whole shells played without acquiring before the first shot.
        n_gain_calibration_readouts : int, optional
            Written as the ``NumGainCalibrationReadouts`` definition.
        """
        system = self.system
        self.fov, self.n_x = fov, n_x
        self.n_dummy = n_dummy
        self.n_gain_calibration_readouts = n_gain_calibration_readouts
        self.exc = sequences.NonSelectiveExcitation(
            system, flip_angle_deg, duration_s=pulse_duration
        )
        self.zte = sequences.ZteReadout(
            system,
            self.exc.rf,
            fov=fov,
            matrix=n_x,
            n_views=n_views,
            n_shots=n_shots,
            readout_bandwidth_hz=readout_bandwidth_hz,
        )
        # One rotation for the whole shell: the shell is written out view by
        # view, so this only places the shot.
        self.rotations = pp.make_rotation(np.asarray(self.zte.shot_rotations))
        self.duration = (n_dummy + len(self.rotations)) * self.zte.duration

    def loop(self) -> None:
        """Play the dummy shells, then every shot."""
        for _ in range(self.n_dummy):
            self.kernel(0, acquire=False)
        for shot in range(len(self.rotations)):
            self.kernel(shot)

    def kernel(self, shot: int, acquire: bool = True) -> None:
        """One shell at shot ``shot``; ``acquire=False`` leaves the ADC off."""
        zte, seq = self.zte, self.seq
        turn = self.rotations[shot]
        seq.add_block(*zte.g_ramp, turn)
        for view in range(len(zte.directions)):
            once = self.labels(ONCE=int(not acquire)) if self.n_dummy else []
            seq.add_block(zte.rf, *zte.g_hold[view], turn, *once)
            if acquire:
                labels = self.labels(LIN=view, SEG=shot)
                seq.add_block(zte.adc, *zte.g_read[view], turn, *labels)
            else:
                seq.add_block(*zte.g_read[view], turn)

    def finalize(self) -> None:
        """Write the prescription, the shell layout and the central gap as definitions."""
        # The volume's offset is applied to the finished sequence with
        # pp.TransformFOV.
        definitions = {
            "FOV": [self.fov] * 3,
            "Matrix": [self.n_x] * 3,
            "Name": self.NAME,
            "TE": 0.0,
            "TR": self.zte.tr,
            "Trajectory": "zte",
            "NumShots": len(self.rotations),
            "ViewsPerShot": len(self.zte.directions),
            "MissingSamples": self.zte.n_missing,
            "NumGainCalibrationReadouts": self.n_gain_calibration_readouts,
        }
        for key, value in definitions.items():
            self.seq.set_definition(key=key, value=value)


main = Zte3DApp.main

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="zte_3d.seq"))
