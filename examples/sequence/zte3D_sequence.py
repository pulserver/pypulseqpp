"""3D zero echo time, a koosh ball of spokes read on a gradient that stays on."""

from __future__ import annotations

import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

#: The shell shapes ``scheme`` selects from.
SCHEMES = ("spiral", "meridian")

NAME = "zte_3d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: Duration of the nonselective hard pulse (s). It plays on the readout
#: gradient, so its bandwidth has to cover the whole spoke, and the
#: readout refuses a pulse the dead-time gap cannot hold.
HARD_PULSE_DURATION = 10e-6


def zte3d(
    system: pp.Opts | None = None,
    *,
    fov: float = 220e-3,
    n: int = 128,
    flip_angle_deg: float = 3.0,
    tr: float | None = None,
    readout_bandwidth_hz: float = 250e3,
    r: int = 1,
    n_shots: int | None = None,
    scheme: str = "spiral",
    n_dummy: int = 2,
    readout_oversampling: float = 2.0,
    n_gain_calibration_readouts: int = 1,
) -> pp.Sequence:
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

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
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

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``scheme`` is unknown, ``r`` is below one, or the hard pulse is
        longer than the dead-time gap can hold.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.zte3D_sequence(n=32, n_shots=2, n_dummy=0)
    >>> seq.check_timing()[0]
    True
    """
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    if r < 1:
        raise ValueError(f"r must be at least 1, got {r}")

    exc = sequences.NonSelectiveExcitation(
        system, flip_angle_deg, duration_s=HARD_PULSE_DURATION
    )
    ro = sequences.ZteReadout(
        system,
        exc.rf,
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
    rotations = pp.make_rotation(np.asarray(ro.shot_rotations))
    shots = list(range(0, len(rotations), r))

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(shot: int, acquire: bool = True) -> None:
        """Add one shell at shot ``shot``; ``acquire=False`` leaves the ADC off."""
        turn = rotations[shot]
        seq.add_block(*ro.g_ramp, turn)
        for view in range(len(ro.directions)):
            once = labels(ONCE=int(not acquire)) if n_dummy else []
            seq.add_block(ro.rf, *ro.g_hold[view], turn, *once)
            if acquire:
                label_events = labels(LIN=view, SEG=shot)
                seq.add_block(ro.adc, *ro.g_read[view], turn, *label_events)
            else:
                seq.add_block(*ro.g_read[view], turn)

    # The dummy shells play first, then every shot the undersampling keeps.
    for _ in range(n_dummy):
        kernel(0, acquire=False)
    for shot in shots:
        kernel(shot)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": [fov] * 3,
        "Matrix": [n, n, n],
        "Name": NAME,
        "TE": 0.0,
        "TR": ro.tr,
        "Trajectory": "zte",
        "NumShots": len(shots),
        "ViewsPerShot": len(ro.directions),
        "MissingSamples": ro.n_missing,
        "NumGainCalibrationReadouts": n_gain_calibration_readouts,
    }
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    return seq


main = zte3d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="zte_3d.seq"))
