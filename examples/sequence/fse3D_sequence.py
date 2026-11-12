"""3D Cartesian fast spin echo, with optionally designed and individually parameterized echo trains."""

from __future__ import annotations

import math
import sys

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, sequences

EXCITATIONS = ("nonselective", "slab")

#: The view orders ``ordering`` selects from.
ORDERINGS = (
    "radial",
    "shuffling",
    "busse_radial",
    "pclo",
    "buonincontri",
    "buonincontri_golden",
)

#: The refocusing trains ``flip_modulation`` selects from.
MODULATIONS = ("constant", "optimized", "busse", "pd")

NAME = "fse_3d"
MAX_GRAD = 80.0
MAX_SLEW = 200.0
#: SLR design of the slab-selective excitation.
PULSE_DURATION = 1e-3
TIME_BW_PRODUCT = 4.0
#: Duration of the nonselective hard pulses (s): every refocusing pulse, and
#: the excitation of a nonselective train.
HARD_PULSE_DURATION = 0.5e-3
#: Dephasing each crusher beside a refocusing pulse winds on the readout
#: axis, in cycles across one voxel.
CRUSHER_CYCLES = 1.0
#: Seed of the shuffled order and of its Poisson-disc views.
SHUFFLE_SEED = 0
#: Pacing of one three-plane navigator (s), and the most one TR wait
#: carries: each takes a little longitudinal magnetisation from the volume
#: the wait is restoring.
NAVIGATOR_TR = 100e-3
NAVIGATOR_COUNT = 5
#: Tissues the ``optimized`` trains are designed for (ms): cartilage,
#: muscle and synovial fluid.
DESIGN_T1_MS = (1200.0, 1420.0, 3600.0)
DESIGN_T2_MS = (35.0, 30.0, 250.0)
#: The two tissues, brighter first, whose difference at the TE echo is the
#: contrast the design keeps, and how much it weighs against sharpness.
DESIGN_CONTRAST = (2, 0)
DESIGN_CONTRAST_WEIGHT = 12.0
#: How much RF power beyond the starting trains' costs.
DESIGN_POWER_WEIGHT = 20.0
#: Lowest refocusing angle a designed train may reach (degrees).
DESIGN_MIN_DEG = 20.0
#: Configuration-state count, iterations and step of the design.
DESIGN_STATES = 12
DESIGN_ITERATIONS = 25
DESIGN_LEARNING_RATE = 0.3
DESIGN_B1_SCALES = (0.8, 1.0, 1.2)


def cubic(fraction):
    """Return the smooth step ``3 u**2 - 2 u**3`` the parameters move along.

    Parameters
    ----------
    fraction : array-like
        Where along the transition, from 0 at the centre to 1 at the
        periphery.

    Returns
    -------
    numpy.ndarray
        The step, over the same range.
    """
    fraction = np.asarray(fraction, dtype=float)
    return 3 * fraction**2 - 2 * fraction**3


def shot_parameters(
    n_views: int, etl: int, etl_periphery: int, tr: float, tr_periphery: float
) -> tuple[list[int], list[float], np.ndarray]:
    """Return every shot's train length and TR, and its place in the transition.

    Shot ``s`` of ``n`` sits at ``u = s / (n - 1)`` and takes its train length
    and TR a cubic step (:func:`cubic`) of the way from the centre values to
    the periphery ones (Buonincontri et al., ISMRM 2025, abstract 566-05-007,
    Fig. 1). ``n`` is the fewest shots whose trains hold every view.

    Parameters
    ----------
    n_views : int
        Views the shots have to hold between them.
    etl : int
        Echo train length of the centre shot.
    etl_periphery : int
        Echo train length of the periphery shot.
    tr : float
        Repetition time of the centre shot, in s.
    tr_periphery : float
        Repetition time of the periphery shot, in s.

    Returns
    -------
    lengths : list of int
        Each shot's train length.
    times : list of float
        Each shot's repetition time, in s.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    """
    n = max(1, -(-n_views // max(etl, etl_periphery)))
    while True:
        place = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
        step = cubic(place)
        lengths = [round(etl + (etl_periphery - etl) * c) for c in step]
        if sum(lengths) >= n_views:
            break
        n += 1
    times = [tr + (tr_periphery - tr) * c for c in step]
    return lengths, times, place


def pclo_rings(te_echo: int, etl: int) -> np.ndarray:
    """Return the PCLO echo permutation for radius-ranked bands (zero-based)."""
    from pypulseqpp._ordering import _pclo_rings

    return _pclo_rings(te_echo, etl)


#: The fractional part of the golden ratio, which spreads consecutive
#: integers evenly around a turn.
_GOLDEN = (5**0.5 - 1) / 2


def deal_trains(
    coords: np.ndarray,
    lengths: list[int],
    place: np.ndarray,
    te_echo: int,
    etl_max: int,
    *,
    golden: bool = False,
) -> list[list[int | None]]:
    """Deal views into trains by adaptive radial reordering.

    Every ``(shot, echo)`` slot a train plays is ranked by its distance from
    the prescription: the ring the echo takes in the PCLO order
    (:func:`pclo_rings`), and how far the shot is along the transition. The
    views are ranked by radius. Both are cut into sections of one slot per
    shot, the innermost views filling the nearest slots, and within a section
    the views, sorted by angle, fill the slots in shot order, so the shot
    index sweeps with the angle within each section (Buonincontri et al.,
    abstract 566-05-007, Fig. 2C and D); the centre of k-space, which has no
    angle, takes the nearest slot. With identical trains the shot term
    vanishes and a section is one echo, so the rings of k-space take the
    echoes in PCLO order.

    With ``golden``, each shot instead holds one place around the turn, the
    golden-ratio fraction of its rank along the transition, and a section's
    slots take its angles in that order: a shot keeps to one angle, and every
    arc of a ring holds the section's whole range of TRs and train lengths
    rather than one end of it.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(n_views, 2)`` phase-encode and partition coordinates of each view,
        centred on k-space.
    lengths : list of int
        Each shot's train length.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    te_echo : int
        The echo the prescribed echo time falls on, counting from 0.
    etl_max : int
        Echoes the longest train holds.
    golden : bool, default=False
        Give each shot a golden-ratio place around the turn rather than
        dealing each section in shot order.

    Returns
    -------
    list of list of int or None
        One ``etl_max``-long train per shot, holding view indices; None where
        an echo plays unencoded and unacquired.
    """
    n_shots = len(lengths)
    radius = np.hypot(coords[:, 0], coords[:, 1])
    angle = np.arctan2(coords[:, 1], coords[:, 0])
    views = sorted(range(len(coords)), key=lambda i: (radius[i], angle[i]))
    span = max(etl_max - 1, 1)
    ring = np.argsort(pclo_rings(te_echo, etl_max))
    slots = sorted(
        ((s, e) for s in range(n_shots) for e in range(lengths[s])),
        key=lambda slot: (
            math.hypot(ring[slot[1]] / span, place[slot[0]]),
            slot[1],
            slot[0],
        ),
    )
    turn = np.arange(n_shots, dtype=float)
    if golden:
        rank = np.argsort(np.argsort(place, kind="stable"), kind="stable")
        turn = (rank * _GOLDEN) % 1.0
    trains: list[list[int | None]] = [[None] * etl_max for _ in range(n_shots)]
    # The centre of k-space, which has no angle, takes the nearest slot.
    if len(views) and radius[views[0]] == 0.0:
        shot, echo = slots.pop(0)
        trains[shot][echo] = views.pop(0)
    for start in range(0, len(views), n_shots):
        section = sorted(views[start : start + n_shots], key=lambda i: angle[i])
        # The last section may hold fewer views than it has slots.
        held = sorted(
            slots[start : start + n_shots], key=lambda slot: (turn[slot[0]], slot)
        )
        for (s, e), view in zip(held, section, strict=False):
            trains[s][e] = view
    return trains


def design_trains(
    refocusing_angle_deg: float,
    individual: bool,
    lengths: list[int],
    times: list[float],
    place: np.ndarray,
    te_echo: int,
    etl_max: int,
    esp: float,
    *,
    first_refocusing_angle_deg: float | None = None,
    first_echo_time: float | None = None,
    b1_scales: tuple[float, ...] = (1.0,),
    optimize_end_angle: bool = False,
) -> np.ndarray:
    """Design the refocusing angles (degrees) of every shot, ``(shots, etl_max)``.

    Cubic transitions connect the early control angle, the angle at the
    prescribed TE and the final angle. By default only the early minimum is
    optimized and the centre and final angles equal the prescription.
    ``optimize_end_angle`` also fits the final angle; the global minimum may
    then occur at the end. This parameterization is fitted directly with EPG,
    rather than inverting a prescribed signal envelope. The prescribed angle
    bounds every pulse except an explicitly stated first refocusing pulse.
    The controls are designed with blochsim against
    the sharpness of the periphery of k-space, the contrast at the centre
    between the tissues ``DESIGN_CONTRAST`` names, and the RF power of the
    starting trains. Trains that are not individually parameterized share one
    minimum, designed over their distinct lengths and TRs, each weighted by the
    shots that play it. Individually parameterized trains have a minimum at
    each end of the transition, the two designed together on the two end
    shots alone, at their own train lengths and TRs: the centre's against the
    contrast, the periphery's against the sharpness, both under the RF power
    of their starting trains. Every shot takes its minimum a cubic step of the
    way between them. Angles past a shot's own train length are zero.

    Parameters
    ----------
    refocusing_angle_deg : float
        The refocusing angle at the TE echo and the peak of every train
        (degrees).
    individual : bool
        Whether the trains are individually parameterized, which gives each
        end of k-space its own minimum angle.
    lengths : list of int
        Each shot's train length.
    times : list of float
        Each shot's repetition time, in s.
    place : numpy.ndarray
        Each shot's place along the transition, from 0 to 1.
    te_echo : int
        The echo the prescribed echo time falls on, counting from 0.
    etl_max : int
        Echoes the longest train holds.
    esp : float
        Echo spacing, in s.
    first_refocusing_angle_deg : float | None, default=None
        First refocusing angle (degrees), included in every cost evaluation.
        ``None`` uses the parameterized profile at the first pulse.
    first_echo_time : float | None, default=None
        First echo time (s). ``None`` uses ``esp``.
    b1_scales : tuple of float, default=(1.0,)
        Relative transmit amplitudes averaged in the objective.
    optimize_end_angle : bool, default=False
        Optimize the final angle as well as the early control angle for a
        shared train, under the starting profile's RF-energy budget.

    Returns
    -------
    numpy.ndarray
        ``(shots, etl_max)`` refocusing angles in degrees, zero past a shot's
        own train length.

    Raises
    ------
    ImportError
        If blochsim is not installed.
    """
    try:
        import torch
        from blochsim.optim import Bounded, SequenceDesign
        from blochsim.simulators import FSESimulator
    except ImportError as error:
        raise ImportError(
            "flip_modulation='optimized' designs the trains with blochsim; "
            "install it with pip install 'pypulseqpp[design]'"
        ) from error

    dtype = torch.float64
    angle = float(refocusing_angle_deg)
    echo = torch.arange(1, etl_max + 1, dtype=dtype)
    centre = float(te_echo + 1)
    optimized_end = None
    if optimize_end_angle and individual:
        raise ValueError("End-angle optimization requires a shared refocusing train")

    def column(values):
        return torch.as_tensor(np.asarray(values, dtype=float), dtype=dtype)[:, None]

    def smooth(knots, values, one):
        """Smooth steps between successive ``(knot, value)`` pairs, held beyond them."""
        out = values[-1] * one
        for k in reversed(range(len(knots) - 1)):
            span = (echo - knots[k]) / (knots[k + 1] - knots[k]).clamp_min(1e-3)
            span = span.clamp(0.0, 1.0)
            ramp = values[k] + (values[k + 1] - values[k]) * span.square() * (
                3 - 2 * span
            )
            out = torch.where(echo <= knots[k + 1], ramp, out)
        return out

    def trains(low, length, end=None):
        """Return the trains of ``length`` echoes whose minimum is ``low``, one row each."""
        one = torch.ones_like(length)
        final = end if end is not None else optimized_end
        final = angle * one if final is None else final * one
        if centre > 5.0:
            knots = [one, 5.0 * one, centre * one, length]
            values = [angle * one, low, angle * one, final]
        else:
            fallen = torch.minimum((centre + 4.0) * one, length)
            knots = [centre * one, fallen, length]
            values = [angle * one, low, final]
        profile = smooth(knots, values, one)
        if first_refocusing_angle_deg is not None:
            first = torch.full_like(profile[:, :1], first_refocusing_angle_deg)
            profile = torch.cat((first, profile[:, 1:]), dim=-1)
        return profile * (echo <= length).to(dtype)

    if not b1_scales or any(b <= 0 for b in b1_scales):
        raise ValueError("b1_scales must contain positive relative transmit amplitudes")
    first_echo = esp if first_echo_time is None else first_echo_time
    if first_echo <= 0:
        raise ValueError("The first echo time must be positive")
    matched = first_refocusing_angle_deg is not None or first_echo != esp

    class TimedFSE(FSESimulator):
        def layout(
            self, *, flip, ESP, phases=0.0, exc_flip=90.0, exc_phase=90.0, TR=1e6
        ):
            parts = super().layout(
                flip=flip,
                ESP=ESP,
                phases=phases,
                exc_flip=exc_flip,
                exc_phase=exc_phase,
                TR=TR,
            )
            extra = first_echo - esp
            return [
                (t + (extra / 2 if i == 1 else extra if i > 1 else 0.0), op)
                for i, (t, op) in enumerate(parts)
            ]

    n_b1 = len(b1_scales)
    nonunit_b1 = any(b != 1.0 for b in b1_scales)
    tissues = len(DESIGN_T1_MS)
    simulator = (TimedFSE if matched else FSESimulator)(
        ESP=esp * 1e3,
        states=max(DESIGN_STATES, etl_max + 1)
        if matched or nonunit_b1
        else DESIGN_STATES,
        T1=list(DESIGN_T1_MS) * n_b1,
        T2=list(DESIGN_T2_MS) * n_b1,
        B1=np.repeat(b1_scales, tissues).tolist(),
    )
    bright, dark = DESIGN_CONTRAST

    def measure(low, length, tr_ms, end=None):
        """Return the sharpness, contrast and RF power of each train."""
        flip = trains(low, length, end)
        acquired = (echo <= length).to(dtype)
        repetitions = 8 if nonunit_b1 else 2
        if (matched or nonunit_b1) and flip.shape[0] == 1:
            signal = simulator.simulate(
                flip=flip[0], TR=float(tr_ms[0, 0]), repetitions=repetitions
            ).abs()[None]
        else:
            signal = simulator.simulate(
                flip=flip, TR=tr_ms, repetitions=repetitions
            ).abs()
        pair = acquired[:, None, :-1] * acquired[:, None, 1:]
        slope = (torch.diff(signal, dim=-1) * pair).square().sum(-1)
        energy = (signal * acquired[:, None, :]).square().sum(-1).clamp_min(1e-12)
        blur = (length / (2 * torch.pi) * (slope / energy).sqrt()).mean(-1)
        samples = signal.reshape(signal.shape[0], n_b1, tissues, etl_max)
        contrast = (samples[:, :, bright, te_echo] - samples[:, :, dark, te_echo]).mean(
            -1
        )
        power = ((flip / 180.0).square() * acquired).sum(-1) / (tr_ms[:, 0] * 1e-3)
        return blur, contrast, power

    def mean(values, weights):
        total = weights.sum()
        return (weights * values).sum() / total if total > 0 else 0.0

    # Each minimum moves between its limits, the midpoint to start; one with
    # no room to move is held at the prescribed angle.
    lowest = min(DESIGN_MIN_DEG, 0.5 * angle)

    def minima(parameter, length, tr_ms, shots, sharp, contrasted):
        """Return the minima the trains take, designed with blochsim.

        Train ``i`` takes minimum ``parameter[i]``. ``shots`` weighs the RF
        power of each train, ``sharp`` its sharpness and ``contrasted`` its
        contrast.
        """
        nonlocal optimized_end
        count = max(parameter) + 1
        if angle - lowest <= 1.0:
            return torch.full((count,), angle, dtype=dtype)
        start = torch.tensor([0.5 * (lowest + angle)], dtype=dtype)
        bounds = {f"low{n}": Bounded(start, lowest, angle) for n in range(count)}
        index = torch.as_tensor(parameter)

        def lows(values):
            return torch.cat([values[f"low{n}"] for n in range(count)])[index, None]

        initial = {name: bound.initial for name, bound in bounds.items()}
        budget = mean(measure(lows(initial), length, tr_ms)[2], shots)

        if optimize_end_angle:
            from scipy.optimize import minimize

            cache = {}

            def evaluate(values):
                if np.array_equal(cache.get("point"), values):
                    return cache["result"]
                controls = torch.tensor(values, dtype=dtype, requires_grad=True)
                low = controls[0].expand(len(parameter), 1)
                end = controls[1].expand(len(parameter), 1)
                blur, contrast, power = measure(low, length, tr_ms, end)
                loss = mean(blur, sharp) - DESIGN_CONTRAST_WEIGHT * mean(
                    contrast, contrasted
                )
                energy = mean(power, shots)
                (gradient,) = torch.autograd.grad(loss, controls, retain_graph=True)
                (energy_gradient,) = torch.autograd.grad(energy, controls)
                result = (
                    loss.detach().item(),
                    gradient.detach().numpy(),
                    energy.detach().item(),
                    energy_gradient.detach().numpy(),
                )
                cache.update(point=np.array(values), result=result)
                return result

            allowed = budget.detach().item()
            result = minimize(
                lambda values: evaluate(values)[:2],
                [0.5 * (lowest + angle), angle],
                method="SLSQP",
                jac=True,
                bounds=((lowest, angle), (lowest, angle)),
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda values: allowed - evaluate(values)[2],
                        "jac": lambda values: -evaluate(values)[3],
                    }
                ],
                options={"maxiter": DESIGN_ITERATIONS, "ftol": 1e-6},
            )
            if not result.success or evaluate(result.x)[2] > allowed + 1e-5:
                raise ValueError(f"End-angle optimization failed: {result.message}")
            optimized_end = torch.tensor(result.x[1], dtype=dtype)
            return torch.tensor([result.x[0]], dtype=dtype)

        if matched and count == 1:
            from scipy.optimize import minimize_scalar

            def scalar_cost(value):
                low = torch.full((len(parameter), 1), value, dtype=dtype)
                with torch.no_grad():
                    blur, contrast, power = measure(low, length, tr_ms)
                    return float(
                        mean(blur, sharp)
                        - DESIGN_CONTRAST_WEIGHT * mean(contrast, contrasted)
                        + DESIGN_POWER_WEIGHT
                        * torch.relu(mean(power, shots) / budget - 1.0)
                    )

            result = minimize_scalar(
                scalar_cost,
                bounds=(lowest, angle),
                method="bounded",
                options={"xatol": 0.02, "maxiter": DESIGN_ITERATIONS},
            )
            if not result.success:
                raise ValueError(
                    f"Refocusing optimization did not converge: {result.message}"
                )
            return torch.tensor([result.x], dtype=dtype)

        def cost(**values):
            blur, contrast, power = measure(lows(values), length, tr_ms)
            return (
                mean(blur, sharp)
                - DESIGN_CONTRAST_WEIGHT * mean(contrast, contrasted)
                + DESIGN_POWER_WEIGHT * torch.relu(mean(power, shots) / budget - 1.0)
            )

        result = SequenceDesign(cost, **bounds).minimize(
            iterations=DESIGN_ITERATIONS, learning_rate=DESIGN_LEARNING_RATE
        )
        return torch.cat([result.parameters[f"low{n}"] for n in range(count)]).detach()

    place = np.asarray(place, dtype=float)
    length = column(lengths)
    if individual:
        # Only the end shots are simulated: the shots between take their
        # minimum from the two.
        ends = [int(np.argmin(place)), int(np.argmax(place))]
        step = column(cubic(place[ends]))[:, 0]
        centre_low, edge_low = minima(
            [0, 1],
            length[ends],
            column(times)[ends] * 1e3,
            torch.ones_like(step),
            step,
            1.0 - step,
        )
        low = centre_low + (edge_low - centre_low) * column(cubic(place))
    else:
        # Each distinct train is simulated once, weighted by the shots playing it.
        rows, shots = np.unique(
            np.column_stack([lengths, times, place]), axis=0, return_counts=True
        )
        weight = column(shots)[:, 0]
        (shared,) = minima(
            [0] * len(rows),
            column(rows[:, 0]),
            column(rows[:, 1]) * 1e3,
            weight,
            weight,
            weight * (1.0 - column(cubic(rows[:, 2]))[:, 0]),
        )
        low = shared * torch.ones_like(length)
    with torch.no_grad():
        flip = trains(low, length)
    return flip.numpy(force=True)


def fse3d(
    system: pp.Opts | None = None,
    *,
    fov_x: float = 160e-3,
    fov_y: float = 160e-3,
    fov_z: float = 160e-3,
    n_x: int = 320,
    n_y: int = 240,
    n_z: int = 240,
    te: float | None = 28e-3,
    tr: float = 1800e-3,
    etl: int = 45,
    refocusing_angle_deg: float = 180.0,
    readout_bandwidth_hz: float = 250e3,
    ry: int = 1,
    rz: int = 1,
    caipi_shift: int = 0,
    partial_fourier_x: float = 1.0,
    partial_fourier_y: float = 1.0,
    partial_fourier_z: float = 1.0,
    n_dummy: int | None = None,
    excitation: str = "slab",
    readout_oversampling: float = 2.0,
    esp: float | None = None,
    crusher_cycles: float = 1.0,
    optimize_end_angle: bool = False,
    n_acs_y: int = 24,
    n_acs_z: int = 16,
    elliptical_acs: bool = False,
    ordering: str = "radial",
    flip_modulation: str = "constant",
    busse_minimum_angle_deg: float = 25.0,
    busse_centre_angle_deg: float = 70.0,
    busse_maximum_angle_deg: float = 120.0,
    busse_startup_echoes: int = 6,
    busse_centre_echo: int | None = None,
    busse_effective_te: float | None = None,
    busse_initial_decay: float = 1.3,
    busse_optimize: bool = False,
    busse_minimum_angle_periphery_deg: float | None = None,
    busse_centre_angle_periphery_deg: float | None = None,
    busse_maximum_angle_periphery_deg: float | None = None,
    pd_first_angle_deg: float = 120.0,
    pd_maximum_angle_deg: float = 140.0,
    pd_final_angle_deg: float = 100.0,
    pd_plateau_fraction: float = 0.7,
    pd_effective_te: float | None = None,
    pd_optimize: bool = False,
    pd_first_angle_periphery_deg: float | None = None,
    pd_maximum_angle_periphery_deg: float | None = None,
    pd_final_angle_periphery_deg: float | None = None,
    pd_plateau_fraction_periphery: float | None = None,
    tr_periphery: float | None = None,
    etl_periphery: int | None = None,
    wave: str = "both",
    wave_cycles: int = 8,
    wave_amplitude: float = 0.0,
    navigator: bool = False,
) -> pp.Sequence:
    """3D Cartesian fast spin echo: one CPMG train per excitation over a (ky, kz) grid.

    Only views inside the inscribed ky-kz ellipse are sampled. ``radial``
    selects PCLO for identical trains and Buonincontri joint ordering for a
    centre/periphery parameter transition. ``shuffling`` selects a
    variable-density Poisson-disc support and assigns its views to echoes
    with :func:`~pypulseqpp.make_complementary_order`, so that each echo is a
    Poisson-disc pattern of its own and neighbouring echoes complement each
    other (T2 Shuffling, Tamir et al., Magn Reson Med 2017;77:180-195, with
    complementary Poisson-disc sampling, Levine et al., Magn Reson Med
    2017;77:1774-1785), for an echo-resolved reconstruction. Every acquisition carries its line and partition as ``LIN``
    and ``PAR``, and, shuffled, its echo, the contrast an echo-resolved
    reconstruction resolves, as ``ECO``; calibration views are marked ``IMA``;
    under wave-CAIPI the calibration region is first acquired wave-free in
    trains of its own, marked ``REF``.

    The refocusing angle is constant, or, under ``optimized``, the angle at the
    TE echo of a train :func:`design_trains` shapes around it. ``busse`` uses
    :func:`~pypulseqpp.make_busse_schedule`: prescribed PSS coherence and inverse
    EPG, with six unacquired startup echoes by default. Its centre is halfway
    through the acquired train unless explicitly prescribed. Giving the
    periphery of k-space its own TR or train length individually parameterizes
    the trains (Buonincontri et al., ISMRM 2025, abstract 566-05-007): every
    shot moves from the centre values to the periphery ones along
    :func:`shot_parameters`, and the view order is radial. Every shot plays the
    longest train; past its own length a shot's refocusing pulses have zero
    amplitude and nothing is acquired, and its closing delay makes its TR, so
    the scan stays one repeating train. Trains that are not individually
    parameterized all acquire ``etl`` views: the unsampled views nearest the
    k-space centre are added until the views fill a whole number of trains.

    Parameters
    ----------
    system : pypulseqpp.Opts, default=None
        System limits, held under the module's ``MAX_GRAD`` (80 mT/m) and
        ``MAX_SLEW`` (200 T/m/s). ``None`` is ``pypulseqpp.Opts()``.
    fov_x, fov_y, fov_z : float, default=0.16
        Fields of view along the readout, the phase encode and the partition
        encode, in that order (m). The slab excited is ``fov_z`` thick.
    n_x : int, default=320
        Matrix size along the readout.
    n_y : int, default=240
        Matrix size along the phase encode.
    n_z : int, default=240
        Matrix size along the partition encode.
    te : float | None, default=0.028
        Effective echo time (s): the echo the centre of k-space is read
        at, rounded onto the echo grid. ``None`` is the first echo.
        Meaningless under ``shuffling``. Under ``busse``, pass ``None`` and
        prescribe the centre with ``busse_centre_echo`` instead.
    tr : float, default=1.8
        Repetition time (s), one per train; at the centre of k-space when
        the trains are individually parameterized.
    etl : int, default=45
        Echo train length; at the centre of k-space when the trains are
        individually parameterized. Under ``busse`` this counts acquired
        echoes; startup pulses are additional.
    refocusing_angle_deg : float, default=180.0
        Refocusing flip angle (degrees): of every pulse under
        ``constant``; at the TE echo, and the peak of the train, under
        ``optimized``. Unused under ``busse``.
    readout_bandwidth_hz : float, default=250000.0
        Requested receiver bandwidth (Hz).
    ry, rz : int, default=1
        Undersampling along the phase and the partition encode.
    caipi_shift : int, default=0
        CAIPIRINHA shift: partitions by which the lattice is displaced per
        acquired line, in ``[0, rz)``.
        Unused by ``shuffling``.
    partial_fourier_x : float, default=1.0
        Fraction of the echo acquired, in ``[0.75, 1]``. Truncates the samples
        before the echo.
    partial_fourier_y, partial_fourier_z : float, default=1.0
        Fraction of the phase- and partition-encode extent acquired, in
        ``[0.75, 1]``.
    n_dummy : int | None, default=None
        Trains played without acquiring before the first. ``None`` is as many
        as bring the longitudinal magnetisation to its steady state
        (:func:`~pypulseqpp.sequences.steady_state_dummies`): the radial
        order reads the centre of k-space in the first train acquired.
    excitation : {'slab', 'nonselective'}, default='slab'
        Slab-selective or hard excitation. The refocusing pulses are hard
        either way, crushed on the readout axis: spins outside the slab are
        never excited, and hard pulses keep the echo spacing short.
    readout_oversampling : float, default=2.0
        Readout oversampling factor, at least one.
    esp : float | None, default=None
        Echo spacing (s). ``None`` is as short as the train admits.
    crusher_cycles : float, default=1.0
        Read-axis crusher area in cycles across one native readout voxel.
    optimize_end_angle : bool, default=False
        Optimize the final refocusing angle under the starting profile's RF
        energy budget. Requires optimized, shared trains.
    n_acs_y : int, default=24
        Extent of the fully sampled calibration region along the phase encode,
        acquired ahead of the rest when undersampled.
    n_acs_z : int, default=16
        Extent of the fully sampled calibration region along the partition
        encode, acquired ahead of the rest when undersampled.
    elliptical_acs : bool, default=False
        Make the calibration region the ellipse inscribed in the
        ``n_acs_y x n_acs_z`` rectangle rather than the rectangle.
    ordering : {'radial', 'shuffling', 'busse_radial', 'pclo', 'buonincontri', 'buonincontri_golden'}, default='radial'
        ``'radial'``: choose PCLO for identical trains, or Buonincontri joint
        shot/echo assignment for a centre/periphery transition. ``'pclo'``
        explicitly selects Rydén's radius-band permutation for an arbitrary
        centre echo, with identical trains only; for a centre echo in the
        second half of the train the echoes after it take the outer bands in
        order instead of Rydén's mirrored order. ``'busse_radial'`` uses the
        basic Busse radius-then-angle order (no sorting jitter/consolidation),
        with identical trains and centre echo 0; under ``busse``, explicitly
        set ``busse_centre_echo=0``. ``'buonincontri'`` requires a parameter
        transition and assigns shots and echoes jointly, the shot index
        sweeping with the angle within each radial section;
        ``'buonincontri_golden'`` keeps each shot to one golden-ratio angle
        instead (:func:`deal_trains`). ``'adaptive_radial'``
        is accepted as an alias for ``'buonincontri'``. ``'shuffling'`` uses
        complementary Poisson-disc echo assignment for identical trains.
    flip_modulation : {'constant', 'optimized', 'busse', 'pd'}, default='constant'
        Constant refocusing angles, or variable trains designed by direct-angle
        optimization, prescribed-PSS Busse inverse EPG or relaxation-aware
        prescribed-signal PD inverse EPG. Every variable train is designed
        with blochsim, which the ``design`` extra installs.
    busse_minimum_angle_deg : float, default=25.0
        Minimum actual refocusing angle (degrees), under ``busse``.
    busse_centre_angle_deg : float, default=70.0
        Actual refocusing angle at the centre echo (degrees), under ``busse``.
    busse_maximum_angle_deg : float, default=120.0
        Actual first and final angle (degrees), under ``busse``.
    busse_startup_echoes : int, default=6
        Leading unacquired refocusing pulses, additional to ``etl``, under
        ``busse`` and ``pd``. They play in imaging, calibration and dummy trains.
    busse_centre_echo : int | None, default=None
        Zero-based acquired echo index under ``busse`` or ``pd``. For Busse, ``None`` selects
        ``(etl - 1) // 2``; PD uses the first acquired echo. An effective-TE
        request selects the nearest feasible echo. Startup precedes this index.
    busse_effective_te : float | None, default=None
        Requested contrast-equivalent TE (s) for Busse trains. Select the
        nearest feasible acquired centre echo by relaxed/relaxation-free EPG
        with reference T1/T2 of 1.0/0.1 s. The centre-angle constraint follows
        the selected echo. ``None`` retains the midpoint prescription;
        an explicit ``busse_centre_echo`` takes precedence. Control fitting
        preserves the selected reference effective TE within 0.1 ms.
        The ``TE`` definition remains physical centre time; the achieved
        reference contrast time is ``EffectiveEchoTime``. Requires ``busse``.
    busse_initial_decay : float, default=1.3
        Initial exponential time constant, measured in echoes, under ``busse`` and ``pd``.
    busse_optimize : bool, default=False
        Optimize only the minimum and centre controls within five degrees of
        their prescription, using forward EPG over ``DESIGN_T1_MS``,
        ``DESIGN_T2_MS`` and ``DESIGN_B1_SCALES``. Minimize the mean PSF RMS
        radius for the actual view order, allowing signed contrast to drop by
        at most 4.5% of its prescribed magnitude and bounding RF energy. The
        startup count, centre echo, maximum angle and initial decay stay fixed.
        Requires ``busse`` trains and the ``design`` extra. With varying train
        parameters, the centre endpoint maximizes model contrast and the
        periphery endpoint minimizes PSF radius. Each endpoint is fitted at
        its own TR/ETL over a representative complete-plane ordering; controls
        then follow the cubic shot transition. This is an endpoint design,
        not a joint optimization of the entire varying-TR acquisition.
    busse_minimum_angle_periphery_deg : float | None, default=None
        Minimum actual angle at the periphery (degrees). ``None`` inherits
        ``busse_minimum_angle_deg``. Requires adaptive radial ordering when
        the endpoints differ.
    busse_centre_angle_periphery_deg : float | None, default=None
        Actual centre angle at the periphery (degrees). ``None`` inherits
        ``busse_centre_angle_deg``.
    busse_maximum_angle_periphery_deg : float | None, default=None
        First/final angle at the periphery (degrees). ``None`` inherits
        ``busse_maximum_angle_deg``. Each shot's controls interpolate cubically;
        inverse EPG constructs its complete train at its own ETL.
    pd_first_angle_deg : float, default=120.0
        First RF angle (degrees) under ``pd``, independently of peak/final.
    pd_maximum_angle_deg : float, default=140.0
        Maximum actual PD RF angle (degrees).
    pd_final_angle_deg : float, default=100.0
        Final actual PD RF angle (degrees), at most the maximum. Equality
        allows the RF maximum to extend to the final pulse.
    pd_plateau_fraction : float, default=0.7
        Acquired-train fraction before the PD signal tail begins. The
        resulting RF peak position follows from relaxation-aware inverse EPG.
    pd_effective_te : float | None, default=None
        Requested reference contrast-equivalent TE (s) under ``pd``. Select
        the nearest acquired echo from the prescribed PD train's reference
        EPG times; ``None`` selects its first acquired echo. An explicit
        ``busse_centre_echo`` overrides this choice. ``TE`` remains physical
        centre time; ``EffectiveEchoTime`` records reference contrast time.
    pd_optimize : bool, default=False
        Fit tail onset within 0.1 of the acquired train using CPU forward
        EPG, retaining the prescribed first, peak and final angles.
        Bound RF energy and retain centre mean signal and signed contrast.
        Preserve a prescribed reference effective TE to 0.1 ms. Requires
        ``pd`` and the ``design`` extra. As for Busse, fit endpoints separately
        and construct the cubic transition with batched inverse EPG.
    pd_first_angle_periphery_deg : float | None, default=None
        Peripheral first RF angle (degrees); ``None`` inherits the centre.
    pd_maximum_angle_periphery_deg : float | None, default=None
        Peripheral maximum RF angle (degrees); ``None`` inherits the centre.
    pd_final_angle_periphery_deg : float | None, default=None
        Peripheral final RF angle (degrees); ``None`` inherits the centre.
    pd_plateau_fraction_periphery : float | None, default=None
        Peripheral signal-tail onset fraction; ``None`` inherits the centre.
    tr_periphery : float | None, default=None
        Repetition time (s) at the periphery of k-space. ``None`` is
        ``tr``.
    etl_periphery : int | None, default=None
        Echo train length at the periphery of k-space. ``None`` is
        ``etl``.
    wave : {'phase', 'partition', 'both'}, default='both'
        Wave-CAIPI channels: a sine on y, a cosine on z, or both. With
        wave-encoding gradients the calibration region is acquired again
        first without them, marked ``REF``, and no wave-encoded view is
        marked ``IMA``.
    wave_cycles : int, default=8
        Wave periods across the sampling window; zero plays no wave.
    wave_amplitude : float, default=0.0
        Requested peak wave-encoding gradient amplitude (T/m); zero, the
        default, plays no wave. The slew rate may lower it.
    navigator : bool, default=False
        Play three-plane spiral navigators after each train, as many as
        the shortest TR holds up to ``NAVIGATOR_COUNT`` and at least one,
        which the shortest TR holds.

    Returns
    -------
    pypulseqpp.Sequence
        The designed sequence.

    Raises
    ------
    ValueError
        If ``excitation``, ``ordering`` or ``flip_modulation`` is unknown, a
        partial Fourier fraction is outside ``[0.75, 1]``, a count is below
        one, ``caipi_shift`` is outside ``[0, rz)``, ``wave`` names no wave
        mode, individually parameterized trains are shuffled, a train ends
        before the TE echo, or a TR is shorter than the longest train.
    ImportError
        If a variable train is asked for without blochsim.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> seq = sequences.fse3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, etl=8, te=20e-3, tr=300e-3
    ... )
    >>> seq.check_timing()[0]
    True
    >>> busse = sequences.fse3D_sequence(
    ...     n_x=32, n_y=16, n_z=8, etl=20, te=None, tr=0.5, n_dummy=0,
    ...     flip_modulation="busse", busse_optimize=False
    ... )
    >>> busse.check_timing()[0]
    True
    """
    if ordering == "adaptive_radial":
        ordering = "buonincontri"
    busse = flip_modulation == "busse"
    pd = flip_modulation == "pd"
    prescribed = busse or pd
    if prescribed and te is not None:
        raise ValueError(
            "Prescribed EPG trains require te=None; prescribe an effective TE or busse_centre_echo instead"
        )
    if busse_effective_te is not None and not busse:
        raise ValueError("busse_effective_te requires flip_modulation='busse'")
    if busse_optimize and not busse:
        raise ValueError("busse_optimize requires flip_modulation='busse'")
    if pd_optimize and not pd:
        raise ValueError("pd_optimize requires flip_modulation='pd'")
    if pd_effective_te is not None and not pd:
        raise ValueError("pd_effective_te requires flip_modulation='pd'")
    startup = busse_startup_echoes if prescribed else 0
    busse_angles = None
    controls_varied = False
    if crusher_cycles < 0:
        raise ValueError("crusher_cycles must be nonnegative")
    if optimize_end_angle and flip_modulation != "optimized":
        raise ValueError("End-angle optimization requires optimized refocusing")
    system = pp.cap_system(
        pp.Opts() if system is None else system,
        max_grad=MAX_GRAD,
        max_slew=MAX_SLEW,
    )
    for name, value, allowed in (
        ("excitation", excitation, EXCITATIONS),
        ("ordering", ordering, ORDERINGS),
        ("flip_modulation", flip_modulation, MODULATIONS),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    for name, fraction in (
        ("partial_fourier_x", partial_fourier_x),
        ("partial_fourier_y", partial_fourier_y),
        ("partial_fourier_z", partial_fourier_z),
    ):
        if not 0.75 <= fraction <= 1.0:
            raise ValueError(f"{name} must lie in [0.75, 1], got {fraction}")
    etl_periphery = etl if etl_periphery is None else etl_periphery
    for name, count in (
        ("ry", ry),
        ("rz", rz),
        ("etl", etl),
        ("etl_periphery", etl_periphery),
    ):
        if count < 1:
            raise ValueError(f"{name} must be at least 1, got {count}")
    if not 0 <= caipi_shift < rz:
        raise ValueError(f"caipi_shift must lie in [0, {rz}), got {caipi_shift}")
    fov = (fov_x, fov_y, fov_z)
    matrix = (n_x, n_y, n_z)
    etl_max = max(etl, etl_periphery)
    if busse:
        if busse_centre_echo is not None and (
            not isinstance(busse_centre_echo, (int, np.integer))
            or isinstance(busse_centre_echo, (bool, np.bool_))
            or not 0 <= busse_centre_echo < etl
        ):
            raise ValueError(
                "busse_centre_echo must be an acquired echo index in [0, etl)"
            )
        busse_centre = (
            (etl - 1) // 2 if busse_centre_echo is None else busse_centre_echo
        )
        centre_controls = np.array(
            [busse_minimum_angle_deg, busse_centre_angle_deg, busse_maximum_angle_deg]
        )
        periphery_controls = np.array(
            [
                busse_minimum_angle_deg
                if busse_minimum_angle_periphery_deg is None
                else busse_minimum_angle_periphery_deg,
                busse_centre_angle_deg
                if busse_centre_angle_periphery_deg is None
                else busse_centre_angle_periphery_deg,
                busse_maximum_angle_deg
                if busse_maximum_angle_periphery_deg is None
                else busse_maximum_angle_periphery_deg,
            ]
        )
        controls_varied = not np.array_equal(centre_controls, periphery_controls)
        busse_protocol = {
            "minimum_flip_angle": np.deg2rad(busse_minimum_angle_deg),
            "centre_flip_angle": np.deg2rad(busse_centre_angle_deg),
            "maximum_flip_angle": np.deg2rad(busse_maximum_angle_deg),
            "centre_echo": startup + busse_centre,
            "startup_echoes": startup,
            "initial_decay": busse_initial_decay,
        }

    if pd:
        pd_controls = np.array(
            [
                pd_first_angle_deg,
                pd_maximum_angle_deg,
                pd_final_angle_deg,
                pd_plateau_fraction,
            ]
        )
        pd_periphery_controls = np.array(
            [
                pd_first_angle_deg
                if pd_first_angle_periphery_deg is None
                else pd_first_angle_periphery_deg,
                pd_maximum_angle_deg
                if pd_maximum_angle_periphery_deg is None
                else pd_maximum_angle_periphery_deg,
                pd_final_angle_deg
                if pd_final_angle_periphery_deg is None
                else pd_final_angle_periphery_deg,
                pd_plateau_fraction
                if pd_plateau_fraction_periphery is None
                else pd_plateau_fraction_periphery,
            ]
        )
        controls_varied = not np.array_equal(pd_controls, pd_periphery_controls)

    # An even number of block rasters puts each hard pulse centre on the raster.
    raster = system.block_duration_raster
    hard = 2 * raster * math.ceil(HARD_PULSE_DURATION / (2 * raster) - 1e-9)
    if excitation == "slab":
        exc = sequences.SpatialSelectiveExcitation(
            system,
            90.0,
            fov_z,
            duration_s=PULSE_DURATION,
            time_bw_product=TIME_BW_PRODUCT,
            is_slab=True,
        )
    else:
        exc = sequences.NonSelectiveExcitation(system, 90.0, duration_s=hard)
    rf_ref = sequences.NonSelectiveRefocusing(system, duration_s=hard).rf_ref
    gz_ref, read_crushing = None, crusher_cycles
    gz = getattr(exc, "gz", None)
    fse = sequences.FseReadout3D(
        system,
        exc.rf,
        gz,
        rf_ref=rf_ref,
        gz_ref=gz_ref,
        fov=fov,
        matrix=matrix,
        etl=etl_max + startup,
        esp=esp,
        partial_echo=partial_fourier_x,
        oversampling=readout_oversampling,
        readout_bandwidth_hz=readout_bandwidth_hz,
        spoiling_cycles=read_crushing,
        wave=wave,
        wave_cycles=wave_cycles,
        wave_amplitude=wave_amplitude,
    )
    if busse:
        if busse_effective_te is not None and busse_centre_echo is None:
            from pypulseqpp._fse_contrast import _select_busse_centre

            busse_centre = _select_busse_centre(
                etl,
                etl_periphery,
                startup,
                centre_controls,
                periphery_controls,
                busse_initial_decay,
                fse.esp,
                float(fse.echo_times[0]),
                busse_effective_te,
            )
        if min(etl, etl_periphery) <= busse_centre + 1:
            raise ValueError(
                "busse_centre_echo must precede the final acquired echo of every train"
            )
        busse_protocol["centre_echo"] = startup + busse_centre
        busse_angles = pp.make_busse_schedule(etl + startup, **busse_protocol)
        pp.make_busse_schedule(
            etl_periphery + startup,
            **{
                **busse_protocol,
                "minimum_flip_angle": np.deg2rad(periphery_controls[0]),
                "centre_flip_angle": np.deg2rad(periphery_controls[1]),
                "maximum_flip_angle": np.deg2rad(periphery_controls[2]),
            },
        )
        prescribed_effective_te = float(
            pp.calc_fse_effective_te(
                busse_angles[: startup + busse_centre + 1],
                fse.esp,
                first_echo_time=float(fse.echo_times[0]),
            )[-1]
        )

    if pd:
        from pypulseqpp._pd import _pd_schedule

        def pd_design(count, controls):
            return _pd_schedule(
                count + startup,
                *controls,
                startup,
                fse.esp,
                float(fse.echo_times[0]),
                1.0,
                0.1,
                busse_initial_decay,
            )

        pd_angles, pd_levels = pd_design(etl, pd_controls)
        _, pd_periphery_levels = pd_design(etl_periphery, pd_periphery_controls)
        equivalent = pp.calc_fse_effective_te(
            pd_angles, fse.esp, first_echo_time=float(fse.echo_times[0])
        )
        pd_centre = 0 if busse_centre_echo is None else busse_centre_echo
        if pd_effective_te is not None and busse_centre_echo is None:
            if not np.isfinite(pd_effective_te) or pd_effective_te <= 0:
                raise ValueError("pd_effective_te must be positive and finite")
            pd_centre = int(
                np.argmin(
                    abs(
                        equivalent[startup : startup + min(etl, etl_periphery)]
                        - pd_effective_te
                    )
                )
            )
        if (
            not isinstance(pd_centre, (int, np.integer))
            or isinstance(pd_centre, (bool, np.bool_))
            or not 0 <= pd_centre < min(etl, etl_periphery)
        ):
            raise ValueError("The PD centre must be an acquired echo of every train")
        prescribed_pd_te = float(equivalent[startup + pd_centre])

    nominal = fse.rf_ref.amplitude
    waves = [
        gradient
        for gradient in (
            getattr(fse, "gy_wave", None),
            getattr(fse, "gz_wave", None),
        )
        if gradient is not None
    ]
    no_waves = [pp.scale_grad(g, 0.0) for g in waves]
    te_echo = (
        busse_centre
        if busse
        else pd_centre
        if pd
        else 0
        if te is None
        else int(np.argmin(np.abs(fse.echo_times - te)))
    )
    if ordering == "busse_radial" and te_echo != 0:
        raise ValueError(
            "busse_radial requires centre echo 0; set te=None and busse_centre_echo=0 for Busse trains"
        )
    if min(etl, etl_periphery) <= te_echo:
        raise ValueError(
            f"a train of {min(etl, etl_periphery)} echoes ends before echo "
            f"{te_echo + 1}, which the TE of {te * 1e3:.1f} ms falls on"
        )

    spiral_navigator = (
        sequences.SpiralNavigator(system, navigator_tr=NAVIGATOR_TR)
        if navigator
        else None
    )
    # The shortest TR holds the train and, where asked, one navigator.
    minimum_tr = (
        fse.duration
        + (spiral_navigator.duration if navigator else 0.0)
        + system.block_duration_raster
    )
    tr = minimum_tr if tr is None else tr
    tr_periphery = tr if tr_periphery is None else tr_periphery
    # Every train holds the longest one's echoes, so both ends of the
    # transition are refused here, before the shots are dealt: a design of
    # a few views, as an evaluation makes, refuses what the whole scan would.
    longest = fse.duration + system.block_duration_raster
    for time in (tr, tr_periphery):
        if pp.round_to_raster(time, system.block_duration_raster) < longest - 1e-9:
            raise ValueError(
                f"the TR of {time * 1e3:.1f} ms is shorter than the "
                f"{longest * 1e3:.1f} ms the longest train takes"
            )
    individual = tr_periphery != tr or etl_periphery != etl or controls_varied
    resolved_ordering = (
        ("buonincontri" if individual else "pclo") if ordering == "radial" else ordering
    )
    adaptive = resolved_ordering in ("buonincontri", "buonincontri_golden")
    if individual and not adaptive:
        raise ValueError("Varying trains require Buonincontri adaptive radial ordering")
    if not individual and adaptive:
        raise ValueError(
            "Buonincontri ordering requires a centre/periphery parameter transition"
        )
    repetition_time, periphery_repetition_time = (
        pp.round_to_raster(time, system.block_duration_raster)
        for time in (tr, tr_periphery)
    )
    if n_dummy is None:
        n_dummy = sequences.steady_state_dummies(repetition_time, 90.0)

    calibrating, imaging = pp.make_cartesian_plane_sampling(
        (n_y, n_z),
        (ry, rz),
        (n_acs_y, n_acs_z),
        caipi_shift=caipi_shift,
        partial_fourier=(partial_fourier_y, partial_fourier_z),
        elliptical=True,
        elliptical_acs=elliptical_acs,
        sampling="poisson" if ordering == "shuffling" else "lattice",
        seed=SHUFFLE_SEED,
        # Identical trains are all full; individually parameterized ones end
        # where each shot's own length does.
        train_length=1 if individual else etl,
    )
    calibration = set(calibrating)
    sampled = sorted({*calibrating, *imaging})
    coords = (np.asarray(sampled, dtype=float) - [n_y // 2, n_z // 2]) / [n_y, n_z]
    lengths, times, place = shot_parameters(
        len(sampled), etl, etl_periphery, tr, tr_periphery
    )
    raster = system.block_duration_raster
    times = [pp.round_to_raster(time, raster) for time in times]
    if not individual:
        # Identical trains sit nowhere along a transition.
        place = np.zeros_like(place)
    if ordering == "shuffling":
        views = np.asarray(sampled) - [n_y // 2, n_z // 2]
        order = pp.make_complementary_order(views, etl, seed=SHUFFLE_SEED, pad=True)
        lengths = [etl] * len(order)
        times = [repetition_time] * len(order)
        place = np.zeros(len(order))
    elif ordering == "busse_radial":
        order = pp.make_centric_order(coords, etl, pad=True)
    elif resolved_ordering == "pclo":
        order = pp.make_pclo_order(coords, etl, center_echo=te_echo, pad=True)
    else:
        order = deal_trains(
            coords,
            lengths,
            place,
            te_echo,
            etl_max,
            golden=resolved_ordering == "buonincontri_golden",
        )
    trains = [[None if i is None else sampled[i] for i in train] for train in order]

    # Refocusing angles, one row per shot; past its own length a shot's
    # pulses play at zero amplitude.
    played = np.arange(etl_max)[None, :] < np.asarray(lengths)[:, None]
    if busse:
        endpoint_controls = np.array([centre_controls, periphery_controls])
        if busse_optimize:
            from pypulseqpp._busse_opt import optimize_controls

            endpoints = (
                ((etl, repetition_time), (etl_periphery, periphery_repetition_time))
                if individual
                else ((etl, repetition_time),)
            )
            for endpoint, (count, period) in enumerate(endpoints):
                if individual:
                    representative = pp.make_pclo_order(
                        coords, count, center_echo=te_echo, pad=True
                    )
                else:
                    representative = order
                echo_map = np.full((n_y, n_z), -1, dtype=np.int32)
                for train in representative:
                    for echo, index in enumerate(train):
                        if index is not None:
                            echo_map[sampled[index]] = echo
                controls = np.deg2rad(endpoint_controls[endpoint])
                fitted = optimize_controls(
                    count + startup,
                    **{
                        **busse_protocol,
                        "minimum_flip_angle": controls[0],
                        "centre_flip_angle": controls[1],
                        "maximum_flip_angle": controls[2],
                    },
                    esp=fse.esp,
                    first_echo_time=float(fse.echo_times[0]),
                    tr=period,
                    echo_map=echo_map,
                    t1_ms=DESIGN_T1_MS,
                    t2_ms=DESIGN_T2_MS,
                    b1_scales=DESIGN_B1_SCALES,
                    contrast_pair=DESIGN_CONTRAST,
                    objective="contrast" if individual and endpoint == 0 else "psf",
                    effective_te=(
                        prescribed_effective_te
                        if busse_effective_te is not None and endpoint == 0
                        else None
                    ),
                )
                endpoint_controls[endpoint] = np.rad2deg(
                    fitted[[np.argmin(fitted), startup + te_echo, -1]]
                )
                if not individual:
                    busse_angles = fitted
        if individual:
            from pypulseqpp._busse import _make_busse_schedules

            controls = endpoint_controls[0] + cubic(place)[:, None] * (
                endpoint_controls[1] - endpoint_controls[0]
            )
            prescriptions, inverse = np.unique(
                np.column_stack((np.asarray(lengths) + startup, controls)),
                axis=0,
                return_inverse=True,
            )
            profiles = _make_busse_schedules(
                prescriptions[:, 0].astype(int),
                prescriptions[:, 1:],
                startup + te_echo,
                busse_initial_decay,
            )
            profiles = np.pad(
                profiles, ((0, 0), (0, etl_max + startup - profiles.shape[1]))
            )
            shot_flips = np.rad2deg(profiles[inverse])
        else:
            endpoint_controls[1] = endpoint_controls[0]
            shot_flips = np.tile(np.rad2deg(busse_angles), (len(trains), 1))
        trains = [[None] * startup + train for train in trains]
    elif pd:
        from pypulseqpp._pd import _make_pd_schedules

        pd_endpoint_controls = np.array([pd_controls, pd_periphery_controls])
        pd_endpoint_levels = np.array([pd_levels, pd_periphery_levels])
        if pd_optimize:
            from pypulseqpp._pd_opt import optimize_pd_controls

            endpoints = (
                ((etl, repetition_time), (etl_periphery, periphery_repetition_time))
                if individual
                else ((etl, repetition_time),)
            )
            for endpoint, (count, period) in enumerate(endpoints):
                representative = (
                    pp.make_pclo_order(coords, count, center_echo=te_echo, pad=True)
                    if individual
                    else order
                )
                echo_map = np.full((n_y, n_z), -1, dtype=np.int32)
                for train in representative:
                    for echo, view in enumerate(train):
                        if view is not None:
                            echo_map[sampled[view]] = echo
                fitted_controls, fitted, fitted_levels = optimize_pd_controls(
                    count + startup,
                    pd_endpoint_controls[endpoint],
                    startup=startup,
                    esp=fse.esp,
                    first_echo=float(fse.echo_times[0]),
                    tr=period,
                    centre=startup + te_echo,
                    echo_map=echo_map,
                    t1_ms=DESIGN_T1_MS,
                    t2_ms=DESIGN_T2_MS,
                    b1_scales=DESIGN_B1_SCALES,
                    contrast_pair=DESIGN_CONTRAST,
                    objective="contrast" if individual and endpoint == 0 else "psf",
                    effective_te=prescribed_pd_te
                    if pd_effective_te is not None and endpoint == 0
                    else None,
                    initial_decay=busse_initial_decay,
                )
                pd_endpoint_controls[endpoint] = fitted_controls
                pd_endpoint_levels[endpoint] = fitted_levels
                if not individual:
                    pd_angles = fitted
        if individual:
            fraction = cubic(place)[:, None]
            controls = pd_endpoint_controls[0] + fraction * (
                pd_endpoint_controls[1] - pd_endpoint_controls[0]
            )
            seed = pd_endpoint_levels[0] + fraction * (
                pd_endpoint_levels[1] - pd_endpoint_levels[0]
            )
            profiles = _make_pd_schedules(
                np.asarray(lengths) + startup,
                controls,
                startup,
                fse.esp,
                float(fse.echo_times[0]),
                decay=busse_initial_decay,
                seed=seed,
            )
            shot_flips = np.rad2deg(profiles)
        else:
            pd_endpoint_controls[1] = pd_endpoint_controls[0]
            shot_flips = np.tile(np.rad2deg(pd_angles), (len(trains), 1))
        trains = [[None] * startup + train for train in trains]
    elif flip_modulation == "optimized":
        shot_flips = design_trains(
            refocusing_angle_deg,
            individual,
            lengths,
            times,
            place,
            te_echo,
            etl_max,
            fse.esp,
            first_refocusing_angle_deg=180.0,
            first_echo_time=float(fse.echo_times[0]),
            b1_scales=DESIGN_B1_SCALES,
            optimize_end_angle=optimize_end_angle,
        )
    else:
        shot_flips = refocusing_angle_deg * played.astype(float)
        shot_flips[:, 0] = 180.0

    # A wave-encoded view calibrates nothing, so with the wave on the
    # calibration region is acquired again wave-free, in trains of its own
    # played as the first shot is.
    reference = []
    if waves:
        region = [v for v in sampled if v in calibration]
        if not individual:
            # The last reference train is filled with the views nearest the
            # centre outside the region.
            nearest = sorted(
                (v for v in sampled if v not in calibration),
                key=lambda v: (
                    ((v[0] - n_y // 2) / n_y) ** 2 + ((v[1] - n_z // 2) / n_z) ** 2,
                    v,
                ),
            )
            region += nearest[: -len(region) % etl]
        for start in range(0, len(region), lengths[0]):
            chunk = region[start : start + lengths[0]]
            reference.append([None] * startup + chunk + [None] * (etl_max - len(chunk)))

    # Navigators ride after each train, where they cost no scan time; the
    # closing delay makes every shot's own TR.
    shortest = min(times)
    if shortest < fse.duration + raster - 1e-9:
        raise ValueError(
            f"the TR of {shortest * 1e3:.1f} ms is shorter than the "
            f"{(fse.duration + raster) * 1e3:.1f} ms the longest train takes"
        )
    n_navigators, navigating = 0, 0.0
    if navigator:
        n_navigators = spiral_navigator.fit(
            shortest - fse.duration - raster + 1e-9, "auto", limit=NAVIGATOR_COUNT
        )
        if n_navigators == 0:
            raise ValueError(
                f"the TR of {shortest * 1e3:.1f} ms leaves no room after the "
                f"{fse.duration * 1e3:.1f} ms train for the "
                f"{spiral_navigator.duration * 1e3:.1f} ms navigator"
            )
        navigating = n_navigators * spiral_navigator.duration
    closings = [
        pp.round_to_raster(time - fse.duration - navigating, raster) for time in times
    ]

    seq, labels = pp.Sequence(system), sequences.Labels()

    def kernel(views, flips, closing: float, kind: str = "image") -> None:
        """Add one echo train over ``views`` at refocusing angles ``flips`` (degrees).

        ``None`` plays an echo unencoded and unacquired, and ``closing`` is the
        delay after the train. A ``dummy`` train acquires nothing and a
        ``reference`` train plays without the wave.
        """
        wave_events = no_waves if kind == "reference" else waves
        if kind == "dummy":
            flags = {"ONCE": 1}
        else:
            flags = {"ONCE": 0} if n_dummy else {}
            if waves:
                flags["REF"] = int(kind == "reference")

        fse.layout.excitation(seq, list(labels(**flags)))
        previous_ky = previous_kz = 0.0
        for echo, (view, flip) in enumerate(zip(views, flips, strict=True)):
            fse.rf_ref.amplitude = nominal * float(flip) / 180.0
            fse.layout.refocusing(seq, echo == 0, previous_ky, previous_kz)
            if echo == 0 and fse.esp_first > fse.esp:
                seq.add_block(pp.make_delay((fse.esp_first - fse.esp) / 2))
            line, partition = (n_y // 2, n_z // 2) if view is None else view
            ky = (line - n_y // 2) / (n_y / 2)
            kz = (partition - n_z // 2) / (n_z / 2)
            fse.layout.encoding(seq, ky, kz)
            if kind != "dummy" and view is not None:
                is_calibration = not waves and view in calibration
                counters = {"LIN": line, "PAR": partition, "IMA": is_calibration}
                if ordering == "shuffling":
                    counters["ECO"] = echo - startup
                label_events = labels(**counters)
                seq.add_block(fse.gx, *wave_events, fse.adc, *label_events)
            else:
                seq.add_block(fse.gx, *wave_events)
            fse.layout.rewinding(seq, ky, kz, echo == fse.etl - 1)
            previous_ky, previous_kz = ky, kz
        fse.rf_ref.amplitude = nominal
        for _ in range(n_navigators):
            for block in spiral_navigator.blocks:
                seq.add_block(*block)
        seq.add_block(pp.make_delay(closing))

    # The dummy trains, the wave-free reference trains, then every shot.
    blank = [None] * fse.etl
    for _ in range(n_dummy):
        kernel(blank, shot_flips[0], closings[0], "dummy")
    for views in reference:
        kernel(views, shot_flips[0], closings[0], "reference")
    for views, flips, closing in zip(trains, shot_flips, closings, strict=True):
        kernel(views, flips, closing)

    # The volume's offset is applied to the finished sequence with
    # pp.TransformFOV.
    definitions = {
        "FOV": list(fov),
        "Matrix": [n_x, n_y, n_z],
        "Name": NAME,
        "TE": float(fse.echo_times[te_echo + startup]),
        "TR": repetition_time,
        "EchoSpacing": fse.esp,
        "FirstEchoSpacing": fse.esp_first,
        "RefocusingEndOptimized": int(optimize_end_angle),
        "EchoTrainLength": etl,
        "Excitation": excitation,
        "ViewOrdering": ordering,
        "ResolvedViewOrdering": resolved_ordering,
        "FlipModulation": flip_modulation,
        "RefocusingFlipAngles": [
            float(f) for f in shot_flips[0][: lengths[0] + startup]
        ],
        "NumShots": len(trains),
        "kSpaceCenterLine": n_y // 2,
        "kSpaceCenterPartition": n_z // 2,
        "kSpaceCenterSample": fse.center_sample,
        "SliceThickness": fov[2],
    }
    if busse:
        definitions.update(
            EffectiveEchoTime=float(
                pp.calc_fse_effective_te(
                    np.deg2rad(shot_flips[0][: startup + te_echo + 1]),
                    fse.esp,
                    first_echo_time=float(fse.echo_times[0]),
                )[-1]
            ),
            EffectiveEchoTimeReferenceT1=1.0,
            EffectiveEchoTimeReferenceT2=0.1,
            StartupEchoes=startup,
            BusseControlAngles=[
                busse_minimum_angle_deg,
                busse_centre_angle_deg,
                busse_maximum_angle_deg,
            ],
            BusseCentreEcho=te_echo,
            BusseInitialDecay=busse_initial_decay,
            BusseOptimized=int(busse_optimize),
            BusseEndpointControlAngles=endpoint_controls.ravel().tolist(),
            RefocusingFlipAnglesPeriphery=[
                float(f) for f in shot_flips[-1][: lengths[-1] + startup]
            ],
        )
    if pd:
        definitions.update(
            EffectiveEchoTime=float(
                pp.calc_fse_effective_te(
                    np.deg2rad(shot_flips[0][: startup + te_echo + 1]),
                    fse.esp,
                    first_echo_time=float(fse.echo_times[0]),
                )[-1]
            ),
            EffectiveEchoTimeReferenceT1=1.0,
            EffectiveEchoTimeReferenceT2=0.1,
            StartupEchoes=startup,
            PDCentreEcho=te_echo,
            PDEndpointControls=pd_endpoint_controls.ravel().tolist(),
            PDControlColumns="first_deg maximum_deg final_deg plateau_fraction",
            PDOptimized=int(pd_optimize),
            PDPeakRfPulse=int(np.argmax(shot_flips[0][: lengths[0] + startup])) + 1,
            RefocusingFlipAnglesPeriphery=[
                float(f) for f in shot_flips[-1][: lengths[-1] + startup]
            ],
        )
    if individual:
        definitions["TRPeriphery"] = periphery_repetition_time
        definitions["EchoTrainLengthPeriphery"] = etl_periphery
    for key, value in definitions.items():
        seq.set_definition(key=key, value=value)
    seq.set_definition(
        key="NumGainCalibrationReadouts",
        value=sequences.gain_calibration_readouts(seq),
    )
    return seq


main = fse3d

if __name__ == "__main__":
    raise SystemExit(cli.run(main, sys.argv[1:], default_output="fse_3d.seq"))
