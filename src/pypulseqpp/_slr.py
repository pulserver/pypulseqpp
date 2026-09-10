"""Shinnar-Le Roux pulse design in NumPy and SciPy.

The filter designs, the forward and inverse SLR transforms, the ripple
bookkeeping, root flipping and Leja ordering are derived from SigPy's
``sigpy.mri.rf.slr`` and ``sigpy.util.leja`` (Copyright (c) 2016, Frank Ong
and The Regents of the University of California; BSD 3-Clause, see
``LICENSES/SigPy-BSD-3-Clause.txt``), which transcribe John Pauly's
``rf_tools``. The exhaustive root-flip search is compiled, in
``pypulseqpp._ext.slr``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal


def _dinf(d1: float, d2: float) -> float:
    a1, a2, a3 = 5.309e-3, 7.114e-2, -4.761e-1
    a4, a5, a6 = -2.66e-3, -5.941e-1, -4.278e-1
    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)
    return float(
        (a1 * l10d1 * l10d1 + a2 * l10d1 + a3) * l10d2
        + (a4 * l10d1 * l10d1 + a5 * l10d1 + a6)
    )


def _calc_ripples(pulse_type: str, d1: float, d2: float) -> tuple[float, float, float]:
    if pulse_type == "st":
        scale = 1.0
    elif pulse_type == "ex":
        scale = np.sqrt(0.5)
        d1, d2 = np.sqrt(d1 / 2.0), d2 / np.sqrt(2.0)
    elif pulse_type == "se":
        scale = 1.0
        d1, d2 = d1 / 4.0, np.sqrt(d2)
    elif pulse_type == "inv":
        scale = 1.0
        d1, d2 = d1 / 8.0, np.sqrt(d2 / 2.0)
    elif pulse_type == "sat":
        scale = np.sqrt(0.5)
        d1, d2 = d1 / 2.0, np.sqrt(d2)
    else:
        raise ValueError("pulse_type must be one of 'st', 'ex', 'se', 'inv', or 'sat'")
    return float(scale), float(d1), float(d2)


def _least_squares(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    transition = _dinf(d1, d2) / tbw
    bands = np.asarray(
        [0.0, (1.0 - transition) * tbw / 2.0, (1.0 + transition) * tbw / 2.0, n / 2.0]
    ) / (n / 2.0)
    h = signal.firls(n + 1, bands, [1.0, 1.0, 0.0, 0.0], weight=[1.0, d1 / d2])
    # MATLAB's firls result is shifted by half a sample relative to SciPy.
    phase = np.exp(
        1j
        * 2.0
        * np.pi
        / (2.0 * (n + 1))
        * np.concatenate((np.arange(0, n / 2 + 1), np.arange(-n / 2, 0)))
    )
    return np.real(np.fft.ifft(np.fft.fft(h) * phase))[:n]


def _linear_phase(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    transition = _dinf(d1, d2) / tbw
    bands = (
        np.asarray(
            [
                0.0,
                (1.0 - transition) * tbw / 2.0,
                (1.0 + transition) * tbw / 2.0,
                n / 2.0,
            ]
        )
        / n
    )
    return signal.remez(n, bands, [1.0, 0.0], weight=[1.0, d1 / d2], fs=1.0)


def _minimum_phase_spectrum(magnitude: np.ndarray) -> np.ndarray:
    log_magnitude = np.log(np.maximum(np.abs(magnitude), np.finfo(float).tiny))
    spectrum = np.fft.fft(log_magnitude)
    causal = np.zeros_like(spectrum)
    causal[0] = spectrum[0]
    causal[1 : magnitude.size // 2] = 2.0 * spectrum[1 : magnitude.size // 2]
    causal[magnitude.size // 2] = spectrum[magnitude.size // 2]
    return np.exp(np.fft.ifft(causal))


def _factor_minimum_phase(h: np.ndarray) -> np.ndarray:
    length = h.size
    padded_length = int(128 * 2 ** np.ceil(np.log2(length)))
    pad_left = int(np.ceil((padded_length - length) / 2.0))
    pad_right = int(np.floor((padded_length - length) / 2.0))
    # Centred, so the linear-phase filter's spectrum is its real, zero-phase
    # amplitude: lifted above zero, that has a spectral square root.
    padded = np.pad(h, (pad_left, pad_right))
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(padded)))
    positive = spectrum - np.min(np.real(spectrum)) * 1.000001
    minimum_phase = _minimum_phase_spectrum(np.sqrt(np.abs(positive)))
    impulse = np.fft.ifft(np.fft.ifftshift(np.conj(minimum_phase)))
    return impulse[: (length + 1) // 2]


def _minimum_phase(n: int, tbw: float, d1: float, d2: float) -> np.ndarray:
    n2 = 2 * n - 1
    transition = 0.5 * _dinf(2.0 * d1, 0.5 * d2 * d2) / tbw
    bands = (
        np.asarray(
            [
                0.0,
                (1.0 - transition) * tbw / 2.0,
                (1.0 + transition) * tbw / 2.0,
                n / 2.0,
            ]
        )
        / n
    )
    linear = signal.remez(
        n2, bands, [1.0, 0.0], weight=[1.0, 2.0 * d1 / (0.5 * d2 * d2)], fs=1.0
    )
    return _factor_minimum_phase(linear)


def _windowed_sinc(n: int, lobes: float) -> np.ndarray:
    x = np.arange(-n / 2, n / 2) / (n / 2)
    arg = lobes * 2.0 * np.pi * x + 1e-5
    result = np.sin(arg) / arg
    result *= 0.54 + 0.46 * np.cos(np.pi * x)
    return result * (4.0 * lobes / n)


def _beta_to_alpha(beta: np.ndarray) -> np.ndarray:
    n = beta.size
    padded = np.zeros(n * 16, dtype=np.complex128)
    padded[:n] = beta
    beta_spectrum = np.fft.fft(padded)
    peak = np.max(np.abs(beta_spectrum))
    if peak >= 1.0:
        beta_spectrum /= 1e-7 + peak
    alpha_spectrum = _minimum_phase_spectrum(np.sqrt(1.0 - np.abs(beta_spectrum) ** 2))
    alpha = np.fft.fft(alpha_spectrum) / padded.size
    return alpha[:n][::-1]


def _alpha_beta_to_rf(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = alpha.size
    rf = np.zeros(n, dtype=np.complex128)
    alpha = alpha.astype(np.complex128, copy=True)
    beta = beta.astype(np.complex128, copy=True)
    for index in range(n - 1, -1, -1):
        cosine = np.sqrt(1.0 / (1.0 + np.abs(beta[index] / alpha[index]) ** 2))
        sine = np.conj(cosine * beta[index] / alpha[index])
        rf[index] = 2.0 * np.arctan2(np.abs(sine), cosine) * np.exp(1j * np.angle(sine))
        if index > 0:
            alpha_tmp = cosine * alpha + sine * beta
            beta_tmp = -np.conj(sine) * alpha + cosine * beta
            alpha = alpha_tmp[1 : index + 1]
            beta = beta_tmp[:index]
    return rf


def _beta_to_rf(beta: np.ndarray, cancel_alpha_phase: bool) -> np.ndarray:
    alpha = _beta_to_alpha(beta)
    if cancel_alpha_phase:
        beta_spectrum = np.fft.fft(beta) * np.exp(
            -1j * np.angle(np.fft.fft(alpha[::-1]))
        )
        beta = np.fft.ifft(beta_spectrum)
    return _alpha_beta_to_rf(alpha, beta)


#: What each pulse type tips by, which a root-flipped beta is scaled to.
NOMINAL_FLIP = {"ex": np.pi / 2.0, "sat": np.pi / 2.0, "se": np.pi, "inv": np.pi}

#: Flippable roots beyond which the exhaustive search is over a million pulses.
MAX_ROOT_FLIP_CANDIDATES = 20

#: Samples a root-flipped pulse is designed at before it is resampled onto the
#: raster: finding the roots of beta costs the cube of its length.
ROOT_FLIP_SAMPLES = 256


def _leja(roots: np.ndarray) -> np.ndarray:
    """Order polynomial roots so the polynomial rebuilds from them accurately.

    Each root taken is the one farthest, as a product of distances, from those
    already taken (Lang and Frenzel, Rice University ECE TR93-08, 1993), which
    keeps the partial products of a sequential rebuild from over- or
    underflowing.
    """
    x = np.asarray(roots, dtype=np.complex128).ravel()
    n = x.size
    if n < 2:
        return x.copy()
    a = np.tile(x.reshape(1, n), (n + 1, 1))
    a[0, :] = np.abs(a[0, :])
    tmp = np.zeros(n + 1, dtype=np.complex128)
    index = int(np.argmax(np.abs(a[0, :])))
    if index != 0:
        tmp[:] = a[:, 0]
        a[:, 0] = a[:, index]
        a[:, index] = tmp
    taken = np.zeros(n, dtype=np.complex128)
    taken[0] = a[n - 1, 0]
    a[1, 1:] = np.abs(a[1, 1:] - taken[0])
    product = a[0, 0:n]
    for step in range(1, n - 1):
        product = np.multiply(product, a[step, :])
        index = int(np.argmax(np.abs(product[step:]))) + step
        if step != index:
            tmp[:] = a[:, step]
            a[:, step] = a[:, index]
            a[:, index] = tmp
            tmp[0] = product[step]
            product[step] = product[index]
            product[index] = tmp[0]
        taken[step] = a[n - 1, step]
        a[step + 1, (step + 1) : n] = np.abs(a[step + 1, (step + 1) :] - taken[step])
    return a[n, :]


def _flipped_pulse(roots, flips, target: float, n: int) -> np.ndarray:
    """Return the pulse whose beta has ``roots``, ``flips`` flipped, at peak ``target``."""
    roots = np.where(flips, 1.0 / np.conj(roots), roots)
    beta = np.zeros(n, dtype=np.complex128)
    beta[n - roots.size - 1 :] = np.poly(roots)
    padded = 1 << int(np.ceil(np.log2(16 * n)))
    beta *= target / np.abs(np.fft.fft(beta, padded)).max()
    return _beta_to_rf(beta, False)


def _root_flip(
    beta: np.ndarray, d1: float, pulse_type: str, time_bandwidth_product: float
) -> np.ndarray:
    """Return the lowest-peak pulse among every flip of beta's passband roots.

    Flipping a root ``r`` to ``1 / conj(r)`` leaves ``|beta|`` on the unit
    circle -- the slice profile -- as it was and moves only its phase, so which
    roots are flipped decides how the pulse's energy spreads in time and
    nothing about the slice it selects (Sharma, Lustig and Grissom, Magn Reson
    Med 75:227, 2016). The roots on the unit circle are the stopband's zeros and
    stay put; the passband's are the ones tried, every subset of them.
    """
    n = beta.size
    target = float(np.sin(NOMINAL_FLIP[pulse_type] / 2.0 + np.arctan(2.0 * d1) / 2.0))
    roots = _leja(np.roots(beta))
    candidates = (np.abs(1.0 - np.abs(roots)) > 0.004) & (
        np.abs(np.angle(roots)) < time_bandwidth_product / n * np.pi
    )
    count = int(candidates.sum())
    if count > MAX_ROOT_FLIP_CANDIDATES:
        raise ValueError(
            f"root flipping would search 2**{count} pulses, above "
            f"2**{MAX_ROOT_FLIP_CANDIDATES}; lower the time-bandwidth product"
        )
    from ._ext import slr

    flips, _ = slr.root_flip_search(roots, candidates, target, n)
    return _flipped_pulse(roots, flips, target, n)


def _resampled(rf: np.ndarray, n: int) -> np.ndarray:
    """Return ``rf`` at ``n`` samples over the same duration.

    Each sample of an SLR pulse is the rotation one dwell carries, so the
    interpolated waveform is scaled by the ratio of the dwells.
    """
    coarse = (np.arange(rf.size) + 0.5) / rf.size
    fine = (np.arange(n) + 0.5) / n
    return (rf.size / n) * (
        np.interp(fine, coarse, rf.real) + 1j * np.interp(fine, coarse, rf.imag)
    )


def design_slr(
    n: int,
    time_bandwidth_product: float,
    *,
    pulse_type: str = "st",
    filter_type: str = "ls",
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    cancel_alpha_phase: bool = False,
    root_flip: bool = False,
) -> np.ndarray:
    """Return a dimensionless SLR RF waveform."""
    if n < 8 or n % 2:
        raise ValueError("n must be an even integer >= 8")
    if time_bandwidth_product <= 0:
        raise ValueError("time_bandwidth_product must be > 0")
    if not 0 < passband_ripple < 1 or not 0 < stopband_ripple < 1:
        raise ValueError("passband_ripple and stopband_ripple must lie in (0, 1)")
    if root_flip and pulse_type not in NOMINAL_FLIP:
        raise ValueError(
            "root flipping needs a nominal flip: pulse_type 'ex', 'se', 'inv' or 'sat'"
        )
    if root_flip and cancel_alpha_phase:
        raise ValueError(
            "root flipping picks the pulse by its peak, which cancelling the "
            "alpha phase would then change"
        )

    # The waveform is dimensionless: neither the flip angle it will be scaled
    # to nor the slab it will select reaches this function, so the arguments
    # that do reach it are the ones an operator changes least. A console
    # re-running a plugin for a new echo time asks for a pulse it has already
    # designed.
    return _design(
        n,
        float(time_bandwidth_product),
        pulse_type,
        filter_type,
        float(passband_ripple),
        float(stopband_ripple),
        bool(cancel_alpha_phase),
        bool(root_flip),
    ).copy()


@lru_cache(maxsize=32)
def _design(
    n: int,
    time_bandwidth_product: float,
    pulse_type: str,
    filter_type: str,
    passband_ripple: float,
    stopband_ripple: float,
    cancel_alpha_phase: bool,
    root_flip: bool,
) -> np.ndarray:
    if root_flip and n > ROOT_FLIP_SAMPLES:
        coarse = _design(
            ROOT_FLIP_SAMPLES,
            time_bandwidth_product,
            pulse_type,
            filter_type,
            passband_ripple,
            stopband_ripple,
            cancel_alpha_phase,
            root_flip,
        )
        return _resampled(coarse, n)
    scale, d1, d2 = _calc_ripples(pulse_type, passband_ripple, stopband_ripple)
    if filter_type == "ls":
        beta = _least_squares(n, time_bandwidth_product, d1, d2)
    elif filter_type == "pm":
        beta = _linear_phase(n, time_bandwidth_product, d1, d2)
    elif filter_type == "min":
        beta = _minimum_phase(n, time_bandwidth_product, d1, d2)[::-1]
    elif filter_type == "max":
        beta = _minimum_phase(n, time_bandwidth_product, d1, d2)
    elif filter_type == "ms":
        beta = _windowed_sinc(n, time_bandwidth_product / 4.0)
    else:
        raise ValueError("filter_type must be one of 'ls', 'pm', 'min', 'max', or 'ms'")

    if pulse_type == "st":
        return np.asarray(beta, dtype=np.complex128)
    if root_flip:
        return _root_flip(
            scale * np.asarray(beta, dtype=np.complex128),
            d1,
            pulse_type,
            time_bandwidth_product,
        )
    return _beta_to_rf(scale * beta, cancel_alpha_phase)
