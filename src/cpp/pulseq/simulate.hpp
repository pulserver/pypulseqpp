/**
 * @file simulate.hpp
 * @brief A sequence's blocks played on isochromats.
 */

#ifndef PULSEQ_SIMULATE_HPP
#define PULSEQ_SIMULATE_HPP

#include <complex>
#include <cstddef>
#include <vector>

#include "pulseq/bloch.hpp"
#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** Which blocks to play, and the field the ppm offsets are resolved at. */
    struct SimulationOptions
    {
        /** Blocks to play, 1-based inclusive; 0 for the last means to the end. */
        int first_block = 1;
        int last_block = 0;

        /** Field strength in tesla and gyromagnetic ratio in Hz/T. */
        double b0 = 1.5;
        double gamma = 42576000.0;
    };

    /**
     * Play the blocks of @p seq on @p isochromats and return every ADC sample,
     * coil-major: entry c * samples + k is coil c's sample k, in play order.
     *
     * Gradients are played after each block's rotation, with the isochromats'
     * positions along the same axes. An RF pulse is the conjugate of the
     * complex waveform Pulseq defines, its amplitude, magnitude and phase
     * shapes and a carrier of the phase offset advancing at the frequency
     * offset from the pulse's start, so that a frequency offset and the same
     * phase ramp in the shape play one pulse and a positive offset excites
     * isochromats precessing at a positive frequency. Each sample is
     * multiplied by exp(i theta), theta being the ADC phase offset plus its
     * phase modulation, advancing at its frequency offset from the window's
     * start.
     *
     * A pTx pulse plays one channel per transmit sensitivity. With no
     * transmit sensitivities its channels are summed and an RF shim is not
     * applied: the flip angle is the one Sequence::rf_flip_angles reports.
     * With them, a single-channel pulse plays on every channel, weighted by
     * the block's RF shim where it has one.
     */
    std::vector<std::complex<double>> simulate(
        const Sequence& seq, Isochromats& isochromats, const SimulationOptions& options);

} // namespace pulseq

#endif /* PULSEQ_SIMULATE_HPP */
