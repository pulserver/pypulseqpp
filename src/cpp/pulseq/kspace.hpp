/**
 * @file kspace.hpp
 * @brief Integrate piecewise-linear physical gradients into k-space trajectories.
 *
 * Excitation resets k-space and refocusing reverses it at the RF centre.
 */

#ifndef PULSEQ_KSPACE_HPP
#define PULSEQ_KSPACE_HPP

#include <array>
#include <vector>

#include "pulseq/sequence.hpp"
#include "pulseq/waveforms.hpp"

namespace pulseq
{

    /** Parameters the k-space trajectory is integrated under. */
    struct KspaceOptions
    {
        /**
         * Timing correction in seconds, subtracted from gradient times (positive advances).
         */
        std::array<double, 3> delay{{0.0, 0.0, 0.0}};
        /** A background gradient per axis, in Hz/m. */
        std::array<double, 3> offset{{0.0, 0.0, 0.0}};

        /** Blocks to follow, 1-based inclusive; 0 for the last means the end. */
        int first_block = 1;
        int last_block = 0;

        double b0 = 1.5;
        double gamma = 42576000.0;

        /**
         * Omit the full trajectory grid and return only ADC-sampled positions.
         * Analytic integration between corners gives the same ADC positions in both modes.
         */
        bool samples_only = false;
    };

    /** The k-space trajectory and the quantities derived from it. */
    struct Kspace
    {
        /** Times in seconds at which the trajectory is evaluated. */
        std::vector<double> times;
        /** Per axis, the trajectory position at each of those times, in 1/m. */
        std::array<std::vector<double>, 3> position;

        /** Per axis, the k-space location of each ADC sample, in 1/m, and its time. */
        std::array<std::vector<double>, 3> sampled;
        std::vector<double> adc_times;
        std::vector<double> adc_modulation;

        std::vector<double> excitation_times;
        std::vector<double> refocusing_times;

        /** Per axis, the slice position each excitation selected. */
        std::array<std::vector<double>, 3> slice_position;

        /**
         * The gradient each axis played, padded to the whole span: the
         * corners the trajectory was integrated from, for a caller that wants
         * them as a spline.
         */
        std::array<std::vector<double>, 3> gradient_times;
        std::array<std::vector<double>, 3> gradient_values;

        std::vector<std::string> warnings;
    };

    /**
     * Follow @p seq into k-space.
     *
     * @param seq      The sequence to follow.
     * @param options  Delays, background gradients and which blocks to follow.
     */
    Kspace calculate_kspace(const Sequence& seq, const KspaceOptions& options);

} // namespace pulseq

#endif /* PULSEQ_KSPACE_HPP */
