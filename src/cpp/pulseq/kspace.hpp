/**
 * @file kspace.hpp
 * @brief Integrate piecewise-linear gradients, after each block's rotation, into k-space.
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

    /**
     * Per readout, the axes its k-space moves along and the samples nearest
     * the centre of k-space, one entry per block that acquires, in play order.
     *
     * k-space is integrated as calculate_kspace() integrates it, after each
     * block's rotation. To bound memory it is integrated a range of blocks at
     * a time, a range ending before an excitation, or a pulse with no use
     * recorded, in a block that does not acquire once the range holds 2^17
     * samples; such a pulse resets k-space, so each sample is where
     * calculate_kspace() puts it. An axis moves when the readout's k-space
     * spans more than 1e-6 of its widest span along it.
     */
    struct AdcEchoes
    {
        /** 1-based block of each readout. */
        std::vector<int32_t> block;
        std::vector<int32_t> num_samples;
        /** Index of the readout's first sample among all ADC samples. */
        std::vector<int64_t> first_sample;
        /** Readouts x 3: 1 where the readout moves along x, y or z. */
        std::vector<uint8_t> moving;
        /**
         * Readouts x 2: the first and last 0-based sample no further from the
         * centre, over the moving axes, than the nearest sample plus 1% of the
         * larger k step beside it; -1 for a readout that does not move or has
         * fewer than two samples.
         */
        std::vector<int32_t> echo;
    };

    /** Find where each readout passes nearest the centre of k-space. */
    AdcEchoes adc_echoes(const Sequence& seq, const KspaceOptions& base);

} // namespace pulseq

#endif /* PULSEQ_KSPACE_HPP */
