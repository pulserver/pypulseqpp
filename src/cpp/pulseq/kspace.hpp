/**
 * @file kspace.hpp
 * @brief Where the sequence goes in k-space, and where it samples.
 *
 * A gradient moves the spins' phase, and the phase they have accumulated is
 * where the sequence has got to in k-space -- so the trajectory is the
 * integral of the gradient waveforms. Between two corners a gradient is a
 * straight line and its integral is a parabola, so the trajectory is exact
 * between the corners rather than sampled at them.
 *
 * Two things reset it. An excitation starts the phase over, so k returns to
 * the origin; a refocusing turns the accumulated phase around, so k reflects
 * through it, which is what makes a spin echo come back. The trajectory is
 * therefore a run of periods separated by the pulses, each shifted to begin
 * where the pulse before it left off.
 */

#ifndef PULSEQ_KSPACE_HPP
#define PULSEQ_KSPACE_HPP

#include <array>
#include <vector>

#include "pulseq/sequence.hpp"
#include "pulseq/waveforms.hpp"

namespace pulseq
{

    /** What the trajectory is followed under. */
    struct KspaceOptions
    {
        /** How late each axis plays what it was asked to, in seconds. */
        std::array<double, 3> delay{{0.0, 0.0, 0.0}};
        /** A background gradient per axis, in Hz/m. */
        std::array<double, 3> offset{{0.0, 0.0, 0.0}};

        /** Blocks to follow, 1-based inclusive; 0 for the last means the end. */
        int first_block = 1;
        int last_block = 0;

        double b0 = 1.5;
        double gamma = 42576000.0;

        /**
         * Answer only where the samples are, not the whole trajectory.
         *
         * The trajectory is reported at every moment it changes direction,
         * and through every gradient ramp at the raster, so that a plot of it
         * is a plot of the gradient. A caller who wants where the samples
         * were taken -- a reconstruction, a description of the scan -- does
         * not need any of those, and building them is nearly all the work.
         *
         * The answer is the same either way: the trajectory between two
         * corners is a parabola and integrating it is exact, so a moment
         * reported in between changes nothing about a sample either side.
         */
        bool samples_only = false;
    };

    /** The trajectory, and everything read off it. */
    struct Kspace
    {
        /** Every moment the trajectory has to be known at. */
        std::vector<double> times;
        /** Per axis, where the trajectory is at each of those moments. */
        std::array<std::vector<double>, 3> position;

        /** Where each ADC sample sits, and when it is taken. */
        std::array<std::vector<double>, 3> sampled;
        std::vector<double> adc_times;
        std::vector<double> adc_modulation;

        std::vector<double> excitation_times;
        std::vector<double> refocusing_times;

        /** Per axis, where each excitation put its slice. */
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
