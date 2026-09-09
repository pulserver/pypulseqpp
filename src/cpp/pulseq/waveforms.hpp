/**
 * @file waveforms.hpp
 * @brief The sequence as what the gradients and the digitiser actually do.
 *
 * A block table says which events a sequence plays and when each block
 * starts. Everything that looks at what a sequence *does* -- where it goes in
 * k-space, how fast the gradients slew, when a sample is taken, what a plot
 * draws -- needs the other view: one waveform per axis over the whole scan,
 * on a time base shared by all of them.
 *
 * Building it is a pass over every block, and a block contributes a handful
 * of points rather than one, so this is the routine whose cost grows fastest
 * with the size of a scan. It is here, in C++, for that reason.
 *
 * ### A waveform is its corners
 *
 * A gradient is played by interpolating linearly between the samples it is
 * given, so the waveform is fully described by the points where its slope
 * changes and nothing is lost by leaving out the rest. A trapezoid is four
 * points however long its flat top is; a shape stored on the raster is
 * restored to the corners the interpreter will draw between.
 */

#ifndef PULSEQ_WAVEFORMS_HPP
#define PULSEQ_WAVEFORMS_HPP

#include <array>
#include <complex>
#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** What to expand, and what the sequence is played on. */
    struct WaveformOptions
    {
        /** Also build the RF envelope, as a fourth channel. */
        bool append_rf = false;

        /** Blocks to expand, 1-based inclusive; 0 for the last means to the end. */
        int first_block = 1;
        int last_block = 0;

        /** Field strength in tesla and gyromagnetic ratio in Hz/T, for the
         *  ppm offsets, which are a fraction of the Larmor frequency. */
        double b0 = 1.5;
        double gamma = 42576000.0;
    };

    /**
     * One RF pulse's moment: when it acts, and at what frequency and phase.
     *
     * The time is the pulse's centre, which is where it is taken to act; the
     * phase is what it has accumulated by then.
     */
    struct PulseMoment
    {
        double time = 0.0;
        double frequency = 0.0;
        double phase = 0.0;
    };

    /** What a sequence does, in the order it does it. */
    struct Waveforms
    {
        /** Per gradient axis, the corners of the waveform: when, and how much. */
        std::array<std::vector<double>, 3> times;
        std::array<std::vector<double>, 3> amplitudes;

        /** The RF envelope, when it was asked for. */
        std::vector<double> rf_times;
        std::vector<std::complex<double>> rf_signal;

        /**
         * Every RF pulse, in play order, with what it is for and which block
         * plays it.
         *
         * Pulseq has seven uses and an answer that carries two of them drops
         * an inversion, a saturation and a preparation on the floor. They are
         * all here, tagged, and a caller wanting the two buckets sorts them
         * out by tag.
         */
        std::vector<PulseMoment> pulses;
        std::vector<char> pulse_uses;
        std::vector<int> pulse_blocks;

        /**
         * Per ADC window rather than per sample: the offsets it was asked
         * for, as the sequence records them and without the ppm terms folded
         * in. What `adc_times` reports, where one window is one row.
         */
        std::vector<double> window_frequency;
        std::vector<double> window_phase;
        /** Which block each window is in, and how many samples it takes. */
        std::vector<int> window_blocks;
        std::vector<int> window_samples;

        /** Every ADC sample: when it is taken, and the phase it is taken at. */
        std::vector<double> adc_times;
        std::vector<double> adc_frequency;
        std::vector<double> adc_phase;
        std::vector<double> adc_modulation;

        /**
         * How long the blocks expanded last, in total.
         *
         * The same running sum the waveform times are measured against, so a
         * caller asking whether an axis stops before the end is comparing two
         * numbers that were added up the same way. Adding the durations again
         * elsewhere gives a different last bit, and the answer flips.
         */
        double duration = 0.0;

        /** What the caller should be told: forced ramps and the like. */
        std::vector<std::string> warnings;
    };

    /**
     * Expand @p seq into the waveforms it plays.
     *
     * @param seq      The sequence to expand.
     * @param options  What to expand and what it is played on.
     * @return One waveform per axis, the RF moments, and the ADC sampling.
     */
    Waveforms waveforms_and_times(const Sequence& seq, const WaveformOptions& options);

} // namespace pulseq

#endif /* PULSEQ_WAVEFORMS_HPP */
