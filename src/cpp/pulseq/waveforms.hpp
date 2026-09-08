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

        std::vector<PulseMoment> excitation;
        std::vector<PulseMoment> refocusing;

        /**
         * Per ADC window rather than per sample: the offsets it was asked
         * for, as the sequence records them and without the ppm terms folded
         * in. What `adc_times` reports, where one window is one row.
         */
        std::vector<double> window_frequency;
        std::vector<double> window_phase;

        /** Every ADC sample: when it is taken, and the phase it is taken at. */
        std::vector<double> adc_times;
        std::vector<double> adc_frequency;
        std::vector<double> adc_phase;
        std::vector<double> adc_modulation;

        /** Blocks whose gradients a rotation remaps, which are not expanded. */
        std::vector<int> rotated_blocks;

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

    /**
     * Restore the corners of a gradient stored on the raster.
     *
     * A shape stored at the centre of each raster interval does not say what
     * the gradient is at the interval boundaries, and those are where its
     * slope changes. They follow from the samples and the recorded first and
     * last value: each boundary is twice the sample before it less the
     * boundary before that, which is exact when the shape really was sampled
     * from a piecewise-linear waveform. Where that recurrence drifts -- it
     * accumulates error, and the recorded last value is the check -- the
     * average of the neighbouring samples is used instead.
     *
     * Points the waveform passes straight through are dropped, so what comes
     * back is the corners and nothing else.
     *
     * @param waveform  The samples, at the centre of each raster interval.
     * @param first     The value at the start of the first interval.
     * @param last      The value at the end of the last.
     * @param raster    The gradient raster time.
     * @param times     Filled with the corner times, from zero.
     * @param values    Filled with the corner values.
     * @return False if the recurrence did not reach @p last, in which case
     *         the samples are returned with the edges added and nothing else.
     */
    bool restore_shape_corners(
        const std::vector<double>& waveform,
        double first,
        double last,
        double raster,
        std::vector<double>& times,
        std::vector<double>& values);

} // namespace pulseq

#endif /* PULSEQ_WAVEFORMS_HPP */
