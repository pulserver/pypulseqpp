/**
 * @file simulate.hpp
 * @brief A sequence's blocks played on isochromats, and the events of one
 *        block as the Bloch equation takes them.
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

    /** A gradient per axis, as corner times from the block's start over
     *  amplitude in Hz/m; an axis without corners plays nothing. */
    struct GradientCorners
    {
        std::vector<double> times[3];
        std::vector<double> values[3];
    };

    /**
     * The gradients @p matrix turns @p given into, row i weighting the given
     * axes into output axis i. Each output axis is the sum on the union of the
     * given corners, where it is exact. A gradient is zero outside its
     * corners, so it steps there from or to a value other than zero; a step
     * is two corners at one time.
     *
     * @throws std::invalid_argument if a corner is not finite or the times of
     *         an axis decrease.
     */
    void rotate_gradients(const double matrix[3][3], const GradientCorners& given, GradientCorners& out);

    /** An RF pulse as the steps Isochromats::play takes: `channels` rows of
     *  `steps` values of b1, in Hz, channel-major, each held for `step` from
     *  `start` on. */
    struct PulseSteps
    {
        double start = 0.0;
        double step = 0.0;
        size_t steps = 0;
        size_t channels = 0;
        std::vector<std::complex<double>> rf;
    };

    /**
     * The field an RF pulse plays: the conjugate of the complex waveform
     * Pulseq defines, the samples @p waveform, in Hz, at @p times from the
     * pulse's start, times a carrier of the phase offset @p phase, in rad,
     * advancing at the frequency offset @p frequency, in Hz, from the pulse's
     * start, which is @p delay into the block. A frequency offset and the
     * same phase ramp in the waveform play one pulse, and a positive offset
     * excites isochromats precessing at a positive frequency.
     *
     * Samples at the middles of equal intervals from the pulse's start are
     * steps as they stand. The points of a time shape are joined linearly and
     * held over steps of @p raster, or of the shape's shortest interval where
     * that is shorter. A dynamic pTx pulse holds its channels one after
     * another over one time base, counted by rf_channels.
     *
     * @throws std::invalid_argument if the times and samples differ in
     *         number, a time is not finite, the times of a channel decrease,
     *         or the raster is not positive.
     */
    void pulse_steps(
        const std::vector<double>& times,
        const std::vector<std::complex<double>>& waveform,
        double delay,
        double phase,
        double frequency,
        double raster,
        PulseSteps& out);

    /** An ADC window's sample times from the block's start, and the phase in
     *  rad each sample is multiplied by exp(i phase) with. */
    struct AdcWindow
    {
        std::vector<double> times;
        std::vector<double> receiver;
    };

    /**
     * The window of an ADC of @p samples samples, @p dwell apart from
     * @p delay into the block, each at the middle of its dwell. Sample k is
     * demodulated by the phase offset @p phase plus @p modulation[k], where
     * given, advancing at the frequency offset @p frequency from the window's
     * start.
     */
    void adc_window(
        size_t samples,
        double dwell,
        double delay,
        double phase,
        double frequency,
        const std::vector<double>& modulation,
        AdcWindow& out);

    /**
     * Play the blocks of @p seq on @p isochromats and return every ADC sample,
     * coil-major: entry c * samples + k is coil c's sample k, in play order.
     *
     * Gradients are played after each block's rotation, with the isochromats'
     * positions along the same axes, as rotate_gradients sums them. An RF
     * pulse plays as pulse_steps takes it, over the sequence's RF raster, and
     * an ADC samples and demodulates as adc_window describes, with ppm
     * offsets resolved at the options' field.
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
