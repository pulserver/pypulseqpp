/**
 * @file waveforms.hpp
 * @brief Expand blocks into physical gradient corners, RF envelopes and ADC sampling.
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
     * RF centre time (seconds), frequency (Hz) and phase at the centre (radians).
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
         * All RF centres in play order, with use codes and 1-based block indices.
         * Includes uses other than excitation and refocusing.
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

        /**
         * Per-sample times in seconds, frequency in Hz, phase and modulation in radians.
         * Frequency and phase include ppm terms; modulation is stored separately.
         */
        std::vector<double> adc_times;
        std::vector<double> adc_frequency;
        std::vector<double> adc_phase;
        std::vector<double> adc_modulation;

        /**
         * Duration in seconds, accumulated in the same order as waveform times.
         * Use this value for endpoint comparisons to avoid roundoff discrepancies.
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
