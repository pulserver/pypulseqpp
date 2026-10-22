/**
 * @file resonance.hpp
 * @brief Windowed gradient spectrum on the physical axes, judged against forbidden bands.
 */

#ifndef PULSEQ_RESONANCE_HPP
#define PULSEQ_RESONANCE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** A frequency range one physical axis, or every axis, must not be driven in. */
    struct ForbiddenBand
    {
        /** 0, 1, 2 for x, y, z; -1 for every axis. */
        int axis = -1;
        /** Edges in Hz, inclusive. */
        double f_min = 0.0;
        double f_max = 0.0;
        /** Amplitude threshold inside the band, in Hz/m. */
        double threshold = 0.0;
        /** With a positive value, in Hz/m: a window exceeding the threshold
         * violates the band only when the axis's gradient also exceeds this
         * over the window. */
        double gradient_limit = 0.0;
    };

    struct ResonanceOptions
    {
        /** Window length and the step between window starts, in seconds. */
        double window = 40e-3;
        double stride = 20e-3;
        /** Transform length as a multiple of the window's sample count. */
        int oversampling = 3;
        /** Prescription rotation, logical to physical after each block's own. */
        double rotation[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        /** MKL runtime library to transform a kept spectrum with; empty for pocketfft. */
        std::string mkl_runtime;
        /** Window whose spectrum to keep, for a diagnostic; -1 keeps none. */
        int64_t keep_spectrum = -1;
        /** A window exceeding a band is read again over a window this many
         * times longer; 1 or less reads each window once. */
        int reread = 4;
        /** Fraction of its amplitude a line must keep on the longer read for
         * the exceeding window to count as leakage. */
        double reread_keep = 0.8;
    };

    /** One forbidden band on one axis: its worst window and its violation count.
     * The worst window is the strongest violating one when any violates, and
     * the strongest reading otherwise. */
    struct BandReading
    {
        int band = 0;
        int axis = 0;
        /** Largest amplitude inside the band over all windows, in Hz/m, with its frequency. */
        double peak = 0.0;
        double frequency = 0.0;
        int64_t window = 0;
        double window_start = 0.0;
        /** The axis's largest gradient magnitude over that window, in Hz/m. */
        double gradient = 0.0;
        int64_t violations = 0;
        /** Windows that exceeded the band but read as leakage on the longer read. */
        int64_t leakage = 0;
    };

    struct ResonanceReport
    {
        /** Band order, then x, y, z within a band that guards every axis. */
        std::vector<BandReading> readings;
        /** Per band, the windows in which any axis it guards exceeds its threshold. */
        std::vector<int64_t> band_violations;
        int64_t windows = 0;
        /** Window and stride as sampled, in seconds; bin spacing in Hz. */
        double window = 0.0;
        double stride = 0.0;
        double frequency_step = 0.0;
        std::string backend;
        /** With `keep_spectrum`: that window's amplitude per axis and bin, in
         * Hz/m, and where it starts, in seconds. Empty without it. */
        std::vector<std::vector<double>> spectrum;
        double spectrum_start = 0.0;
    };

    /**
     * Read every window of the physical-axis gradient against the bands.
     *
     * Each axis is sampled at the centres of the sequence's own gradient
     * raster, with block rotations and then the prescription rotation
     * applied. A window of N samples starts every stride; the last starts
     * where it covers the end of the sequence, zero-filled. Each window is
     * mean-subtracted, Hann-tapered and zero-padded to `oversampling * N`,
     * and a bin's amplitude is `2 |X_k| / sum(w)`: a sustained sinusoid of
     * amplitude A at a bin frequency reads A. A band is violated by a window
     * whose largest amplitude on a bin inside it exceeds its threshold; a band
     * narrower than a bin is read at the bin nearest its centre. A band with
     * a gradient limit is violated only by a window whose gradient on that
     * axis also exceeds the limit; the gradient's magnitude is bounded by the
     * logical channels' extremes turned onto the axis.
     *
     * A window exceeding a band is read again over a window `reread` times
     * longer centred on it, around the band widened by the short window's
     * main lobe (2 / window). When the strongest line there lies outside the
     * band and keeps `reread_keep` of the short reading, the excess is a line
     * outside the band leaking in through the short window, and it is counted
     * as leakage, not as a violation.
     *
     * Bins a band cannot exceed its threshold at are not read. A window's
     * reading at f is at most 3 V / (sum(w) sin(pi f dt)), V being the total
     * variation of the axis's samples over the window, which the blocks the
     * window overlaps bound from their corners. A band wider than a few
     * hundred bins is read only below the frequency where this bound, at the
     * largest V of any window, meets its threshold.
     *
     * The bins a band reads are summed directly rather than transformed.
     * A channel's sums are remembered from the second block that plays its
     * waveform and reused by every later block that plays it over as many
     * samples, its first sample at the same time from the block's start;
     * block start times are summed in whole picoseconds, so that time is
     * exact anywhere in the sequence.
     */
    ResonanceReport mech_resonance(
        const Sequence& seq,
        const std::vector<ForbiddenBand>& bands,
        const ResonanceOptions& options);

} // namespace pulseq

#endif /* PULSEQ_RESONANCE_HPP */
