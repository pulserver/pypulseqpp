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
        /** Largest amplitude allowed inside, in Hz/m. */
        double threshold = 0.0;
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
        /** MKL runtime library to transform with; empty for pocketfft. */
        std::string mkl_runtime;
    };

    /** One band on one axis: its worst window, and how many windows violate it. */
    struct BandReading
    {
        int band = 0;
        int axis = 0;
        /** Largest amplitude inside the band over all windows, Hz/m, and where. */
        double peak = 0.0;
        double frequency = 0.0;
        int64_t window = 0;
        double window_start = 0.0;
        int64_t violations = 0;
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
     * narrower than a bin is read at the bin nearest its centre.
     */
    ResonanceReport mech_resonance(
        const Sequence& seq,
        const std::vector<ForbiddenBand>& bands,
        const ResonanceOptions& options);

} // namespace pulseq

#endif /* PULSEQ_RESONANCE_HPP */
