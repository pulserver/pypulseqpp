/**
 * @file safety.hpp
 * @brief Physical-axis gradient amplitude, slew and boundary-continuity checks.
 *
 * Checks use reconstructed waveform corners and each block's rotation.
 */

#ifndef PULSEQ_SAFETY_HPP
#define PULSEQ_SAFETY_HPP

#include <array>
#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** Gradient hardware limits a sequence is checked against. */
    struct GradientLimits
    {
        /**
         * Per-axis limits in Hz/m and Hz/m/s. A non-positive value disables
         * that check.
         */
        double max_grad = 0.0;
        double max_slew = 0.0;
        /** Gradient raster period in seconds. */
        double grad_raster_time = 10e-6;
    };

    /** A peak value and the block containing it. */
    struct Peak
    {
        double value = 0.0;
        int block = 0;
        int axis = -1; /**< 0, 1, 2; -1 when the peak is a vector magnitude. */
    };

    /** A gradient amplitude discontinuity at a block boundary. */
    struct Discontinuity
    {
        int block = 0;
        int axis = 0;
        /** Amplitudes on either side of the boundary, in Hz/m. */
        double before = 0.0;
        double after = 0.0;
        /** Slew rate the step implies, in Hz/m/s, and the limit it is compared with. */
        double slew = 0.0;
        double limit = 0.0;
    };

    /** Peak gradient amplitudes reached by a sequence. */
    struct GradientReport
    {
        /**
         * Peak physical-axis amplitude and peak simultaneous vector magnitude.
         */
        Peak per_axis;
        Peak vector;
        /** Peak amplitude on each axis in turn, x, y, z.
         *
         * `per_axis` is the largest of these; the individual entries identify
         * which gradient amplifier reaches which amplitude. */
        std::array<Peak, 3> axes;
    };

    /**
     * Peak slew rates within blocks.
     *
     * Transitions *between* blocks are a separate check: a gradient starting
     * at an amplitude the previous block did not end at implies an unbounded
     * slew rate. `continuity` covers those.
     */
    struct SlewReport
    {
        Peak per_axis;
        Peak vector;
        /** Peak slew rate on each axis in turn, x, y, z. */
        std::array<Peak, 3> axes;
    };

    /** Gradient amplitude discontinuities between consecutive blocks. */
    struct ContinuityReport
    {
        std::vector<Discontinuity> discontinuities;
        /** Whether every gradient waveform ends at zero amplitude. */
        bool ends_at_zero = true;
    };

    /**
     * Peak gradient amplitude played by the sequence.
     *
     * @param seq  The sequence to check.
     */
    GradientReport max_gradient(const Sequence& seq);

    /**
     * Measure within-block slew on physical axes after block rotation.
     *
     * The vector peak uses simultaneous axis values, not the norm of separate
     * axis maxima. Rotation preserves vector magnitude but changes axis peaks.
     */
    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits);

    /**
     * Compare physical-axis endpoint amplitudes across blocks and at the end
     * of the sequence.
     *
     * An absent gradient contributes zero amplitude. A discontinuity is
     * evaluated as a change over one gradient raster period. Each endpoint
     * uses its own block's rotation.
     */
    ContinuityReport continuity(const Sequence& seq, const GradientLimits& limits);

    /**
     * The most negative and most positive value each physical axis plays in
     * each block, in Hz/m, after the block's rotation.
     *
     * Row-major, blocks x 3 axes x {low, high}. An axis counts zero among its
     * values, so a block that plays nothing on it reads {0, 0}.
     */
    std::vector<double> block_extremes(const Sequence& seq);

} // namespace pulseq

#endif /* PULSEQ_SAFETY_HPP */
