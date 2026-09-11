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

    /** What the scanner will allow. */
    struct GradientLimits
    {
        /**
         * Per-axis limits in Hz/m and Hz/m/s. Non-positive values disable judgment.
         */
        double max_grad = 0.0;
        double max_slew = 0.0;
        /** How often a gradient may change value. */
        double grad_raster_time = 10e-6;
    };

    /** The strongest thing found, and the block that plays it. */
    struct Peak
    {
        double value = 0.0;
        int block = 0;
        int axis = -1; /**< 0, 1, 2; -1 when the peak is a vector magnitude. */
    };

    /** One place where a gradient does not continue what came before it. */
    struct Discontinuity
    {
        int block = 0;
        int axis = 0;
        /** What the gradient jumps from and to, in Hz/m. */
        double before = 0.0;
        double after = 0.0;
        /** The step that jump asks for, in Hz/m/s, against what is allowed. */
        double slew = 0.0;
        double limit = 0.0;
    };

    /** What the gradients reach. */
    struct GradientReport
    {
        /**
         * Peak physical-axis amplitude and peak simultaneous vector magnitude.
         */
        Peak per_axis;
        Peak vector;
        /** The strongest each axis reaches, x, y, z.
         *
         * `per_axis` is the worst of these; these say which amplifier is
         * asked for what, which is the question when one of them is over. */
        std::array<Peak, 3> axes;
    };

    /**
     * What the gradients ask in the way of slewing, within the blocks.
     *
     * What happens *between* two blocks is a different question with a
     * different answer -- a gradient starting where the last one did not end
     * asks for its whole step in no time at all -- and `continuity` is what
     * asks it.
     */
    struct SlewReport
    {
        Peak per_axis;
        Peak vector;
        /** The steepest each axis is asked to change at, x, y, z. */
        std::array<Peak, 3> axes;
    };

    /** Where the gradients do not carry on from one block to the next. */
    struct ContinuityReport
    {
        std::vector<Discontinuity> discontinuities;
        /** Whether the sequence leaves its gradients at zero. */
        bool ends_at_zero = true;
    };

    /**
     * The strongest gradient the sequence plays.
     *
     * @param seq  The sequence to weigh.
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
     * Compare physical-axis endpoints across blocks and at sequence end.
     *
     * Missing gradients are zero. Boundary jumps are judged as a change over
     * one gradient raster interval. Each endpoint uses its own block's rotation.
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
