/**
 * @file safety.hpp
 * @brief What the gradients ask of the amplifiers.
 *
 * Two limits bound every gradient a scanner will play: how strong it may be,
 * and how fast it may change. Neither is a property of one waveform -- three
 * axes play at once, and what an amplifier sees on its own axis depends on
 * how the sequence is rotated -- so both are asked of the block, not of the
 * event.
 *
 * ### What the fork buys here
 *
 * A gradient is stored as a normalised shape and one amplitude, and the shape
 * is shared by every event that plays it. So the steepest step in a waveform
 * is a property of the *shape*, worked out once however many times it is
 * played, and what an instance slews at is that step times its own amplitude
 * over the interval between samples. A readout repeated a hundred thousand
 * times at a hundred thousand amplitudes costs one pass over its shape and a
 * multiply per block.
 *
 * The same is true of where a waveform starts and ends, which is what decides
 * whether one block's gradient continues the last one's or jumps.
 */

#ifndef PULSEQ_SAFETY_HPP
#define PULSEQ_SAFETY_HPP

#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** What the scanner will allow. */
    struct GradientLimits
    {
        /** Per axis, in Hz/m and Hz/m/s. Zero means do not judge it. */
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
        /** The strongest single axis, and the strongest vector magnitude.
         *
         * A sequence played as written only ever asks one amplifier for the
         * first; one played rotated can ask for as much as the second, which
         * is why both are worth knowing. */
        Peak per_axis;
        Peak vector;
    };

    /** What the gradients ask in the way of slewing. */
    struct SlewReport
    {
        Peak per_axis;
        Peak vector;
        /** Every place a gradient jumps rather than ramps. */
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
     * What the sequence asks in the way of slewing, and where it jumps.
     *
     * @param seq     The sequence to weigh.
     * @param limits  The rasters and the slew limit a jump is judged against.
     */
    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits);

} // namespace pulseq

#endif /* PULSEQ_SAFETY_HPP */
