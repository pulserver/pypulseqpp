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
 * What is weighed is the waveform an interpreter draws, corner to corner --
 * not the samples the file stores, which are not the same thing: a shape kept
 * at the centre of each raster interval turns its corners half a raster from
 * any sample it holds, and passes outside all of them.
 *
 * Drawing it is nevertheless cheap, because the corners belong to the
 * gradient rather than to the block. An event plays the same shape every time
 * it is played and only where it starts moves, so the corners are worked out
 * once for it and read per block: a readout repeated a hundred thousand times
 * costs one pass over `restore_shape_corners` and a walk per block. The same
 * is true of the strongest it gets and the fastest it changes, which is what
 * bounds the block without weighing it.
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
     * What the sequence asks in the way of slewing, within its blocks.
     *
     * The vector magnitude is exact rather than bounded. Each axis slews at
     * one rate at a time -- a trapezoid at one rate up its ramp and another
     * down it, a waveform at one rate per sample -- so the three together are
     * constant between the moments any of them changes, and the largest they
     * reach is the largest on one of those stretches. Taking each axis's own
     * peak and combining those would answer a different question: what the
     * amplifiers would be asked for if the three peaks happened at once,
     * which they need not.
     *
     * The magnitude does not depend on how the block is rotated. Turning a
     * vector does not change how long it is, so what the amplifiers are asked
     * for between them is the same whichever way the block faces; only which
     * one is asked for what changes, and that is what `axes` reports.
     *
     * @param seq     The sequence to weigh.
     * @param limits  The raster and the slew limit.
     */
    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits);

    /**
     * Where a gradient does not carry on from the block before it.
     *
     * An axis is at zero wherever nothing is playing on it, so a waveform
     * that starts away from where the last block left the axis asks the
     * amplifier for that whole step within one raster interval -- and a
     * sequence that ends with an axis still on has never ramped it down.
     *
     * The endpoints are compared in the frame the amplifiers work in, so a
     * block that turns its gradients has its own endpoints turned first: two
     * blocks playing the same waveform at different rotations do not continue
     * one another, and saying they do would miss the jump.
     *
     * @param seq     The sequence to check.
     * @param limits  The raster and the slew limit a jump is judged against.
     */
    ContinuityReport continuity(const Sequence& seq, const GradientLimits& limits);

} // namespace pulseq

#endif /* PULSEQ_SAFETY_HPP */
