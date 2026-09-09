/**
 * @file corners.hpp
 * @brief A gradient as the points where its slope changes.
 *
 * A gradient is played by interpolating linearly between the samples it is
 * given, so the waveform is fully described by the points where its slope
 * changes and nothing is lost by leaving out the rest. A trapezoid is four
 * points however long its flat top is.
 *
 * A shape stored at the centre of each raster interval is a different matter:
 * the samples are not the corners, and the waveform the interpreter draws
 * passes through points half a raster from any of them. Anything asking what
 * a sequence *does* -- what it draws, how strong it gets, how fast it changes
 * -- has to ask about those points rather than about the samples, which is
 * what `restore_shape_corners` is for and why this is one place rather than
 * one per caller.
 *
 * The corners belong to the gradient rather than to the block: the same event
 * played a hundred thousand times draws the same shape, and only where it
 * starts moves. So they are worked out once per gradient and offset per
 * block, which is what `CornerCache` is.
 */

#ifndef PULSEQ_CORNERS_HPP
#define PULSEQ_CORNERS_HPP

#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** One gradient's corners, relative to the start of its own delay. */
    struct Corners
    {
        std::vector<double> times;
        std::vector<double> values;
        double delay = 0.0;
        bool known = false;
        /** A trapezoid with no ramps and no flat top, but an amplitude. */
        bool empty_with_amplitude = false;

        /**
         * A trapezoid's ramps, kept so its corners can be added up from where
         * the block starts rather than offset from zero.
         *
         * `(start + rise) + flat` and `(rise + flat) + start` are not the same
         * double, and a corner one bit out lands on the other side of
         * `floor(t / raster)` -- which changes which raster points a
         * trajectory is followed through. Four additions is nothing; the shape
         * of a long waveform is what the cache is really for.
         */
        bool trapezoid = false;
        double ramps[3] = {0.0, 0.0, 0.0};
        double amplitude = 0.0;

        /** Where this gradient's corners fall, given where its block starts. */
        void at(double block_starts, std::vector<double>& into) const;
    };

    /** Every gradient's corners, worked out the first time it is asked for. */
    class CornerCache
    {
    public:
        /** Corners as @p seq stores them, on its own gradient raster. */
        explicit CornerCache(const Sequence& seq);
        /** Corners as they would be drawn on a raster of @p grad_raster. */
        CornerCache(const Sequence& seq, double grad_raster);

        /** The corners of gradient @p id.  Id 0 is empty. */
        const Corners& operator[](int32_t id);

    private:
        const Sequence& seq_;
        double raster_;
        ShapeCache shapes_;
        std::vector<Corners> held_;
    };

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

#endif /* PULSEQ_CORNERS_HPP */
