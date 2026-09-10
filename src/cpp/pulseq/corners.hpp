/**
 * @file corners.hpp
 * @brief Piecewise-linear gradient corners and lazy per-event corner caching.
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
         * Keep trapezoid timing components to preserve floating-point addition order.
         *
         * (start + rise) + flat can differ from (rise + flat) + start enough to
         * change floor(t / raster), and hence the trajectory evaluation grid.
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
     * Restore gradient corners from raster-centred samples and recorded endpoints.
     *
     * If the boundary recurrence reaches @p last, use its boundaries, replacing
     * them with adjacent-sample averages where they agree within tolerance,
     * and remove collinear points. Otherwise retain
     * the original sample centres with the recorded endpoints added.
     *
     * @param waveform  Gradient samples in Hz/m.
     * @param first     Value at the start of the first interval, in Hz/m.
     * @param last      Value at the end of the last interval, in Hz/m.
     * @param raster    Gradient raster in seconds.
     * @param times     Output times in seconds relative to waveform start.
     * @param values    Output gradient values in Hz/m.
     * @return Whether the boundary recurrence reached the recorded last value.
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
