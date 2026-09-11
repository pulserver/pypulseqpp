/**
 * @file raster.hpp
 * @brief The played gradient on the physical axes, sampled on the sequence's own raster.
 */

#ifndef PULSEQ_RASTER_HPP
#define PULSEQ_RASTER_HPP

#include <cstdint>
#include <vector>

#include "pulseq/corners.hpp"
#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * A forward reader of the physical-axis gradient, in Hz/m.
     *
     * Sample n is the gradient at the centre (n + 1/2) dt of the n-th interval
     * of the sequence's gradient raster, the one its definitions record, from
     * the start of the first block to the end of the last. Each block's own
     * rotation is applied and then the prescription rotation. Samples are
     * produced on demand, so a reader never holds the timeline.
     */
    class PhysicalRaster
    {
    public:
        /** @param rotation  Prescription rotation, logical to physical. */
        PhysicalRaster(const Sequence& seq, const double rotation[3][3]);

        double raster() const
        {
            return dt_;
        }

        /** Index of the next sample to be read. */
        int64_t position() const
        {
            return position_;
        }

        /** 1-based block the last read came from; 0 before any. */
        int block() const
        {
            return block_;
        }

        /**
         * Read up to @p count samples of each axis, all from one block.
         *
         * @return How many were read; 0 once the sequence is exhausted.
         */
        int64_t read(int64_t count, double* x, double* y, double* z);

    private:
        struct Played
        {
            const double* times = nullptr;
            const double* values = nullptr;
            size_t count = 0;
            double offset = 0.0;
            size_t at = 0;

            double when(size_t i) const
            {
                return times[i] + offset;
            }
            double value(double t);
        };

        bool enter_next_block();

        const Sequence& seq_;
        double dt_;
        double prescription_[3][3];
        CornerCache corners_;
        std::vector<double> trapezoid_times_[3];

        int next_block_ = 0;
        int block_ = 0;
        double starts_ = 0.0;
        double ends_ = 0.0;
        int64_t position_ = 0;
        int64_t block_end_ = 0;
        Played played_[3];
        bool any_ = false;
        double turn_[3][3];
    };

} // namespace pulseq

#endif /* PULSEQ_RASTER_HPP */
