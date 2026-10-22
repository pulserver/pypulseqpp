/**
 * @file raster.hpp
 * @brief The played gradient on the physical axes, sampled on the sequence's own raster.
 */

#ifndef PULSEQ_RASTER_HPP
#define PULSEQ_RASTER_HPP

#include <cmath>
#include <cstdint>
#include <vector>

#include "pulseq/corners.hpp"
#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * @p seconds as whole picoseconds, halves to even: the resolution the
     * binary format writes times at, and the one block start times are summed
     * at.
     */
    inline int64_t to_picoseconds(double seconds)
    {
        return static_cast<int64_t>(std::nearbyint(seconds * 1e12));
    }

    /** What makes two blocks play one physical gradient over their samples. */
    struct BlockKey
    {
        int32_t gradient[3];
        int32_t rotation;
        int64_t samples;
        /**
         * Time from the block's start to its first sample, in half
         * picoseconds. It is exact, so the same block at the same place on
         * the raster has the same offset however far into the sequence it is.
         */
        int64_t offset;

        bool operator==(const BlockKey& other) const
        {
            return gradient[0] == other.gradient[0] && gradient[1] == other.gradient[1] &&
                gradient[2] == other.gradient[2] && rotation == other.rotation &&
                samples == other.samples && offset == other.offset;
        }
    };

    struct BlockKeyHash
    {
        size_t operator()(const BlockKey& key) const
        {
            uint64_t h = 1469598103934665603ull;
            const auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            for (int32_t g : key.gradient)
                mix(static_cast<uint32_t>(g));
            mix(static_cast<uint32_t>(key.rotation));
            mix(static_cast<uint64_t>(key.samples));
            mix(static_cast<uint64_t>(key.offset));
            return static_cast<size_t>(h);
        }
    };

    /**
     * A forward reader of the physical-axis gradient, in Hz/m.
     *
     * Sample n is the gradient at the centre (n + 1/2) dt of the n-th interval
     * of the sequence's gradient raster, the one its definitions record, from
     * the start of the first block to the end of the last; or, when read at
     * interval ends, at its end (n + 1) dt. Either way the sample belongs to
     * the block that holds the interval's centre. Each block's own
     * rotation is applied and then the prescription rotation. Samples are
     * produced on demand, so a reader never holds the timeline.
     *
     * The raster and each block's duration are taken in whole picoseconds
     * and block start times summed in them, so a sample's time from its
     * block's start is exact and does not depend on how far into the
     * sequence the block is.
     */
    class PhysicalRaster
    {
    public:
        /**
         * @param rotation  Prescription rotation, logical to physical.
         * @param at_ends   Read each interval's end instead of its centre.
         */
        PhysicalRaster(const Sequence& seq, const double rotation[3][3], bool at_ends = false);

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

        /**
         * Move to the next block that plays any sample, from a block boundary.
         *
         * @return Its samples; 0 once the sequence is exhausted.
         */
        int64_t enter_block();

        /** Time of sample position() + @p from, in s from the start of the current block. */
        double time_in_block(int64_t from) const
        {
            return static_cast<double>(offset_of(position_ + from)) * 0.5e-12;
        }

        /**
         * Samples from the next one to the next corner of any axis, within one block.
         *
         * @param steady  Set when every axis holds one value over all of them.
         * @return At least 1; 0 once the sequence is exhausted.
         */
        int64_t run(bool* steady);

        /** Key of the current block's next @p count samples, after enter_block(). */
        BlockKey block_key(int64_t count) const;

        /** Corners of the current block's gradient on logical channel @p axis; null where none plays. */
        const Corners* channel(int axis) const
        {
            return channels_[axis];
        }

        /** Logical to physical rotation of the current block, prescription included. */
        const double (&rotation() const)[3][3]
        {
            return turn_;
        }

        /** Corners of logical channel @p axis as the current block plays them,
         * in s from the block's start. */
        struct ChannelCorners
        {
            const double* times = nullptr;
            const double* values = nullptr;
            size_t count = 0;
            double offset = 0.0;
        };
        ChannelCorners played(int axis) const
        {
            const Played& p = played_[axis];
            return {p.times, p.values, p.count, p.offset};
        }

        /** Samples @p from to @p from + @p count of logical channel @p axis,
         * counted from the next sample, without moving. */
        void sample_channel(int axis, int64_t from, int64_t count, double* out) const;

        /**
         * Move to the start of the block holding sample @p sample, without
         * reading the blocks passed over, so that enter_block() enters it.
         * A block already entered is entered again from its start.
         */
        void advance(int64_t sample);

        /** Pass over @p count samples of the current block. */
        void skip(int64_t count)
        {
            position_ += count;
        }

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

        /** Time from the current block's start to sample @p n, in half picoseconds. */
        int64_t offset_of(int64_t n) const
        {
            return (2 * n + phase_) * dt_ps_ - 2 * starts_ps_;
        }

        /** First sample whose centre is not before @p time, in ps from the first block's start. */
        int64_t first_sample_from(int64_t time) const;

        const Sequence& seq_;
        double dt_;
        int64_t dt_ps_;
        /** Where a sample sits in its interval, in half intervals. */
        int64_t phase_;
        double prescription_[3][3];
        CornerCache corners_;
        std::vector<double> trapezoid_times_[3];

        int next_block_ = 0;
        int block_ = 0;
        /* Start and end of the current block, in ps from the first block's start. */
        int64_t starts_ps_ = 0;
        int64_t ends_ps_ = 0;
        int64_t position_ = 0;
        int64_t block_first_ = 0;
        int64_t block_end_ = 0;
        Played played_[3];
        const Corners* channels_[3] = {nullptr, nullptr, nullptr};
        bool any_ = false;
        double turn_[3][3];
    };

} // namespace pulseq

#endif /* PULSEQ_RASTER_HPP */
