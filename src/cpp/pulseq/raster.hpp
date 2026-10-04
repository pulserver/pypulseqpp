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
     * Resolution, in rasters, at which two blocks' sample offsets are told
     * apart. Blocks closer than this are answered as one: their samples sit
     * within it of each other on the same waveform.
     */
    constexpr double kOffsetStep = 1e-4;

    /** What makes two blocks play one physical gradient over their samples. */
    struct BlockKey
    {
        int32_t gradient[3];
        int32_t rotation;
        int64_t samples;
        /**
         * Time from the block's start to its first sample centre, in units of
         * kOffsetStep rasters: coarse enough that the rounding of block start
         * times summed over a long sequence does not split a key.
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

        /**
         * Move to the next block that plays any sample, from a block boundary.
         *
         * @return Its samples; 0 once the sequence is exhausted.
         */
        int64_t enter_block();

        /** Start of the current block, in s from the start of the first. */
        double block_start() const
        {
            return starts_;
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
        const Corners* channels_[3] = {nullptr, nullptr, nullptr};
        bool any_ = false;
        double turn_[3][3];
    };

} // namespace pulseq

#endif /* PULSEQ_RASTER_HPP */
