/**
 * @file raster.cpp
 * @brief Physical-axis gradient on the sequence raster.  See raster.hpp.
 */

#include "pulseq/raster.hpp"

#include <algorithm>
#include <cmath>

namespace pulseq
{

    namespace
    {

        /* Fraction of a raster interval a time may miss a sample by. */
        constexpr double kRasterEps = 1e-9;

        /* Time, in s, a sample on a shape's last corner may be read past it by
         * rounding, below the picosecond the format writes times in. */
        constexpr double kCornerEps = 1e-13;

    } // namespace

    double PhysicalRaster::Played::value(double t)
    {
        if (count == 0 || t < when(0) || t > when(count - 1) + kCornerEps)
            return 0.0;
        while (at + 1 < count && when(at + 1) <= t)
            ++at;
        if (at + 1 >= count)
            return values[count - 1];
        const double span = when(at + 1) - when(at);
        if (span <= 0.0)
            return values[at + 1];
        return values[at] + (values[at + 1] - values[at]) * (t - when(at)) / span;
    }

    PhysicalRaster::PhysicalRaster(const Sequence& seq, const double rotation[3][3], bool at_ends)
        : seq_(seq), dt_(seq.grad_raster_time()), dt_ps_(to_picoseconds(seq.grad_raster_time())),
          phase_(at_ends ? 2 : 1), corners_(seq, seq.grad_raster_time())
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                prescription_[i][j] = rotation[i][j];
    }

    int64_t PhysicalRaster::first_sample_from(int64_t time) const
    {
        /* The least n with (2 n + 1) dt >= 2 time: a division rounded up,
         * which C++ rounds toward zero. */
        const int64_t numerator = 2 * time - dt_ps_;
        const int64_t denominator = 2 * dt_ps_;
        const int64_t quotient = numerator / denominator;
        return numerator % denominator > 0 ? quotient + 1 : quotient;
    }

    bool PhysicalRaster::enter_next_block()
    {
        if (next_block_ >= seq_.num_blocks())
            return false;
        const int index = next_block_++;
        const int32_t* row = seq_.block_events() + static_cast<size_t>(index) * BLOCK_WIDTH;
        block_ = index + 1;
        block_first_ = position_;
        starts_ps_ = ends_ps_;
        ends_ps_ = starts_ps_ + to_picoseconds(seq_.block_durations()[index]);
        block_end_ = std::max(position_, first_sample_from(ends_ps_));

        any_ = false;
        for (int axis = 0; axis < 3; ++axis)
        {
            Played& p = played_[axis];
            p = Played();
            channels_[axis] = nullptr;
            const int32_t id = row[1 + axis];
            if (id <= 0)
                continue;
            const Corners& shape = corners_[id];
            /* No ramps and no flat top plays for no time. */
            if (shape.empty_with_amplitude || shape.values.empty())
                continue;
            channels_[axis] = &shape;
            p.values = shape.values.data();
            p.count = shape.values.size();
            if (shape.trapezoid)
            {
                shape.at(0.0, trapezoid_times_[axis]);
                p.times = trapezoid_times_[axis].data();
            }
            else
            {
                p.times = shape.times.data();
                p.offset = shape.delay;
            }
            any_ = true;
        }

        double own[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        const int32_t rotation = row[BLOCK_ROTATION_COLUMN];
        if (rotation >= 1 && rotation <= seq_.rotation_library().size())
            rotation_matrix(seq_.rotation_library().row(rotation), own);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                turn_[i][j] = prescription_[i][0] * own[0][j] +
                    prescription_[i][1] * own[1][j] + prescription_[i][2] * own[2][j];
        return true;
    }

    void PhysicalRaster::advance(int64_t sample)
    {
        if (block_ > 0 && position_ < block_end_)
        {
            next_block_ = block_ - 1;
            ends_ps_ = starts_ps_;
            position_ = block_first_;
            block_end_ = position_;
        }
        const double* durations = seq_.block_durations();
        while (next_block_ < seq_.num_blocks())
        {
            const int64_t ends = ends_ps_ + to_picoseconds(durations[next_block_]);
            const int64_t end = std::max(position_, first_sample_from(ends));
            if (end > sample)
                break;
            ends_ps_ = ends;
            position_ = end;
            block_end_ = end;
            block_ = ++next_block_;
        }
    }

    int64_t PhysicalRaster::enter_block()
    {
        while (position_ >= block_end_)
        {
            if (!enter_next_block())
                return 0;
        }
        return block_end_ - position_;
    }

    BlockKey PhysicalRaster::block_key(int64_t count) const
    {
        const int32_t* row = seq_.block_events() + static_cast<size_t>(block_ - 1) * BLOCK_WIDTH;
        return {{row[1], row[2], row[3]}, row[BLOCK_ROTATION_COLUMN], count, offset_of(position_)};
    }

    int64_t PhysicalRaster::run(bool* steady)
    {
        while (position_ >= block_end_)
        {
            if (!enter_next_block())
                return 0;
        }
        const double t = time_in_block(0);
        double next = static_cast<double>(ends_ps_ - starts_ps_) * 1e-12;
        *steady = true;
        for (Played& p : played_)
        {
            if (p.count == 0)
                continue;
            if (t < p.when(0))
            {
                next = std::min(next, p.when(0));
                continue;
            }
            if (t > p.when(p.count - 1) + kCornerEps)
                continue;
            while (p.at + 1 < p.count && p.when(p.at + 1) <= t)
                ++p.at;
            if (p.at + 1 >= p.count)
            {
                /* On the last corner: zero from the next sample on. */
                next = t;
                *steady = false;
                continue;
            }
            next = std::min(next, p.when(p.at + 1));
            if (p.values[p.at] != p.values[p.at + 1])
                *steady = false;
        }
        /* Samples from this one to the first not before next. */
        const double ahead = std::ceil((next - t) / dt_ - kRasterEps);
        const int64_t until = std::min(block_end_ - position_, static_cast<int64_t>(ahead));
        return std::max<int64_t>(1, until);
    }

    void PhysicalRaster::sample_channel(int axis, int64_t from, int64_t count, double* out) const
    {
        Played p = played_[axis];
        for (int64_t i = 0; i < count; ++i)
            out[i] = p.value(time_in_block(from + i));
    }

    int64_t PhysicalRaster::read(int64_t count, double* x, double* y, double* z)
    {
        while (position_ >= block_end_)
        {
            if (!enter_next_block())
                return 0;
        }
        const int64_t got = std::min(count, block_end_ - position_);
        double* out[3] = {x, y, z};
        if (!any_)
        {
            for (int axis = 0; axis < 3; ++axis)
                std::fill(out[axis], out[axis] + got, 0.0);
        }
        else
        {
            for (int64_t i = 0; i < got; ++i)
            {
                const double t = time_in_block(i);
                double logical[3];
                for (int axis = 0; axis < 3; ++axis)
                    logical[axis] = played_[axis].value(t);
                for (int axis = 0; axis < 3; ++axis)
                {
                    out[axis][i] = turn_[axis][0] * logical[0] + turn_[axis][1] * logical[1] +
                        turn_[axis][2] * logical[2];
                }
            }
        }
        position_ += got;
        return got;
    }

} // namespace pulseq
