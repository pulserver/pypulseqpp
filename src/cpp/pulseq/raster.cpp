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

        /* Fraction of a raster interval a time may miss a sample centre by. */
        constexpr double kRasterEps = 1e-9;

        /** First sample whose centre (n + 1/2) dt is not before @p time. */
        int64_t first_sample_from(double time, double dt)
        {
            return static_cast<int64_t>(std::ceil(time / dt - 0.5 - kRasterEps));
        }

    } // namespace

    double PhysicalRaster::Played::value(double t)
    {
        if (count == 0 || t < when(0) || t > when(count - 1))
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

    PhysicalRaster::PhysicalRaster(const Sequence& seq, const double rotation[3][3])
        : seq_(seq), dt_(seq.grad_raster_time()), corners_(seq, seq.grad_raster_time())
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                prescription_[i][j] = rotation[i][j];
    }

    bool PhysicalRaster::enter_next_block()
    {
        if (next_block_ >= seq_.num_blocks())
            return false;
        const int index = next_block_++;
        const int32_t* row = seq_.block_events() + static_cast<size_t>(index) * BLOCK_WIDTH;
        block_ = index + 1;
        starts_ = ends_;
        ends_ = starts_ + seq_.block_durations()[index];
        block_end_ = std::max(position_, first_sample_from(ends_, dt_));

        any_ = false;
        for (int axis = 0; axis < 3; ++axis)
        {
            Played& p = played_[axis];
            p = Played();
            const int32_t id = row[1 + axis];
            if (id <= 0)
                continue;
            const Corners& shape = corners_[id];
            /* No ramps and no flat top plays for no time. */
            if (shape.empty_with_amplitude || shape.values.empty())
                continue;
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
                const double t = (static_cast<double>(position_ + i) + 0.5) * dt_ - starts_;
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
