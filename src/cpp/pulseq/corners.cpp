/**
 * @file corners.cpp
 * @brief A gradient as the points where its slope changes.  See corners.hpp.
 */

#include "pulseq/corners.hpp"

#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pulseq
{

    namespace
    {
        /** A nanosecond: what two times have to differ by to be two times. */
        constexpr double kEps = 1e-9;
    }

    void Corners::at(double block_starts, std::vector<double>& into) const
    {
        into.clear();
        if (trapezoid)
        {
            /* Added up from where the block starts rather than offset from
             * zero: see `ramps`. */
            double when = block_starts + delay;
            into.push_back(when);
            for (size_t i = 1; i < values.size(); ++i)
            {
                when += ramps[i - 1];
                into.push_back(when);
            }
            return;
        }
        const double offset = block_starts + delay;
        into.reserve(times.size());
        for (size_t i = 0; i < times.size(); ++i)
            into.push_back(times[i] + offset);
    }

    CornerCache::CornerCache(const Sequence& seq)
        : CornerCache(seq, seq.grad_raster_time())
    {
    }

    CornerCache::CornerCache(const Sequence& seq, double grad_raster)
        : seq_(seq),
          raster_(grad_raster),
          shapes_(seq.shape_library()),
          held_(static_cast<size_t>(seq.num_gradients()) + 1)
    {
    }

    const Corners& CornerCache::operator[](int32_t id)
    {
        Corners& made = held_[static_cast<size_t>(id)];
        if (made.known)
            return made;
        made.known = true;
        if (id <= 0 || id > seq_.num_gradients())
            return made;

            if (seq_.grad_kind(id) == GradKind::Trap)
            {
                const double* trap = seq_.trap_library().row(seq_.grad_row(id));
                const double amplitude = trap[0];
                const double rise = trap[1];
                const double flat = trap[2];
                const double fall = trap[3];
                made.delay = trap[4];

                made.trapezoid = true;
                made.amplitude = amplitude;
                if (std::fabs(flat) > kEps)
                {
                    made.ramps[0] = rise;
                    made.ramps[1] = flat;
                    made.ramps[2] = fall;
                    made.values = {0.0, amplitude, amplitude, 0.0};
                }
                else if (std::fabs(rise) > kEps && std::fabs(fall) > kEps)
                {
                    made.ramps[0] = rise;
                    made.ramps[1] = fall;
                    made.values = {0.0, amplitude, 0.0};
                }
                else if (std::fabs(amplitude) > kEps)
                {
                    made.empty_with_amplitude = true;
                }
                return made;
            }

            const double* arb = seq_.arb_library().row(seq_.grad_row(id));
            const double amplitude = arb[0];
            const int shape = static_cast<int>(arb[3]);
            const int time_shape = static_cast<int>(arb[4]);
            made.delay = arb[5];

            const std::vector<double>& normalised = shapes_[shape];
            std::vector<double> waveform(normalised.size());
            for (size_t i = 0; i < normalised.size(); ++i)
                waveform[i] = amplitude * normalised[i];

            if (time_shape == 0)
            {
                /* Stored at the centre of each raster interval: the corners
                 * in between have to be put back. */
                restore_shape_corners(
                    waveform, arb[1], arb[2], raster_, made.times, made.values);
                return made;
            }

            const std::vector<double>& ticks = shapes_[time_shape];
            std::vector<double> tt(ticks.size());
            for (size_t i = 0; i < ticks.size(); ++i)
                tt[i] = ticks[i] * raster_;

            /* Times of its own, but starting half a raster in: the shape says
             * nothing about the edges, so the recorded first and last close it. */
            const bool starts_at_a_centre =
                !tt.empty() && std::fabs(tt[0] / raster_ - 0.5) < 1e-6;
            if (starts_at_a_centre)
            {
                made.times.push_back(0.0);
                made.values.push_back(arb[1]);
            }
            for (size_t i = 0; i < tt.size(); ++i)
            {
                made.times.push_back(tt[i]);
                made.values.push_back(waveform[i]);
            }
            if (starts_at_a_centre)
            {
                made.times.push_back(tt.empty() ? 0.0 : tt.back());
                made.values.push_back(arb[2]);
            }
        return made;
    }

    bool restore_shape_corners(
        const std::vector<double>& waveform,
        double first,
        double last,
        double raster,
        std::vector<double>& times,
        std::vector<double>& values)
    {
        const size_t count = waveform.size();
        times.clear();
        values.clear();
        if (count == 0)
            return true;

        double peak = 0.0;
        for (size_t i = 0; i < count; ++i)
            peak = std::max(peak, std::fabs(waveform[i]));
        const double threshold = 2e-5 * peak;

        /* Each interval boundary from the one before it: b[i+1] = 2 s[i] - b[i],
         * starting at the recorded first. Exact for a waveform that really was
         * sampled at the centres, and drifting otherwise -- which is what the
         * recorded last value is here to catch. */
        std::vector<double> recurrence(count + 1);
        recurrence[0] = first;
        for (size_t i = 0; i < count; ++i)
            recurrence[i + 1] = 2.0 * waveform[i] - recurrence[i];

        std::vector<double> interpolated(count + 1);
        interpolated[0] = first;
        for (size_t i = 0; i + 1 < count; ++i)
            interpolated[i + 1] = 0.5 * (waveform[i] + waveform[i + 1]);
        interpolated[count] = last;

        if (std::fabs(recurrence[count] - last) > threshold)
        {
            /* The recurrence did not land where the shape says it should, so
             * the shape was not sampled the way this assumes. Give back the
             * samples with the edges on and let the caller draw through them. */
            times.push_back(0.0);
            values.push_back(first);
            for (size_t i = 0; i < count; ++i)
            {
                times.push_back((static_cast<double>(i) + 0.5) * raster);
                values.push_back(waveform[i]);
            }
            times.push_back((static_cast<double>(count) - 0.5) * raster + raster / 2.0);
            values.push_back(last);
            return false;
        }

        std::vector<double> boundaries(count + 1);
        const double tolerance = std::numeric_limits<double>::epsilon() + threshold;
        for (size_t i = 0; i <= count; ++i)
        {
            boundaries[i] = std::fabs(recurrence[i] - interpolated[i]) <= tolerance
                ? interpolated[i]
                : recurrence[i];
        }

        /* Boundary, sample, boundary, sample, ... at half-raster spacing. */
        std::vector<double> dense;
        dense.reserve(2 * count + 1);
        for (size_t i = 0; i < count; ++i)
        {
            dense.push_back(boundaries[i]);
            dense.push_back(waveform[i]);
        }
        dense.push_back(boundaries[count]);

        /* Keep the corners: a point its neighbours draw straight through adds
         * nothing to a waveform played by interpolating. */
        const size_t dense_count = dense.size();
        for (size_t i = 0; i < dense_count; ++i)
        {
            const bool corner = i == 0 || i + 1 >= dense_count ||
                std::fabs(dense[i + 1] - 2.0 * dense[i] + dense[i - 1]) > 1e-8;
            if (corner)
            {
                times.push_back(static_cast<double>(i) * raster * 0.5);
                values.push_back(dense[i]);
            }
        }
        return true;
    }

} // namespace pulseq
