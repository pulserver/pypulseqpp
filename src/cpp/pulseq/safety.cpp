/**
 * @file safety.cpp
 * @brief What the gradients ask of the amplifiers.  See safety.hpp.
 */

#include "pulseq/safety.hpp"

#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pulseq
{

    namespace
    {

        constexpr double kEps = 1e-12;

        /** One gradient's amplitude, however it is stored. */
        double amplitude_of(const Sequence& seq, int32_t id)
        {
            if (id <= 0 || id > seq.num_gradients())
                return 0.0;
            const int row = seq.grad_row(id);
            return seq.grad_kind(id) == GradKind::Trap ? seq.trap_library().row(row)[0]
                                                       : seq.arb_library().row(row)[0];
        }

        /**
         * How fast a normalised waveform slews at its steepest, per unit
         * amplitude, in 1/s.
         *
         * Worked out once per row of the arbitrary-gradient library and kept,
         * because it belongs to the waveform rather than to the event: a
         * readout played a hundred thousand times at a hundred thousand
         * amplitudes asks this once and multiplies.
         *
         * How steep a step is depends on how long it has to happen in, so
         * this is a property of the shape *and* the times it is played at --
         * the pair, not either one. On the raster the interval is the raster;
         * with a time shape of its own the samples can sit anywhere, and the
         * closest two are what set the steepest step.
         */
        class NormalisedSlew
        {
        public:
            NormalisedSlew(const Sequence& seq, double grad_raster)
                : seq_(seq),
                  raster_(grad_raster),
                  known_(static_cast<size_t>(seq.arb_library().size()) + 1, -1.0)
            {
            }

            double operator()(int row)
            {
                double& held = known_[static_cast<size_t>(row)];
                if (held >= 0.0)
                    return held;

                held = 0.0;
                const double* arb = seq_.arb_library().row(row);
                const int shape = static_cast<int>(arb[3]);
                const ShapeLibrary& shapes = seq_.shape_library();
                if (shape < 1 || shape > shapes.size())
                    return held;

                const int time_shape = static_cast<int>(arb[4]);
                if (time_shape <= 0)
                {
                    /* Evenly spaced: one interval for the whole waveform, and
                     * the steepest step is the shape's own. A time id of -1
                     * marks a waveform oversampled by two. */
                    const double interval =
                        time_shape == -1 ? raster_ / 2.0 : raster_;
                    if (interval > kEps)
                        held = shapes.normalised_slew(shape) / interval;
                    return held;
                }

                const std::vector<double> waveform = decompress_shape(
                    shapes.samples(shape),
                    shapes.num_compressed(shape),
                    shapes.num_uncompressed(shape));
                const std::vector<double> ticks = decompress_shape(
                    shapes.samples(time_shape),
                    shapes.num_compressed(time_shape),
                    shapes.num_uncompressed(time_shape));

                for (size_t i = 1; i < waveform.size() && i < ticks.size(); ++i)
                {
                    const double over = (ticks[i] - ticks[i - 1]) * raster_;
                    if (over > kEps)
                        held = std::max(
                            held, std::fabs(waveform[i] - waveform[i - 1]) / over);
                }
                return held;
            }

        private:
            const Sequence& seq_;
            double raster_;
            std::vector<double> known_;
        };

        /**
         * How fast one gradient slews at its steepest, in Hz/m/s.
         *
         * A trapezoid's ramps are the whole of it: its amplitude over the
         * shorter of the two. A waveform's is what its shape asks for at unit
         * amplitude, times the amplitude this event plays it at.
         */
        double slew_of(
            const Sequence& seq, int32_t id, NormalisedSlew& normalised)
        {
            if (id <= 0 || id > seq.num_gradients())
                return 0.0;
            const int row = seq.grad_row(id);

            if (seq.grad_kind(id) == GradKind::Trap)
            {
                const double* trap = seq.trap_library().row(row);
                const double amplitude = std::fabs(trap[0]);
                double steepest = 0.0;
                if (trap[1] > kEps)
                    steepest = std::max(steepest, amplitude / trap[1]);
                if (trap[3] > kEps)
                    steepest = std::max(steepest, amplitude / trap[3]);
                return steepest;
            }

            return std::fabs(seq.arb_library().row(row)[0]) * normalised(row);
        }

        /** Where a gradient starts and ends, in Hz/m. */
        void edges_of(const Sequence& seq, int32_t id, double* first, double* last)
        {
            *first = 0.0;
            *last = 0.0;
            if (id <= 0 || id > seq.num_gradients())
                return;
            // A trapezoid begins and ends at zero; that is what makes it one.
            if (seq.grad_kind(id) == GradKind::Trap)
                return;
            const double* arb = seq.arb_library().row(seq.grad_row(id));
            *first = arb[1];
            *last = arb[2];
        }

        void note(Peak& worst, double value, int block, int axis)
        {
            if (value > worst.value)
            {
                worst.value = value;
                worst.block = block;
                worst.axis = axis;
            }
        }

        /**
         * What the three axes slew at together, moment by moment, within one
         * block.
         *
         * Each axis slews at one rate at a time -- a trapezoid at one rate up
         * its ramp and another down it, a waveform at one rate per sample --
         * so the three together are constant between the moments any of them
         * changes. The largest they reach is therefore the largest on one of
         * those stretches, and walking the stretches is exact where combining
         * the three axes' separate peaks is only a bound.
         *
         * What is walked is the stored representation, not an expanded
         * waveform: the steps are read off the trapezoid's ramps and the
         * shape's samples where they are, which is what keeps this a pass
         * over the events a block names rather than over the scan.
         */
        class SlewProfile
        {
        public:
            SlewProfile(const Sequence& seq, double grad_raster)
                : seq_(seq), raster_(grad_raster), shapes_(seq.shape_library())
            {
            }

            /**
             * Weigh the block @p row names.
             *
             * @param row      Its block table row.
             * @param vector   Filled with the longest the slew vector gets.
             * @param axes     Filled with the most each amplifier is asked
             *                 for, once the block's own rotation is applied.
             */
            void weigh(const int32_t* row, double* vector, double axes[3])
            {
                *vector = 0.0;
                axes[0] = axes[1] = axes[2] = 0.0;

                edges_.clear();
                for (int axis = 0; axis < 3; ++axis)
                {
                    steps_of(row[1 + axis], times_[axis], slews_[axis]);
                    edges_.insert(
                        edges_.end(), times_[axis].begin(), times_[axis].end());
                }
                std::sort(edges_.begin(), edges_.end());
                edges_.erase(std::unique(edges_.begin(), edges_.end()), edges_.end());
                if (edges_.size() < 2)
                    return;

                double matrix[3][3];
                const int32_t turned = row[BLOCK_WIDTH - 1];
                const bool rotated =
                    turned >= 1 && turned <= seq_.rotation_library().size();
                if (rotated)
                    rotation_matrix(seq_.rotation_library().row(turned), matrix);

                size_t at[3] = {0, 0, 0};
                for (size_t i = 0; i + 1 < edges_.size(); ++i)
                {
                    double slew[3];
                    for (int axis = 0; axis < 3; ++axis)
                        slew[axis] = held(axis, edges_[i], at[static_cast<size_t>(axis)]);

                    /* How long the vector is does not depend on which way the
                     * block faces, so it is read off before the turn; which
                     * amplifier is asked for what does, so that is read off
                     * after it. */
                    *vector = std::max(
                        *vector,
                        std::sqrt(
                            slew[0] * slew[0] + slew[1] * slew[1] + slew[2] * slew[2]));
                    if (rotated)
                        rotate(matrix, slew);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        axes[axis] =
                            std::max(axes[axis], std::fabs(slew[static_cast<size_t>(axis)]));
                    }
                }
            }

        private:
            /** What axis @p axis slews at from @p when, walking forward. */
            double held(int axis, double when, size_t& at) const
            {
                const std::vector<double>& times = times_[static_cast<size_t>(axis)];
                const std::vector<double>& slews = slews_[static_cast<size_t>(axis)];
                if (slews.empty() || when + kEps < times.front() ||
                    when + kEps >= times.back())
                    return 0.0;
                while (at + 1 < slews.size() && times[at + 1] <= when + kEps)
                    ++at;
                return slews[at];
            }

            /**
             * One gradient as the times its slew changes and what it is
             * between them: @p times holds one more entry than @p slews, and
             * `slews[i]` is what the gradient slews at over
             * `[times[i], times[i + 1])`.
             */
            void steps_of(
                int32_t id, std::vector<double>& times, std::vector<double>& slews)
            {
                times.clear();
                slews.clear();
                if (id <= 0 || id > seq_.num_gradients())
                    return;
                const int row = seq_.grad_row(id);

                if (seq_.grad_kind(id) == GradKind::Trap)
                {
                    const double* trap = seq_.trap_library().row(row);
                    double when = trap[4];
                    /* A ramp that takes no time asks for an infinite slew,
                     * which is a fault of the event rather than a rate, and
                     * is left out here as it is everywhere else. */
                    const auto add = [&](double span, double slew) {
                        if (span <= kEps)
                            return;
                        if (times.empty())
                            times.push_back(when);
                        when += span;
                        times.push_back(when);
                        slews.push_back(slew);
                    };
                    add(trap[1], trap[0] / trap[1]);
                    add(trap[2], 0.0);
                    add(trap[3], -trap[0] / trap[3]);
                    return;
                }

                const double* arb = seq_.arb_library().row(row);
                const double amplitude = arb[0];
                const std::vector<double>& shape = shapes_[static_cast<int>(arb[3])];
                if (shape.size() < 2)
                    return;
                const int time_shape = static_cast<int>(arb[4]);
                const double delay = arb[5];

                if (time_shape <= 0)
                {
                    /* Evenly spaced, at the centre of each interval; a time
                     * id of -1 marks a waveform oversampled by two. */
                    const double interval = time_shape == -1 ? raster_ / 2.0 : raster_;
                    if (interval <= kEps)
                        return;
                    times.reserve(shape.size());
                    slews.reserve(shape.size() - 1);
                    for (size_t i = 0; i < shape.size(); ++i)
                        times.push_back(delay + (static_cast<double>(i) + 0.5) * interval);
                    for (size_t i = 0; i + 1 < shape.size(); ++i)
                        slews.push_back(amplitude * (shape[i + 1] - shape[i]) / interval);
                    return;
                }

                const std::vector<double>& ticks = shapes_[time_shape];
                const size_t count = std::min(shape.size(), ticks.size());
                if (count < 2)
                    return;
                times.reserve(count);
                slews.reserve(count - 1);
                for (size_t i = 0; i < count; ++i)
                    times.push_back(delay + ticks[i] * raster_);
                for (size_t i = 0; i + 1 < count; ++i)
                {
                    const double over = (ticks[i + 1] - ticks[i]) * raster_;
                    slews.push_back(
                        over > kEps ? amplitude * (shape[i + 1] - shape[i]) / over : 0.0);
                }
            }

            const Sequence& seq_;
            double raster_;
            ShapeCache shapes_;
            /* Kept across blocks so a scan allocates once. */
            std::vector<double> times_[3];
            std::vector<double> slews_[3];
            std::vector<double> edges_;
        };

    } // namespace

    GradientReport max_gradient(const Sequence& seq)
    {
        GradientReport out;

        const int32_t* events = seq.block_events();
        const int blocks = seq.num_blocks();

        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
            double squared = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                const double amplitude = std::fabs(amplitude_of(seq, row[1 + axis]));
                squared += amplitude * amplitude;
                note(out.per_axis, amplitude, index + 1, axis);
                note(out.axes[static_cast<size_t>(axis)], amplitude, index + 1, axis);
            }
            note(out.vector, std::sqrt(squared), index + 1, -1);
        }
        return out;
    }

    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits)
    {
        SlewReport out;

        const double raster = limits.grad_raster_time > 0.0 ? limits.grad_raster_time
                                                            : seq.grad_raster_time();
        NormalisedSlew normalised(seq, raster);
        SlewProfile profile(seq, raster);

        const int32_t* events = seq.block_events();
        const int blocks = seq.num_blocks();

        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
            const int32_t turned = row[BLOCK_WIDTH - 1];
            const bool rotated = turned >= 1 && turned <= seq.rotation_library().size();

            /* What each of the block's gradients asks for on its own, and the
             * bound their peaks put on what the three ask for together. */
            double bound = 0.0;
            for (int axis = 0; axis < 3; ++axis)
            {
                const double slew = slew_of(seq, row[1 + axis], normalised);
                bound += slew * slew;
                /* Played as written, an amplifier is asked for exactly what
                 * its own gradient asks for. Played turned, it is asked for a
                 * share of all three, and only the block knows what share. */
                if (!rotated)
                {
                    note(out.per_axis, slew, index + 1, axis);
                    note(out.axes[static_cast<size_t>(axis)], slew, index + 1, axis);
                }
            }
            bound = std::sqrt(bound);

            /* The bound is reached only if the peaks fall together, which
             * they need not, so a block that beats it is asked what it really
             * reaches -- and a block that cannot beat what has been found
             * already is not asked at all. */
            double least = out.vector.value;
            if (rotated)
            {
                for (int axis = 0; axis < 3; ++axis)
                    least = std::min(least, out.axes[static_cast<size_t>(axis)].value);
            }
            if (bound <= least)
                continue;

            double vector = 0.0;
            double axes[3];
            profile.weigh(row, &vector, axes);

            note(out.vector, vector, index + 1, -1);
            if (rotated)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    note(out.per_axis, axes[axis], index + 1, axis);
                    note(out.axes[static_cast<size_t>(axis)], axes[axis], index + 1, axis);
                }
            }
        }
        return out;
    }

    ContinuityReport continuity(const Sequence& seq, const GradientLimits& limits)
    {
        ContinuityReport out;

        const double raster = limits.grad_raster_time > 0.0 ? limits.grad_raster_time
                                                            : seq.grad_raster_time();
        /* What a gradient may move between two neighbouring raster points
         * without asking for more than the limit. */
        const double step_allowed = limits.max_slew * raster;

        const int32_t* events = seq.block_events();
        const int blocks = seq.num_blocks();

        /* Where each axis was left, in the frame the amplifiers work in. */
        double left_at[3] = {0.0, 0.0, 0.0};

        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;

            double begins[3] = {0.0, 0.0, 0.0};
            double ends[3] = {0.0, 0.0, 0.0};
            for (int axis = 0; axis < 3; ++axis)
                edges_of(seq, row[1 + axis], &begins[axis], &ends[axis]);

            /* A block that turns its gradients plays them on other axes, and
             * it is the axis an amplifier drives that has to carry on. */
            const int32_t turned = row[BLOCK_WIDTH - 1];
            if (turned >= 1 && turned <= seq.rotation_library().size())
            {
                double matrix[3][3];
                rotation_matrix(seq.rotation_library().row(turned), matrix);
                rotate(matrix, begins);
                rotate(matrix, ends);
            }

            for (int axis = 0; axis < 3; ++axis)
            {
                const double jump = std::fabs(begins[axis] - left_at[axis]);
                if (limits.max_slew > 0.0 && jump > step_allowed)
                {
                    Discontinuity found;
                    found.block = index + 1;
                    found.axis = axis;
                    found.before = left_at[axis];
                    found.after = begins[axis];
                    found.slew = jump / raster;
                    found.limit = limits.max_slew;
                    out.discontinuities.push_back(found);
                }
                left_at[axis] = ends[axis];
            }
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            if (std::fabs(left_at[axis]) > kEps)
                out.ends_at_zero = false;
        }
        return out;
    }

} // namespace pulseq
