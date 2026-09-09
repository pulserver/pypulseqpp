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
         * What the three gradients of one block do together, moment by
         * moment.
         *
         * A gradient is a handful of points and straight lines between them,
         * so what the three ask for together is decided at the moments any of
         * them turns a corner: the strongest they reach between them is at one
         * of those points, and how fast they change is constant between two of
         * them. Taking each axis's own peak and combining those answers a
         * different question -- what the amplifiers would be asked for if the
         * peaks happened at once, which they need not.
         *
         * What is walked is the stored representation, not an expanded
         * waveform: the points are the trapezoid's corners and the shape's
         * samples where they are, which is what keeps this a pass over the
         * events a block names rather than over the scan.
         */
        class BlockProfile
        {
        public:
            BlockProfile(const Sequence& seq, double grad_raster)
                : seq_(seq), raster_(grad_raster), shapes_(seq.shape_library())
            {
            }

            /**
             * The strongest the block at @p row gets.
             *
             * @param row      Its block table row.
             * @param vector   Filled with the longest the gradient vector gets.
             * @param axes     Filled with the most each amplifier is asked
             *                 for, once the block's own rotation is applied.
             */
            void amplitude(const int32_t* row, double* vector, double axes[3])
            {
                if (!gather(row))
                    return;
                size_t at[3] = {0, 0, 0};
                for (size_t i = 0; i < edges_.size(); ++i)
                {
                    double here[3];
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        here[axis] = drawn(axis, edges_[i], at[static_cast<size_t>(axis)]);
                    }
                    take(here, vector, axes);
                }
            }

            /** The steepest the block at @p row changes.  See `amplitude`. */
            void slew(const int32_t* row, double* vector, double axes[3])
            {
                if (!gather(row))
                    return;
                size_t at[3] = {0, 0, 0};
                for (size_t i = 0; i + 1 < edges_.size(); ++i)
                {
                    double here[3];
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        here[axis] =
                            sloped(axis, edges_[i], at[static_cast<size_t>(axis)]);
                    }
                    take(here, vector, axes);
                }
            }

        private:
            /**
             * Read the block's three gradients and put their corners on one
             * time base.  @return whether there is anything to weigh.
             */
            bool gather(const int32_t* row)
            {
                edges_.clear();
                for (int axis = 0; axis < 3; ++axis)
                {
                    knots_of(
                        row[1 + axis],
                        times_[static_cast<size_t>(axis)],
                        values_[static_cast<size_t>(axis)]);
                    edges_.insert(
                        edges_.end(),
                        times_[static_cast<size_t>(axis)].begin(),
                        times_[static_cast<size_t>(axis)].end());
                }
                std::sort(edges_.begin(), edges_.end());
                edges_.erase(std::unique(edges_.begin(), edges_.end()), edges_.end());

                turned_ = false;
                const int32_t rotation = row[BLOCK_ROTATION_COLUMN];
                if (rotation >= 1 && rotation <= seq_.rotation_library().size())
                {
                    turned_ = true;
                    rotation_matrix(seq_.rotation_library().row(rotation), matrix_);
                }
                return !edges_.empty();
            }

            /** Record one moment's three numbers against what is worst so far. */
            void take(double here[3], double* vector, double axes[3]) const
            {
                /* How much is asked for between the amplifiers is what a turn
                 * leaves alone -- turning a vector does not change how long it
                 * is -- so the magnitude is read off before the turn and each
                 * amplifier's share after it. */
                *vector = std::max(
                    *vector,
                    std::sqrt(
                        here[0] * here[0] + here[1] * here[1] + here[2] * here[2]));
                if (turned_)
                    rotate(matrix_, here);
                for (int axis = 0; axis < 3; ++axis)
                {
                    axes[axis] =
                        std::max(axes[axis], std::fabs(here[static_cast<size_t>(axis)]));
                }
            }

            /** What axis @p axis is at at @p when, walking forward. */
            double drawn(int axis, double when, size_t& at) const
            {
                const std::vector<double>& times = times_[static_cast<size_t>(axis)];
                const std::vector<double>& values = values_[static_cast<size_t>(axis)];
                /* An axis is at zero wherever this block is not playing on it. */
                if (times.empty() || when + kEps < times.front() ||
                    when > times.back() + kEps)
                    return 0.0;
                while (at + 1 < times.size() && times[at + 1] <= when + kEps)
                    ++at;
                if (at + 1 >= times.size())
                    return values.back();
                const double span = times[at + 1] - times[at];
                if (span <= kEps)
                    return values[at + 1];
                return values[at] +
                    (values[at + 1] - values[at]) * (when - times[at]) / span;
            }

            /** What axis @p axis changes at from @p when.  See `drawn`. */
            double sloped(int axis, double when, size_t& at) const
            {
                const std::vector<double>& times = times_[static_cast<size_t>(axis)];
                const std::vector<double>& values = values_[static_cast<size_t>(axis)];
                if (times.size() < 2 || when + kEps < times.front() ||
                    when + kEps >= times.back())
                    return 0.0;
                while (at + 1 < times.size() && times[at + 1] <= when + kEps)
                    ++at;
                const double span = times[at + 1] - times[at];
                /* A step that takes no time asks for an infinite rate, which
                 * is a fault of the event rather than a rate, and is left out
                 * here as it is everywhere else. */
                if (span <= kEps)
                    return 0.0;
                return (values[at + 1] - values[at]) / span;
            }

            /**
             * One gradient as the points it turns a corner at: @p times and
             * @p values are the same length, and the gradient is drawn
             * straight between them.
             */
            void knots_of(
                int32_t id, std::vector<double>& times, std::vector<double>& values)
            {
                times.clear();
                values.clear();
                if (id <= 0 || id > seq_.num_gradients())
                    return;
                const int row = seq_.grad_row(id);

                if (seq_.grad_kind(id) == GradKind::Trap)
                {
                    const double* trap = seq_.trap_library().row(row);
                    const double amplitude = trap[0];
                    double when = trap[4];

                    /* A trapezoid with no ramps and no flat top is no shape at
                     * all, and what it asks for is its amplitude at an instant.
                     * The waveform expansion warns about these; here it is the
                     * one point there is. */
                    if (trap[1] + trap[2] + trap[3] <= kEps)
                    {
                        if (std::fabs(amplitude) > kEps)
                        {
                            times.push_back(when);
                            values.push_back(amplitude);
                        }
                        return;
                    }

                    times.push_back(when);
                    values.push_back(0.0);
                    /* A stretch that takes no time is a step rather than a
                     * corner: it moves where the gradient already is. */
                    const auto reach = [&](double span, double value) {
                        if (span <= kEps)
                        {
                            values.back() = value;
                            return;
                        }
                        when += span;
                        times.push_back(when);
                        values.push_back(value);
                    };
                    reach(trap[1], amplitude);
                    reach(trap[2], amplitude);
                    reach(trap[3], 0.0);
                    return;
                }

                const double* arb = seq_.arb_library().row(row);
                const double amplitude = arb[0];
                const std::vector<double>& shape = shapes_[static_cast<int>(arb[3])];
                if (shape.empty())
                    return;
                const int time_shape = static_cast<int>(arb[4]);
                const double delay = arb[5];

                if (time_shape <= 0)
                {
                    /* Evenly spaced, at the centre of each interval; a time id
                     * of -1 marks a waveform oversampled by two. */
                    const double interval = time_shape == -1 ? raster_ / 2.0 : raster_;
                    if (interval <= kEps)
                        return;
                    times.reserve(shape.size());
                    values.reserve(shape.size());
                    for (size_t i = 0; i < shape.size(); ++i)
                    {
                        times.push_back(delay + (static_cast<double>(i) + 0.5) * interval);
                        values.push_back(amplitude * shape[i]);
                    }
                    return;
                }

                const std::vector<double>& ticks = shapes_[time_shape];
                const size_t count = std::min(shape.size(), ticks.size());
                times.reserve(count);
                values.reserve(count);
                for (size_t i = 0; i < count; ++i)
                {
                    times.push_back(delay + ticks[i] * raster_);
                    values.push_back(amplitude * shape[i]);
                }
            }

            const Sequence& seq_;
            double raster_;
            ShapeCache shapes_;
            /* Kept across blocks so a scan allocates once. */
            std::vector<double> times_[3];
            std::vector<double> values_[3];
            std::vector<double> edges_;
            bool turned_ = false;
            double matrix_[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        };

    } // namespace

    namespace
    {

        /**
         * Weigh every block, exactly, without weighing every block.
         *
         * Each axis's own peak is one number read off its library row, and
         * combining the three bounds what they reach together. The bound is
         * reached only if the peaks fall together, which they need not -- so a
         * block that beats it is asked what it really reaches, and a block
         * that cannot beat what has been found already is not asked at all.
         *
         * @param seq      The sequence to weigh.
         * @param out      The report to fill: a GradientReport or a
         *                 SlewReport, which say the same three things.
         * @param profile  How to ask a block what it really reaches.
         * @param exactly  `&BlockProfile::amplitude` or `&BlockProfile::slew`.
         * @param alone    What one gradient reaches on its own.
         */
        template <typename Report, typename Alone>
        void weigh_every_block(
            const Sequence& seq,
            Report& out,
            BlockProfile& profile,
            void (BlockProfile::*exactly)(const int32_t*, double*, double[3]),
            Alone alone)
        {
            const int32_t* events = seq.block_events();
            const int blocks = seq.num_blocks();

            for (int index = 0; index < blocks; ++index)
            {
                const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
                const int32_t rotation = row[BLOCK_ROTATION_COLUMN];
                const bool turned =
                    rotation >= 1 && rotation <= seq.rotation_library().size();

                double bound = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double reaches = alone(row[1 + axis]);
                    bound += reaches * reaches;
                    /* Played as written, an amplifier is asked for exactly
                     * what its own gradient asks for. Played turned, it is
                     * asked for a share of all three, and only the block knows
                     * what share. */
                    if (!turned)
                    {
                        note(out.per_axis, reaches, index + 1, axis);
                        note(
                            out.axes[static_cast<size_t>(axis)], reaches, index + 1, axis);
                    }
                }
                bound = std::sqrt(bound);

                double least = out.vector.value;
                if (turned)
                {
                    for (int axis = 0; axis < 3; ++axis)
                        least = std::min(least, out.axes[static_cast<size_t>(axis)].value);
                }
                if (bound <= least)
                    continue;

                double vector = 0.0;
                double axes[3] = {0.0, 0.0, 0.0};
                (profile.*exactly)(row, &vector, axes);

                note(out.vector, vector, index + 1, -1);
                if (turned)
                {
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        note(out.per_axis, axes[axis], index + 1, axis);
                        note(
                            out.axes[static_cast<size_t>(axis)], axes[axis], index + 1,
                            axis);
                    }
                }
            }
        }

    } // namespace

    GradientReport max_gradient(const Sequence& seq)
    {
        GradientReport out;
        BlockProfile profile(seq, seq.grad_raster_time());
        weigh_every_block(
            seq, out, profile, &BlockProfile::amplitude,
            [&seq](int32_t id) { return std::fabs(amplitude_of(seq, id)); });
        return out;
    }

    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits)
    {
        const double raster = limits.grad_raster_time > 0.0 ? limits.grad_raster_time
                                                            : seq.grad_raster_time();
        NormalisedSlew normalised(seq, raster);
        BlockProfile profile(seq, raster);

        SlewReport out;
        weigh_every_block(
            seq, out, profile, &BlockProfile::slew,
            [&](int32_t id) { return slew_of(seq, id, normalised); });
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
            const int32_t turned = row[BLOCK_ROTATION_COLUMN];
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
