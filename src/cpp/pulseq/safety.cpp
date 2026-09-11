/**
 * @file safety.cpp
 * @brief What the gradients ask of the amplifiers.  See safety.hpp.
 */

#include "pulseq/safety.hpp"

#include "pulseq/corners.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace pulseq
{

    namespace
    {

        constexpr double kEps = 1e-12;

        /** What one gradient reaches on its own, over its own corners. */
        struct Reach
        {
            double amplitude = 0.0;
            double slew = 0.0;
            /** Signed extremes, with zero among them. */
            double low = 0.0;
            double high = 0.0;
            bool known = false;
        };

        /**
         * Where a gradient starts and ends, in Hz/m.
         *
         * The recorded first and last value, which is what the restoration
         * draws from and lands on, so this is the drawn waveform's own ends
         * without drawing it.
         */
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
                : seq_(seq),
                  corners_(seq, grad_raster),
                  reach_(static_cast<size_t>(seq.num_gradients()) + 1)
            {
            }

            /**
             * What gradient @p id reaches on its own, worked out once.
             *
             * A gradient draws the same corners every time it is played, so
             * the strongest it gets and the fastest it changes belong to the
             * event rather than to the block: a readout played a hundred
             * thousand times costs one pass over its corners.
             */
            const Reach& alone(int32_t id)
            {
                if (id <= 0 || id > seq_.num_gradients())
                    return nothing_;
                Reach& found = reach_[static_cast<size_t>(id)];
                if (found.known)
                    return found;
                found.known = true;

                const Played drawn = played_by(id, scratch_);
                for (size_t i = 0; i < drawn.count; ++i)
                {
                    found.amplitude = std::max(found.amplitude, std::fabs(drawn.values[i]));
                    found.low = std::min(found.low, drawn.values[i]);
                    found.high = std::max(found.high, drawn.values[i]);
                    if (i == 0)
                        continue;
                    const double span = drawn.times[i] - drawn.times[i - 1];
                    if (span > kEps)
                    {
                        found.slew = std::max(
                            found.slew,
                            std::fabs(drawn.values[i] - drawn.values[i - 1]) / span);
                    }
                }
                return found;
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
                        here[axis] = drawn(axis, edges_[i], at[static_cast<size_t>(axis)]);
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
                        here[axis] = sloped(axis, edges_[i], at[static_cast<size_t>(axis)]);
                    take(here, vector, axes);
                }
            }

            /** The lowest and highest each axis reaches in the block at @p row. */
            void extremes(const int32_t* row, double low[3], double high[3])
            {
                if (!gather(row))
                    return;
                size_t at[3] = {0, 0, 0};
                for (size_t i = 0; i < edges_.size(); ++i)
                {
                    double here[3];
                    for (int axis = 0; axis < 3; ++axis)
                        here[axis] = drawn(axis, edges_[i], at[static_cast<size_t>(axis)]);
                    if (turned_)
                        rotate(matrix_, here);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        low[axis] = std::min(low[axis], here[axis]);
                        high[axis] = std::max(high[axis], here[axis]);
                    }
                }
            }

        private:
            /**
             * One gradient as the block plays it: the corners it turns, and
             * how far into the block they fall.
             *
             * A view rather than a copy. The corners belong to the gradient
             * and are held once for it; what the block adds is where they
             * start, and a readout of ten thousand corners is not worth
             * copying to say so.
             */
            struct Played
            {
                const double* times = nullptr;
                const double* values = nullptr;
                size_t count = 0;
                double offset = 0.0;

                double when(size_t i) const
                {
                    return times[i] + offset;
                }
            };

            /**
             * Where gradient @p id draws, timed from the start of its block.
             *
             * These are the corners an interpreter draws between, not the
             * samples the file stores: a shape kept at the centre of each
             * raster interval turns its corners half a raster from any sample
             * it holds, and reaches values none of them do.
             *
             * @param own  Filled where the corner times are not already a run
             *             of doubles to point at: a trapezoid's, which are
             *             added up from its ramps, and the single instant an
             *             "empty" trapezoid with an amplitude asks for.
             */
            Played played_by(int32_t id, std::vector<double>& own)
            {
                Played out;
                const Corners& shape = corners_[id];
                if (shape.empty_with_amplitude)
                {
                    /* No ramps and no flat top, but an amplitude: what it asks
                     * for is that amplitude at an instant. The waveform
                     * expansion warns about these; here it is the one point
                     * there is. */
                    own.assign(1, shape.delay);
                    instant_ = shape.amplitude;
                    out.times = own.data();
                    out.values = &instant_;
                    out.count = 1;
                    return out;
                }
                if (shape.values.empty())
                    return out;

                out.values = shape.values.data();
                out.count = shape.values.size();
                if (shape.trapezoid)
                {
                    shape.at(0.0, own);
                    out.times = own.data();
                    return out;
                }
                out.times = shape.times.data();
                out.offset = shape.delay;
                return out;
            }

            /**
             * Read the block's three gradients and put their corners on one
             * time base.  @return whether there is anything to weigh.
             *
             * Each axis's corners are already in order, so the time base is a
             * merge rather than a sort: what this walks is what the three of
             * them hold, once.
             */
            bool gather(const int32_t* row)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    played_[static_cast<size_t>(axis)] =
                        played_by(row[1 + axis], own_[static_cast<size_t>(axis)]);
                }

                edges_.clear();
                size_t at[3] = {0, 0, 0};
                for (;;)
                {
                    double next = std::numeric_limits<double>::infinity();
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const Played& played = played_[static_cast<size_t>(axis)];
                        if (at[static_cast<size_t>(axis)] < played.count)
                            next = std::min(next, played.when(at[static_cast<size_t>(axis)]));
                    }
                    if (!(next < std::numeric_limits<double>::infinity()))
                        break;
                    edges_.push_back(next);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const Played& played = played_[static_cast<size_t>(axis)];
                        size_t& i = at[static_cast<size_t>(axis)];
                        while (i < played.count && played.when(i) <= next)
                            ++i;
                    }
                }

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
                    std::sqrt(here[0] * here[0] + here[1] * here[1] + here[2] * here[2]));
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
                const Played& played = played_[static_cast<size_t>(axis)];
                /* An axis is at zero wherever this block is not playing on it. */
                if (played.count == 0 || when + kEps < played.when(0) ||
                    when > played.when(played.count - 1) + kEps)
                    return 0.0;
                while (at + 1 < played.count && played.when(at + 1) <= when + kEps)
                    ++at;
                if (at + 1 >= played.count)
                    return played.values[played.count - 1];
                const double span = played.when(at + 1) - played.when(at);
                if (span <= kEps)
                    return played.values[at + 1];
                return played.values[at] +
                    (played.values[at + 1] - played.values[at]) *
                    (when - played.when(at)) / span;
            }

            /** What axis @p axis changes at from @p when.  See `drawn`. */
            double sloped(int axis, double when, size_t& at) const
            {
                const Played& played = played_[static_cast<size_t>(axis)];
                if (played.count < 2 || when + kEps < played.when(0) ||
                    when + kEps >= played.when(played.count - 1))
                    return 0.0;
                while (at + 1 < played.count && played.when(at + 1) <= when + kEps)
                    ++at;
                const double span = played.when(at + 1) - played.when(at);
                /* A step that takes no time asks for an infinite rate, which
                 * is a fault of the event rather than a rate, and is left out
                 * here as it is everywhere else. */
                if (span <= kEps)
                    return 0.0;
                return (played.values[at + 1] - played.values[at]) / span;
            }

            const Sequence& seq_;
            CornerCache corners_;
            std::vector<Reach> reach_;
            const Reach nothing_{};
            /* Kept across blocks so a scan allocates once. */
            Played played_[3];
            std::vector<double> own_[3];
            std::vector<double> scratch_;
            double instant_ = 0.0;
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
         * Each gradient's own peak is one number worked out once for it, and
         * what the three of them reach together is bounded by those: the
         * vector by their root sum of squares, and each amplifier -- once a
         * turn has spread all three over all three -- by what the rotation
         * can send it. Neither bound is reached unless the peaks fall
         * together, which they need not, so a block that beats one is asked
         * what it really reaches and a block that beats none is not asked at
         * all.
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

                double reaches[3];
                double squared = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    reaches[axis] = alone(row[1 + axis]);
                    squared += reaches[axis] * reaches[axis];
                }
                const double vector_bound = std::sqrt(squared);

                /* Played as written, an amplifier is asked for exactly what
                 * its own gradient asks for, and there is nothing to bound.
                 * Played turned, it is asked for a share of all three, and
                 * the most the turn can send it is what bounds it. */
                double axis_bound[3];
                if (!turned)
                {
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        axis_bound[axis] = 0.0;
                        note(out.per_axis, reaches[axis], index + 1, axis);
                        note(
                            out.axes[static_cast<size_t>(axis)], reaches[axis], index + 1,
                            axis);
                    }
                }
                else
                {
                    double matrix[3][3];
                    rotation_matrix(seq.rotation_library().row(rotation), matrix);
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        axis_bound[axis] = std::fabs(matrix[axis][0]) * reaches[0] +
                            std::fabs(matrix[axis][1]) * reaches[1] +
                            std::fabs(matrix[axis][2]) * reaches[2];
                    }
                }

                bool worth_asking = vector_bound > out.vector.value;
                for (int axis = 0; axis < 3 && !worth_asking; ++axis)
                    worth_asking = axis_bound[axis] > out.axes[static_cast<size_t>(axis)].value;
                if (!worth_asking)
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
            [&profile](int32_t id) { return profile.alone(id).amplitude; });
        return out;
    }

    SlewReport max_slew(const Sequence& seq, const GradientLimits& limits)
    {
        const double raster = limits.grad_raster_time > 0.0 ? limits.grad_raster_time
                                                            : seq.grad_raster_time();
        BlockProfile profile(seq, raster);

        SlewReport out;
        weigh_every_block(
            seq, out, profile, &BlockProfile::slew,
            [&profile](int32_t id) { return profile.alone(id).slew; });
        return out;
    }

    std::vector<double> block_extremes(const Sequence& seq)
    {
        const int blocks = seq.num_blocks();
        std::vector<double> out(static_cast<size_t>(blocks) * 6, 0.0);
        BlockProfile profile(seq, seq.grad_raster_time());
        const int32_t* events = seq.block_events();

        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
            double low[3] = {0.0, 0.0, 0.0};
            double high[3] = {0.0, 0.0, 0.0};
            const int32_t rotation = row[BLOCK_ROTATION_COLUMN];
            if (rotation >= 1 && rotation <= seq.rotation_library().size())
            {
                profile.extremes(row, low, high);
            }
            else
            {
                /* Played as written, each axis plays its own gradient, whose
                 * extremes belong to the event and are worked out once. */
                for (int axis = 0; axis < 3; ++axis)
                {
                    const Reach& alone = profile.alone(row[1 + axis]);
                    low[axis] = alone.low;
                    high[axis] = alone.high;
                }
            }
            double* into = out.data() + static_cast<size_t>(index) * 6;
            for (int axis = 0; axis < 3; ++axis)
            {
                into[2 * axis] = low[axis];
                into[2 * axis + 1] = high[axis];
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
