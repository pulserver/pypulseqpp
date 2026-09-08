/**
 * @file safety.cpp
 * @brief What the gradients ask of the amplifiers.  See safety.hpp.
 */

#include "pulseq/safety.hpp"

#include "pulseq/shape.hpp"

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
        /* What the gradient may move between two neighbouring raster points
         * without asking for more than the limit. */
        const double step_allowed = limits.max_slew * raster;

        NormalisedSlew normalised(seq, raster);
        const int32_t* events = seq.block_events();
        const int blocks = seq.num_blocks();

        /* Where each axis was left by the block before, so a jump is seen. */
        double left_at[3] = {0.0, 0.0, 0.0};

        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
            double squared = 0.0;

            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                const double slew = slew_of(seq, id, normalised);
                squared += slew * slew;
                note(out.per_axis, slew, index + 1, axis);

                double begins = 0.0;
                double ends = 0.0;
                edges_of(seq, id, &begins, &ends);

                const double jump = std::fabs(begins - left_at[axis]);
                if (limits.max_slew > 0.0 && jump > step_allowed)
                {
                    Discontinuity found;
                    found.block = index + 1;
                    found.axis = axis;
                    found.before = left_at[axis];
                    found.after = begins;
                    found.slew = jump / raster;
                    found.limit = limits.max_slew;
                    out.discontinuities.push_back(found);
                }
                left_at[axis] = ends;
            }

            note(out.vector, std::sqrt(squared), index + 1, -1);
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            if (std::fabs(left_at[axis]) > kEps)
                out.ends_at_zero = false;
        }
        return out;
    }

} // namespace pulseq
