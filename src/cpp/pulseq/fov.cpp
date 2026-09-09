/**
 * @file fov.cpp
 * @brief Where the trajectory stands, block by block.  See fov.hpp.
 */

#include "pulseq/fov.hpp"

#include "pulseq/corners.hpp"
#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace pulseq
{

    namespace
    {

        constexpr double kEps = 1e-12;
        constexpr double kPi = 3.14159265358979323846;

        /**
         * What one gradient sweeps between the start of its block and @p when.
         *
         * The corners are a polyline, so this is the area under it clipped at
         * @p when -- and clipped rather than interpolated to the raster,
         * because where a pulse acts is where it acts.
         */
        double swept_by(const Corners& drawn, std::vector<double>& corner_times, double when)
        {
            if (drawn.values.empty())
                return 0.0;
            drawn.at(0.0, corner_times);

            double area = 0.0;
            for (size_t i = 0; i + 1 < corner_times.size(); ++i)
            {
                const double from = corner_times[i];
                const double to = corner_times[i + 1];
                if (when <= from)
                    break;
                const double span = to - from;
                if (span <= kEps)
                    continue;
                const double first = drawn.values[i];
                const double last = drawn.values[i + 1];
                if (when >= to)
                {
                    area += 0.5 * (first + last) * span;
                    continue;
                }
                /* Part of a segment: the polyline is straight across it, so
                 * what it sweeps is a trapezium of its own. */
                const double along = (when - from) / span;
                const double reached = first + along * (last - first);
                area += 0.5 * (first + reached) * (when - from);
                break;
            }
            return area;
        }


        /** The fractional part, which is all a phase in turns means. */
        double turns(double value)
        {
            return value - std::floor(value);
        }

        /**
         * One gradient as the block plays it: the corners, timed from the
         * block's start, and what it is worth in phase.
         *
         * The times are materialised once per block per axis; the values are
         * the cache's own, so a readout of ten thousand corners is not copied
         * to be integrated.
         */
        struct Played
        {
            std::vector<double> times;
            const std::vector<double>* values = nullptr;

            size_t count() const
            {
                return values == nullptr ? 0 : values->size();
            }

            /** What the gradient is at @p when; zero outside what it plays. */
            double at(double when) const
            {
                const size_t n = count();
                if (n == 0 || when < times.front() || when > times.back())
                    return 0.0;
                const auto found =
                    std::upper_bound(times.begin(), times.begin() + static_cast<long>(n), when);
                const size_t after = static_cast<size_t>(found - times.begin());
                if (after == 0)
                    return (*values)[0];
                if (after >= n)
                    return (*values)[n - 1];
                const double span = times[after] - times[after - 1];
                if (span <= kEps)
                    return (*values)[after];
                const double along = (when - times[after - 1]) / span;
                return (*values)[after - 1] +
                    along * ((*values)[after] - (*values)[after - 1]);
            }

            /** Whether it holds one value across `[from, to]`. */
            bool constant_over(double from, double to) const
            {
                const size_t n = count();
                if (n == 0)
                    return true;
                const double first = at(from);
                for (size_t i = 0; i < n; ++i)
                {
                    if (times[i] < from || times[i] > to)
                        continue;
                    if (std::fabs((*values)[i] - first) > 1e-10)
                        return false;
                }
                return std::fabs(at(to) - first) <= 1e-10;
            }

            /**
             * The phase, in turns, that @p shift has accumulated by @p when.
             *
             * Segment by segment, taking the fraction of each: across a scan
             * the whole is thousands of turns and only the fraction is a
             * phase, so summing first would spend the precision on the part
             * that is thrown away.
             */
            /** The area under it up to @p when, in Hz/m times seconds. */
            double swept(double when) const
            {
                const size_t n = count();
                double total = 0.0;
                for (size_t i = 0; i + 1 < n; ++i)
                {
                    const double from = times[i];
                    if (when <= from)
                        break;
                    const double to = times[i + 1];
                    const double span = to - from;
                    if (span <= kEps)
                        continue;
                    const double first = (*values)[i];
                    const double last = (*values)[i + 1];
                    if (when >= to)
                    {
                        total += 0.5 * (first + last) * span;
                        continue;
                    }
                    const double along = (when - from) / span;
                    total += 0.5 * (first + first + along * (last - first)) * (when - from);
                    break;
                }
                return total;
            }

            double swept_turns(double when, double shift) const
            {
                const size_t n = count();
                double total = 0.0;
                for (size_t i = 0; i + 1 < n; ++i)
                {
                    const double from = times[i];
                    if (when <= from)
                        break;
                    const double to = times[i + 1];
                    const double span = to - from;
                    if (span <= kEps)
                        continue;
                    const double first = (*values)[i];
                    const double last = (*values)[i + 1];
                    if (when >= to)
                    {
                        total = turns(total + turns(0.5 * (first + last) * span * shift));
                        continue;
                    }
                    const double along = (when - from) / span;
                    const double reached = first + along * (last - first);
                    total = turns(
                        total + turns(0.5 * (first + reached) * (when - from) * shift));
                    break;
                }
                return total;
            }
        };



        /**
         * When each sample of a pulse is played, measured from its delay.
         *
         * A pulse without times of its own is sampled at the centre of each
         * RF raster interval, which is where the format puts it; one with a
         * time shape says where its samples sit and those are in raster
         * units.
         */
        void when_at(const Sequence& seq, const double* rf, std::vector<double>& into)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            const int magnitude = static_cast<int>(rf[1]);
            const int time_shape = static_cast<int>(rf[3]);
            into.clear();
            if (magnitude < 1 || magnitude > shapes.size())
                return;
            const size_t count =
                static_cast<size_t>(shapes.num_uncompressed(magnitude));
            const double raster = seq.rf_raster_time();
            if (time_shape >= 1 && time_shape <= shapes.size())
            {
                const std::vector<double> ticks = decompress_shape(
                    shapes.samples(time_shape),
                    shapes.num_compressed(time_shape),
                    shapes.num_uncompressed(time_shape));
                into.reserve(ticks.size());
                for (size_t i = 0; i < ticks.size(); ++i)
                    into.push_back(ticks[i] * raster);
                return;
            }
            into.reserve(count);
            for (size_t i = 0; i < count; ++i)
                into.push_back((static_cast<double>(i) + 0.5) * raster);
        }

        /**
         * A phase shape with @p added turns folded into it, as a new row.
         *
         * Registered rather than written over: a shape is shared by every
         * event that names it, and only the ones this block plays are being
         * moved. Deduplication collapses the copies afterwards, and the added
         * turns depend on the pulse and the gradient under it and not on
         * where the block sits, so every block playing that pair lands on one
         * row.
         */
        int phase_shape_with(
            Sequence& seq, int existing, const std::vector<double>& added, double whole)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            std::vector<double> samples(added.size(), 0.0);
            if (existing >= 1 && existing <= shapes.size())
            {
                const std::vector<double> held = decompress_shape(
                    shapes.samples(existing),
                    shapes.num_compressed(existing),
                    shapes.num_uncompressed(existing));
                for (size_t i = 0; i < samples.size() && i < held.size(); ++i)
                    samples[i] = held[i];
            }
            /* Wrapped to one turn, in whatever a turn is here: an RF phase
             * shape is stored in turns and an ADC's in radians. */
            for (size_t i = 0; i < samples.size(); ++i)
            {
                const double sum = samples[i] + added[i];
                samples[i] = sum - whole * std::floor(sum / whole);
            }
            return seq.shape_library().append_raw(
                samples.data(), static_cast<int>(samples.size()));
        }

    } // namespace

    bool advance_origin(char use, const double at[3], double origin[3])
    {
        /* A pulse that records no use is read as an excitation, which is what
         * `calculate_kspace` reads it as -- one sequence cannot have two
         * stories about where its trajectory restarts. Before revision 1.5.0
         * the format had nowhere to write a use, so a file older than that
         * arrives with every pulse undefined; `detect_rf_use` is what fills
         * them in from what each pulse does. */
        if (use == 'e' || use == 'u')
        {
            /* A new trajectory: k is zero from here. */
            for (int axis = 0; axis < 3; ++axis)
                origin[axis] = 0.0;
            return true;
        }
        if (use == 'r')
        {
            /* Turned around: what was k is now -k, which is what walks a spin
             * echo back towards the origin. */
            for (int axis = 0; axis < 3; ++axis)
                origin[axis] = -at[axis];
            return true;
        }
        return false;
    }

    std::vector<std::array<double, 3>> block_k_origins(
        const Sequence& seq, int first, int last, double carry[3])
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;

        std::vector<std::array<double, 3>> out;
        if (from > to)
            return out;
        out.reserve(static_cast<size_t>(to - from + 1));

        CornerCache corners(seq);
        const int32_t* events = seq.block_events();
        const std::vector<char>& uses = seq.rf_uses();

        std::vector<double> corner_times;

        for (int index = from; index <= to; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            out.push_back({carry[0], carry[1], carry[2]});

            /* Where the block's pulse acts, if it has one, and what it is for.
             * The centre rather than the block boundary: a refocusing between
             * two crushers has each crusher counted on its own side. */
            double acts_at = -1.0;
            char use = 'u';
            const int32_t rf_id = row[0];
            if (rf_id > 0)
            {
                const double* rf = seq.rf_library().row(rf_id);
                acts_at = rf[5] + rf[4];
                use = rf_id <= static_cast<int32_t>(uses.size())
                    ? uses[static_cast<size_t>(rf_id) - 1]
                    : 'u';
            }

            double whole[3];
            double before[3];
            double at_pulse[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                whole[axis] = swept_by(drawn, corner_times, 1e30);
                before[axis] =
                    acts_at >= 0.0 ? swept_by(drawn, corner_times, acts_at) : 0.0;
                at_pulse[axis] = carry[axis] + before[axis];
            }

            if (acts_at >= 0.0 && advance_origin(use, at_pulse, carry))
            {
                /* The origin moved where the pulse acts, so only what the
                 * block sweeps after it counts towards the next block. */
                for (int axis = 0; axis < 3; ++axis)
                    carry[axis] += whole[axis] - before[axis];
            }
            else
            {
                for (int axis = 0; axis < 3; ++axis)
                    carry[axis] += whole[axis];
            }
        }
        return out;
    }


    void apply_fov_shift(
        Sequence& seq,
        const double shift_m[3],
        FovShiftScope scope,
        int first,
        int last,
        double carry[3])
    {
        const int blocks = seq.num_blocks();
        const int from = first > 1 ? first : 1;
        const int to = (last > 0 && last < blocks) ? last : blocks;
        if (from > to)
            return;

        bool moves = false;
        for (int axis = 0; axis < 3; ++axis)
            moves = moves || std::fabs(shift_m[axis]) > 0.0;

        if (!moves)
            return;

        CornerCache corners(seq);
        const int32_t* events = seq.block_events();
        Played played[3];

        /** An axis whose gradient moves under an event, and what it was worth. */
        struct Turning
        {
            int axis;
            double slope;
            double swept;
            double at;
        };
        std::vector<Turning> turning;
        std::vector<double> moment;
        std::vector<double> added;

        for (int index = from; index <= to; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            for (int axis = 0; axis < 3; ++axis)
            {
                const Corners& drawn = corners[row[1 + axis]];
                played[axis].values = drawn.values.empty() ? nullptr : &drawn.values;
                if (played[axis].values != nullptr)
                    drawn.at(0.0, played[axis].times);
            }

            /* What the block is entered with: the shift against everything
             * swept before it. Every event in the block carries it, and that
             * is the point -- what a readout is measured by is its phase
             * against the phase its own excitation was given, so the two have
             * to be counted from the same place. */
            double entering = 0.0;
            for (int axis = 0; axis < 3; ++axis)
                entering = turns(entering + turns(shift_m[axis] * carry[axis]));

            const int32_t rf_id = row[0];
            if (rf_id > 0)
            {
                turning.clear();
                double* rf = seq.rf_library().row(rf_id);
                const double delay = rf[5];
                const double centre = delay + rf[4];
                double frequency = 0.0;
                double phase = entering;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (std::fabs(shift_m[axis]) == 0.0 || played[axis].values == nullptr)
                        continue;
                    /* A pulse under a gradient that does not change is a
                     * frequency and a phase; one under a gradient that does
                     * needs its shape, and is referenced to its own centre so
                     * that what the pulse does is untouched. */
                    const bool steady = played[axis].constant_over(delay, centre);
                    const double at = steady ? delay : centre;
                    const double slope = played[axis].at(at);
                    const double swept = played[axis].swept_turns(at, shift_m[axis]);
                    frequency += shift_m[axis] * slope;
                    phase = turns(
                        phase +
                        turns(swept - shift_m[axis] * slope * (at - delay)));
                    if (!steady)
                        turning.push_back({axis, slope, swept, at});
                }
                rf[8] += frequency;
                rf[9] += 2.0 * kPi * phase;

                if (!turning.empty())
                {
                    /* The gradient moves under the pulse, so two numbers
                     * cannot say what the shift does to it: what is left over
                     * goes into the phase the pulse is played with, sample by
                     * sample and referenced to the pulse's own centre so that
                     * what the pulse *does* is untouched. */
                    when_at(seq, rf, moment);
                    added.assign(moment.size(), 0.0);
                    for (const Turning& axis : turning)
                    {
                        for (size_t i = 0; i < moment.size(); ++i)
                        {
                            added[i] = turns(
                                added[i] +
                                turns(
                                    played[axis.axis].swept_turns(
                                        moment[i] + delay, shift_m[axis.axis]) -
                                    axis.slope * (moment[i] - rf[4]) * shift_m[axis.axis] -
                                    axis.swept));
                        }
                    }
                    rf[2] = static_cast<double>(
                        phase_shape_with(seq, static_cast<int>(rf[2]), added, 1.0));
                }
            }

            const int32_t adc_id = row[4];
            if (adc_id > 0 && scope == FovShiftScope::RfAndAdc)
            {
                turning.clear();
                double* adc = seq.adc_library().row(adc_id);
                const double delay = adc[2];
                const double middle = delay + 0.5 * adc[1] * adc[0];
                double frequency = 0.0;
                double phase = entering;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (std::fabs(shift_m[axis]) == 0.0 || played[axis].values == nullptr)
                        continue;
                    const bool steady = played[axis].constant_over(delay, middle);
                    const double at = steady ? delay : middle;
                    const double slope = played[axis].at(at);
                    const double swept = played[axis].swept_turns(at, shift_m[axis]);
                    frequency += shift_m[axis] * slope;
                    phase = turns(
                        phase +
                        turns(swept - shift_m[axis] * slope * (at - delay)));
                    if (!steady)
                        turning.push_back({axis, slope, swept, at});
                }
                adc[5] += frequency;
                adc[6] += 2.0 * kPi * phase;

                if (!turning.empty())
                {
                    /* What a frequency and a phase cannot say. Under a
                     * gradient that does not move this is identically zero,
                     * which is why a Cartesian readout costs two numbers and
                     * carries no shape at all.
                     *
                     * The reconstructor does not need this -- it has the
                     * trajectory and applies the shift itself, which is what
                     * lets it re-apply one without the sequence being touched
                     * again. It is here because a file handed to another
                     * toolbox has nobody to do that for it. */
                    const int samples = static_cast<int>(adc[0]);
                    const double dwell = adc[1];
                    added.assign(static_cast<size_t>(samples), 0.0);
                    for (const Turning& axis : turning)
                    {
                        for (int i = 0; i < samples; ++i)
                        {
                            const double when =
                                delay + dwell * (static_cast<double>(i) + 0.5);
                            const double left =
                                played[axis.axis].swept_turns(when, shift_m[axis.axis]) -
                                axis.swept -
                                shift_m[axis.axis] * axis.slope * (when - axis.at);
                            added[static_cast<size_t>(i)] = turns(
                                added[static_cast<size_t>(i)] + turns(left));
                        }
                    }
                    for (size_t i = 0; i < added.size(); ++i)
                        added[i] *= 2.0 * kPi;
                    adc[7] = static_cast<double>(phase_shape_with(
                        seq, static_cast<int>(adc[7]), added, 2.0 * kPi));
                }
            }

            /* What this block swept, added to the unbroken running total.
             * Not where the trajectory stands -- `block_k_origins` is that,
             * and it restarts at every excitation. A phase means something
             * only as a difference, and the difference a readout is measured
             * by is against its own excitation, so resetting between the two
             * would reference them to different zeros and put a phase on the
             * signal that is not the shift. */
            for (int axis = 0; axis < 3; ++axis)
            {
                if (played[axis].values != nullptr)
                    carry[axis] += played[axis].swept(1e30);
            }
        }
    }

} // namespace pulseq
