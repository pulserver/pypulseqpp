/**
 * @file kspace.cpp
 * @brief Integrating the gradients into a trajectory.  See kspace.hpp.
 */

#include "pulseq/kspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pulseq
{

    namespace
    {

        /** A nanosecond: what two gradient corners must differ by to be two. */
        constexpr double kEps = 1e-9;

        /**
         * How close two moments have to be to be one moment.
         *
         * Coarser than the nanosecond a file is written on, because a moment
         * reached two ways -- as a gradient corner and as an ADC sample --
         * has to come out as one point on the trajectory, and the two
         * arithmetics that got there do not agree to the last bit.
         */
        constexpr double kAccuracy = 1e-10;

        /** A gap small enough to put a point either side of the waveform in. */
        constexpr double kTiny = 1e-12;

        double snapped(double when)
        {
            return kAccuracy * std::nearbyint(when / kAccuracy);
        }

        /**
         * One axis's gradient over the whole span, ready to integrate.
         *
         * A spline is only defined between its knots, so an axis that starts
         * late or stops early is held at zero either side of what it plays;
         * an axis that plays nothing at all still has a background gradient
         * to carry if it was given one.
         */
        void pad(
            const std::vector<double>& times,
            const std::vector<double>& values,
            double delay,
            double offset,
            double total,
            std::vector<double>& into_t,
            std::vector<double>& into_v,
            std::vector<std::string>& warnings)
        {
            into_t.clear();
            into_v.clear();

            if (times.empty())
            {
                if (std::fabs(offset) <= kEps)
                    return;
                into_t = {0.0, total};
                into_v = {offset, offset};
                return;
            }

            const bool late = times.front() - delay > kTiny;
            const bool early = times.back() - delay < total - kEps;

            into_t.reserve(times.size() + 4);
            into_v.reserve(values.size() + 4);

            if (late)
            {
                into_t.push_back(-kTiny);
                into_t.push_back(times.front() - delay - kTiny);
                into_v.push_back(0.0);
                into_v.push_back(0.0);
            }
            for (size_t i = 0; i < times.size(); ++i)
            {
                into_t.push_back(times[i] - delay);
                into_v.push_back(values[i]);
            }
            if (early)
            {
                into_t.push_back(times.back() - delay + kTiny);
                into_t.push_back(total + kTiny);
                into_v.push_back(0.0);
                into_v.push_back(0.0);
            }

            if (std::fabs(offset) > kEps)
            {
                for (size_t i = 0; i < into_v.size(); ++i)
                    into_v[i] += offset;
            }

            /* Two corners at the same moment are one corner, and a waveform
             * that goes backwards in time is not one. */
            size_t kept = 1;
            bool backwards = false;
            for (size_t i = 1; i < into_t.size(); ++i)
            {
                const double step = into_t[i] - into_t[i - 1];
                if (step <= 0.0)
                    backwards = true;
                if (step > kEps)
                {
                    into_t[kept] = into_t[i];
                    into_v[kept] = into_v[i];
                    ++kept;
                }
            }
            if (backwards)
                warnings.push_back(
                    "Warning: not all elements of the generated time vector are unique "
                    "and sorted in accending order!");
            into_t.resize(kept);
            into_v.resize(kept);

            for (size_t i = 0; i < into_v.size(); ++i)
            {
                if (into_v[i] == 0.0)
                    into_v[i] = 0.0; // -0.0 reads back as 0.0
            }
        }

        /** The gradient at @p when, zero outside what the axis plays. */
        double gradient_at(
            const std::vector<double>& times,
            const std::vector<double>& values,
            double when)
        {
            if (times.size() < 2 || when < times.front() || when > times.back())
                return 0.0;
            const size_t after = static_cast<size_t>(std::distance(
                times.begin(), std::lower_bound(times.begin(), times.end(), when)));
            if (after == 0)
                return values.front();
            if (after >= times.size())
                return values.back();
            const double span = times[after] - times[after - 1];
            if (span <= 0.0)
                return values[after];
            const double along = (when - times[after - 1]) / span;
            return values[after - 1] + along * (values[after] - values[after - 1]);
        }

        /**
         * The phase accumulated up to each of @p when, in order.
         *
         * Both the corners and the moments asked about are sorted, so this
         * walks them together. The integral of a straight line is a parabola,
         * so what it reports between two corners is exact rather than
         * sampled.
         */
        void integrate_at(
            const std::vector<double>& t,
            const std::vector<double>& v,
            const std::vector<double>& when,
            std::vector<double>& into)
        {
            into.assign(when.size(), 0.0);
            if (t.size() < 2)
                return;

            double accumulated = 0.0;
            size_t piece = 0;
            double held = 0.0;
            bool started = false;

            for (size_t i = 0; i < when.size(); ++i)
            {
                if (when[i] < t.front())
                    continue;
                if (when[i] > t.back())
                {
                    // Past the last gradient the phase stays where it was.
                    into[i] = started ? held : 0.0;
                    continue;
                }
                while (piece + 2 < t.size() && when[i] > t[piece + 1])
                {
                    accumulated +=
                        0.5 * (v[piece] + v[piece + 1]) * (t[piece + 1] - t[piece]);
                    ++piece;
                }
                const double within = when[i] - t[piece];
                const double width = t[piece + 1] - t[piece];
                const double slope = width > 0.0 ? (v[piece + 1] - v[piece]) / width : 0.0;
                into[i] = accumulated + v[piece] * within + 0.5 * slope * within * within;
                held = into[i];
                started = true;
            }
        }

        /**
         * Where the samples were taken, without the trajectory in between.
         *
         * The pulses still have to be accounted for -- an excitation puts the
         * phase back at the origin and a refocusing turns it around -- so the
         * phase is read at each of them too, and the shift each period
         * carries follows from those alone.
         */
        void samples_alone(Kspace& out)
        {
            /* The moments the shifts are worked out at: the start, then every
             * pulse in the order it acts. */
            std::vector<double> pulses;
            pulses.push_back(0.0);
            std::vector<char> is_excitation;
            is_excitation.push_back(0);
            {
                size_t e = 0;
                size_t r = 0;
                while (e < out.excitation_times.size() || r < out.refocusing_times.size())
                {
                    const bool take_excitation = r >= out.refocusing_times.size() ||
                        (e < out.excitation_times.size() &&
                         out.excitation_times[e] <= out.refocusing_times[r]);
                    pulses.push_back(
                        take_excitation ? out.excitation_times[e++]
                                        : out.refocusing_times[r++]);
                    is_excitation.push_back(take_excitation ? 1 : 0);
                }
            }

            std::array<std::vector<double>, 3> at_pulse;
            for (int axis = 0; axis < 3; ++axis)
            {
                integrate_at(
                    out.gradient_times[static_cast<size_t>(axis)],
                    out.gradient_values[static_cast<size_t>(axis)],
                    pulses,
                    at_pulse[static_cast<size_t>(axis)]);
                integrate_at(
                    out.gradient_times[static_cast<size_t>(axis)],
                    out.gradient_values[static_cast<size_t>(axis)],
                    out.adc_times,
                    out.sampled[static_cast<size_t>(axis)]);
            }

            /* What each period carries: the origin at an excitation, and the
             * accumulated phase reflected at a refocusing. */
            std::vector<std::array<double, 3>> shift(pulses.size());
            for (int axis = 0; axis < 3; ++axis)
                shift[0][static_cast<size_t>(axis)] =
                    -at_pulse[static_cast<size_t>(axis)][0];
            for (size_t p = 1; p < pulses.size(); ++p)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double k = at_pulse[static_cast<size_t>(axis)][p];
                    shift[p][static_cast<size_t>(axis)] = is_excitation[p]
                        ? -k
                        : -2.0 * k - shift[p - 1][static_cast<size_t>(axis)];
                }
            }

            /* Each sample belongs to the period the pulse before it began. */
            size_t period = 0;
            for (size_t i = 0; i < out.adc_times.size(); ++i)
            {
                while (period + 1 < pulses.size() && pulses[period + 1] <= out.adc_times[i])
                    ++period;
                for (int axis = 0; axis < 3; ++axis)
                    out.sampled[static_cast<size_t>(axis)][i] +=
                        shift[period][static_cast<size_t>(axis)];
            }
        }

    } // namespace

    Kspace calculate_kspace(const Sequence& seq, const KspaceOptions& options)
    {
        Kspace out;

        WaveformOptions expanding;
        expanding.append_rf = false;
        expanding.first_block = options.first_block;
        expanding.last_block = options.last_block;
        expanding.b0 = options.b0;
        expanding.gamma = options.gamma;
        const Waveforms played = waveforms_and_times(seq, expanding);
        out.warnings = played.warnings;

        const double total = played.duration;
        const double grad_raster = seq.grad_raster_time();
        const double rf_raster = seq.rf_raster_time();

        out.adc_times = played.adc_times;
        out.adc_modulation = played.adc_modulation;
        for (size_t i = 0; i < played.excitation.size(); ++i)
            out.excitation_times.push_back(played.excitation[i].time);
        for (size_t i = 0; i < played.refocusing.size(); ++i)
            out.refocusing_times.push_back(played.refocusing[i].time);

        for (int axis = 0; axis < 3; ++axis)
        {
            pad(played.times[static_cast<size_t>(axis)],
                played.amplitudes[static_cast<size_t>(axis)],
                options.delay[static_cast<size_t>(axis)],
                options.offset[static_cast<size_t>(axis)],
                total,
                out.gradient_times[static_cast<size_t>(axis)],
                out.gradient_values[static_cast<size_t>(axis)],
                out.warnings);
        }

        /* Every moment the trajectory has to be known at: the corners it
         * changes direction at, the raster through each ramp because it
         * curves there, the moments the pulses act -- and just before them,
         * a pulse making it discontinuous -- every sample, and the ends. */
        if (options.samples_only)
        {
            samples_alone(out);
            return out;
        }

        /* Every moment is already in order within the stream it comes from
         * -- a gradient's corners, the raster through its ramps, the pulses,
         * the samples -- so the moments are gathered as sorted streams and
         * merged. Pouring them into one array and sorting it is the same
         * answer for several times the work, and this is the longest array
         * the calculation handles. */
        std::vector<std::vector<double>> streams;

        streams.push_back({0.0, snapped(total)});
        for (int axis = 0; axis < 3; ++axis)
        {
            const std::vector<double>& t = out.gradient_times[static_cast<size_t>(axis)];
            const std::vector<double>& v = out.gradient_values[static_cast<size_t>(axis)];
            if (t.empty())
                continue;

            std::vector<double> corners(t.size());
            for (size_t i = 0; i < t.size(); ++i)
                corners[i] = snapped(t[i]);
            streams.push_back(std::move(corners));

            /* The ramps are walked in order and their raster ranges overlap
             * by at most a tick, so keeping the last one emitted is enough to
             * come out sorted. */
            std::vector<double> ticks;
            long emitted = std::numeric_limits<long>::min();
            for (size_t i = 0; i + 1 < t.size(); ++i)
            {
                const double slope = (v[i + 1] - v[i]) / (t[i + 1] - t[i]);
                if (std::fabs(slope / 2.0) <= kEps)
                    continue;
                const long first = static_cast<long>(std::floor(t[i] / grad_raster));
                const long last = static_cast<long>(std::ceil(t[i + 1] / grad_raster));
                for (long tick = std::max(first, emitted + 1); tick <= last; ++tick)
                    ticks.push_back(snapped(static_cast<double>(tick) * grad_raster));
                emitted = std::max(emitted, last);
            }
            if (!ticks.empty())
                streams.push_back(std::move(ticks));
        }

        for (int back = 0; back < 3; ++back)
        {
            std::vector<double> before(out.excitation_times.size());
            for (size_t i = 0; i < out.excitation_times.size(); ++i)
                before[i] = snapped(
                    out.excitation_times[i] - static_cast<double>(back) * rf_raster);
            if (!before.empty())
                streams.push_back(std::move(before));
        }
        for (int back = 0; back < 2; ++back)
        {
            std::vector<double> before(out.refocusing_times.size());
            for (size_t i = 0; i < out.refocusing_times.size(); ++i)
                before[i] = snapped(
                    out.refocusing_times[i] - static_cast<double>(back) * rf_raster);
            if (!before.empty())
                streams.push_back(std::move(before));
        }
        {
            std::vector<double> samples(out.adc_times.size());
            for (size_t i = 0; i < out.adc_times.size(); ++i)
                samples[i] = snapped(out.adc_times[i]);
            if (!samples.empty())
                streams.push_back(std::move(samples));
        }

        /* Merged in pairs, so the whole set is passed over log(streams)
         * times rather than sorted. */
        std::vector<double> merged;
        while (streams.size() > 1)
        {
            std::vector<std::vector<double>> next;
            for (size_t i = 0; i + 1 < streams.size(); i += 2)
            {
                merged.resize(streams[i].size() + streams[i + 1].size());
                std::merge(
                    streams[i].begin(), streams[i].end(),
                    streams[i + 1].begin(), streams[i + 1].end(),
                    merged.begin());
                next.push_back(merged);
            }
            if (streams.size() % 2 == 1)
                next.push_back(std::move(streams.back()));
            streams.swap(next);
        }
        out.times = streams.empty() ? std::vector<double>() : std::move(streams.front());
        out.times.erase(std::unique(out.times.begin(), out.times.end()), out.times.end());

        const size_t moments = out.times.size();
        for (int axis = 0; axis < 3; ++axis)
            out.position[static_cast<size_t>(axis)].assign(moments, 0.0);

        /* The integral, walked once: both the corners and the moments asked
         * about are in order, so neither is searched for. */
        for (int axis = 0; axis < 3; ++axis)
        {
            const std::vector<double>& t = out.gradient_times[static_cast<size_t>(axis)];
            const std::vector<double>& v = out.gradient_values[static_cast<size_t>(axis)];
            if (t.size() < 2)
                continue;

            std::vector<double>& k = out.position[static_cast<size_t>(axis)];
            const double begins = snapped(t.front());
            const double ends = snapped(t.back());

            double accumulated = 0.0;
            size_t piece = 0;
            size_t last_inside = 0;
            bool any = false;

            for (size_t i = 0; i < moments; ++i)
            {
                const double when = out.times[i];
                if (when < begins || when > ends)
                    continue;
                while (piece + 2 < t.size() && when > t[piece + 1])
                {
                    /* Whole pieces behind us contribute their area once. */
                    const double width = t[piece + 1] - t[piece];
                    accumulated += 0.5 * (v[piece] + v[piece + 1]) * width;
                    ++piece;
                }
                const double within = when - t[piece];
                const double width = t[piece + 1] - t[piece];
                const double slope = width > 0.0 ? (v[piece + 1] - v[piece]) / width : 0.0;
                k[i] = accumulated + v[piece] * within + 0.5 * slope * within * within;
                last_inside = i;
                any = true;
            }

            /* After the last gradient the phase stays where it was left. */
            if (any)
            {
                for (size_t i = last_inside + 1; i < moments; ++i)
                    k[i] = k[last_inside];
            }
        }

        /* Where each excitation put its slice: a selective pulse excites
         * where its frequency offset matches what the gradient makes the
         * Larmor frequency there, so one over the other is the position. */
        for (int axis = 0; axis < 3; ++axis)
        {
            const std::vector<double>& t = out.gradient_times[static_cast<size_t>(axis)];
            const std::vector<double>& v = out.gradient_values[static_cast<size_t>(axis)];
            std::vector<double>& where = out.slice_position[static_cast<size_t>(axis)];
            where.assign(played.excitation.size(), 0.0);
            if (t.empty())
                continue;
            for (size_t i = 0; i < played.excitation.size(); ++i)
            {
                const double gradient = gradient_at(t, v, played.excitation[i].time);
                const double position = played.excitation[i].frequency / gradient;
                where[i] = std::isfinite(position) ? position : 0.0;
            }
        }

        /* The pulses. Between two of them the trajectory is the integral as
         * it stands, shifted to begin where the last pulse left it. */
        const auto index_of = [&](double when) {
            const auto found =
                std::lower_bound(out.times.cbegin(), out.times.cend(), snapped(when));
            return static_cast<size_t>(std::distance(out.times.cbegin(), found));
        };

        std::vector<size_t> boundaries;
        boundaries.push_back(0);
        std::vector<size_t> at_excitation;
        std::vector<size_t> at_refocusing;
        for (size_t i = 0; i < out.excitation_times.size(); ++i)
        {
            at_excitation.push_back(index_of(out.excitation_times[i]));
            boundaries.push_back(at_excitation.back());
        }
        for (size_t i = 0; i < out.refocusing_times.size(); ++i)
        {
            at_refocusing.push_back(index_of(out.refocusing_times[i]));
            boundaries.push_back(at_refocusing.back());
        }
        if (moments != 0)
            boundaries.push_back(moments - 1);
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(
            std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

        double shift[3] = {0.0, 0.0, 0.0};
        for (int axis = 0; axis < 3; ++axis)
            shift[axis] = moments ? -out.position[static_cast<size_t>(axis)][0] : 0.0;

        size_t next_excitation = 0;
        size_t next_refocusing = 0;
        size_t ends_at = 0;
        for (size_t b = 0; b + 1 < boundaries.size(); ++b)
        {
            const size_t begins_at = boundaries[b];
            ends_at = boundaries[b + 1];

            if (next_excitation < at_excitation.size() &&
                at_excitation[next_excitation] == begins_at)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    shift[axis] = -out.position[static_cast<size_t>(axis)][begins_at];
                    // The moment before an excitation belongs to what ended.
                    if (begins_at > 0)
                        out.position[static_cast<size_t>(axis)][begins_at - 1] =
                            std::numeric_limits<double>::quiet_NaN();
                }
                ++next_excitation;
            }
            else if (
                next_refocusing < at_refocusing.size() &&
                at_refocusing[next_refocusing] == begins_at)
            {
                for (int axis = 0; axis < 3; ++axis)
                    shift[axis] =
                        -2.0 * out.position[static_cast<size_t>(axis)][begins_at] -
                        shift[axis];
                ++next_refocusing;
            }

            for (int axis = 0; axis < 3; ++axis)
            {
                std::vector<double>& k = out.position[static_cast<size_t>(axis)];
                for (size_t i = begins_at; i < ends_at; ++i)
                    k[i] += shift[axis];
            }
        }
        if (moments != 0)
        {
            for (int axis = 0; axis < 3; ++axis)
                out.position[static_cast<size_t>(axis)][ends_at] += shift[axis];
        }

        /* The samples are in order and every one of them is a moment the
         * trajectory knows, so where they sit is one walk down both -- not a
         * search of the whole trajectory per sample, which is the longest
         * thing this calculation would otherwise do. */
        for (int axis = 0; axis < 3; ++axis)
            out.sampled[static_cast<size_t>(axis)].resize(out.adc_times.size());
        size_t at = 0;
        for (size_t i = 0; i < out.adc_times.size(); ++i)
        {
            const double when = snapped(out.adc_times[i]);
            while (at + 1 < moments && out.times[at] < when)
                ++at;
            for (int axis = 0; axis < 3; ++axis)
                out.sampled[static_cast<size_t>(axis)][i] =
                    out.position[static_cast<size_t>(axis)][at];
        }

        return out;
    }

} // namespace pulseq
