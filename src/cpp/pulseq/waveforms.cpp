/**
 * @file waveforms.cpp
 * @brief Expanding a block table into what it plays.  See waveforms.hpp.
 */

#include "pulseq/waveforms.hpp"

#include "pulseq/corners.hpp"

#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {

        /** A nanosecond: what two times have to differ by to be two times. */
        constexpr double kEps = 1e-9;
        constexpr double kPi = 3.14159265358979323846;

        std::string microseconds(double seconds)
        {
            char printed[32];
            std::snprintf(printed, sizeof printed, "%.0f", 1e6 * seconds);
            return printed;
        }

        /** One channel of the answer, stitched together as pieces arrive. */
        struct Channel
        {
            std::vector<double>* times = nullptr;
            std::vector<double>* values = nullptr;
            int axis = 0;
        };

        /**
         * Add one event's corners to a channel, closing any gap before them.
         *
         * Two events on one axis with nothing between them are one waveform;
         * with a gap they are two, and the gradient is off in between -- so
         * the one before has to reach zero and the one after start there. A
         * sequence that leaves a gradient hanging is told so rather than
         * quietly ramped.
         */
        /**
         * A gradient's value at @p when, zero outside what it plays.
         *
         * The waveform is played by interpolating between its corners, so
         * reading it between two of them is that same interpolation. Outside
         * its own span the axis is off.
         *
         * The times are the ones being asked about, not the ones a gradient
         * stores: recovering a corner's own time by taking an offset back off
         * an absolute one does not return the corner, and the endpoint then
         * lands just inside the waveform and gets interpolated instead of
         * read.
         */
        double sampled(
            const std::vector<double>& times,
            const std::vector<double>& values,
            double when)
        {
            const size_t count = times.size();
            if (count == 0 || when < times.front() || when > times.back())
                return 0.0;

            const size_t after = static_cast<size_t>(std::distance(
                times.begin(), std::lower_bound(times.begin(), times.end(), when)));
            if (after == 0)
                return values.front();
            if (after >= count)
                return values.back();

            const double span = times[after] - times[after - 1];
            if (span <= 0.0)
                return values[after];
            const double along = (when - times[after - 1]) / span;
            return values[after - 1] + along * (values[after] - values[after - 1]);
        }

        void extend(
            Channel& channel,
            const std::vector<double>& times,
            const std::vector<double>& values,
            double offset,
            double raster,
            std::vector<std::string>& warnings)
        {
            if (times.empty())
                return;

            std::vector<double>& into_t = *channel.times;
            std::vector<double>& into_v = *channel.values;

            double first_time = times.front() + offset;
            double first_value = values.front();
            /* Where the piece starts being new. Everything a gap or an
             * overlap changes is at one end or the other, so the piece is
             * never copied to be adjusted -- what changes is which sample it
             * is appended from, and one value at each edge. */
            size_t from = 0;

            if (!into_t.empty() && into_t.back() + raster < first_time)
            {
                /* A gap: the gradient is off in between, so the one before
                 * has to reach zero and the one after start there. */
                if (into_v.back() != 0.0)
                {
                    if (std::fabs(into_v.back()) > 1e-6)
                    {
                        warnings.push_back(
                            "waveforms_and_times(): forcing ramp-down from a non-zero "
                            "gradient sample on axis " +
                            std::to_string(channel.axis) + " at t=" +
                            microseconds(into_t.back()) +
                            " us \ncheck your sequence, some calculations are possibly "
                            "wrong. If using mr.makeArbitraryGrad() consider using "
                            "explicit values for 'first' and 'last' and setting them "
                            "correctly.");
                        into_t.push_back(into_t.back() + raster / 2.0);
                        into_v.push_back(0.0);
                    }
                    else
                    {
                        into_v.back() = 0.0;
                    }
                }
                if (first_value != 0.0)
                {
                    if (std::fabs(first_value) > 1e-6)
                    {
                        warnings.push_back(
                            "waveforms_and_times(): forcing ramp-up to a non-zero "
                            "gradient sample on axis " +
                            std::to_string(channel.axis) + " at t=" +
                            microseconds(first_time) +
                            " us \ncheck your sequence, some calculations are probably "
                            "wrong. If using mr.makeArbitraryGrad() consider using "
                            "explicit values for 'first' and 'last' and setting them "
                            "correctly.");
                        into_t.push_back(first_time - raster / 2.0);
                        into_v.push_back(0.0);
                    }
                    else
                    {
                        first_value = 0.0;
                    }
                }
            }

            if (!into_t.empty() && into_t.back() >= first_time - kEps)
            {
                /* The pieces meet or overlap: keep only what comes after what
                 * is already there. */
                if (first_time < into_t.back() - kEps)
                    warnings.push_back(
                        "Warning: looks like rounding errors for some elements exceed "
                        "the acceptable tolerance!");
                while (from < times.size() && times[from] + offset <= into_t.back() + kEps)
                    ++from;
            }

            const size_t held = into_t.size();
            const size_t adding = times.size() - from;
            into_t.resize(held + adding);
            into_v.resize(held + adding);
            for (size_t i = 0; i < adding; ++i)
            {
                into_t[held + i] = times[from + i] + offset;
                into_v[held + i] = values[from + i];
            }
            if (from == 0 && adding != 0)
                into_v[held] = first_value;
        }

    } // namespace

    Waveforms waveforms_and_times(const Sequence& seq, const WaveformOptions& options)
    {
        Waveforms out;

        const double grad_raster = seq.grad_raster_time();
        const double rf_raster = seq.rf_raster_time();
        const double larmor = options.gamma * options.b0;


        ShapeCache shapes(seq.shape_library());
        CornerCache corners_of(seq);

        Channel channels[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            channels[axis].times = &out.times[static_cast<size_t>(axis)];
            channels[axis].values = &out.amplitudes[static_cast<size_t>(axis)];
            channels[axis].axis = axis + 1;
        }

        const int blocks = seq.num_blocks();
        const int first = options.first_block > 1 ? options.first_block : 1;
        const int last =
            (options.last_block > 0 && options.last_block < blocks) ? options.last_block : blocks;

        const double* durations = seq.block_durations();
        const int32_t* events = seq.block_events();

        /* Reused across blocks, so a rotated scan allocates once. */
        std::vector<double> union_times;
        std::vector<double> combined;
        std::vector<double> absolute[3];

        double elapsed = 0.0;
        for (int index = first; index <= last; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            /* A rotation remaps the gradients onto other axes, which is a
             * different waveform on each, not this one moved. Which rotation
             * is a column of the block table, so this is a read rather than a
             * walk down the block's extension chain. */
            const int32_t rotation_row = row[BLOCK_ROTATION_COLUMN];

            const Corners* played[3] = {nullptr, nullptr, nullptr};
            Corners here[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0)
                    continue;
                const Corners& shape = corners_of[id];
                if (shape.empty_with_amplitude)
                {
                    out.warnings.push_back(
                        "\"empty\" gradient with non-zero magnitude detected in block " +
                        std::to_string(index));
                    continue;
                }
                if (shape.trapezoid)
                {
                    if (shape.values.empty())
                        continue;
                    Corners& built = here[axis];
                    built.trapezoid = true;
                    built.values = shape.values;
                    built.times.resize(shape.values.size());
                    double when = elapsed + shape.delay;
                    built.times[0] = when;
                    for (size_t i = 1; i < shape.values.size(); ++i)
                    {
                        when += shape.ramps[i - 1];
                        built.times[i] = when;
                    }
                    played[axis] = &built;
                }
                else if (!shape.times.empty())
                {
                    played[axis] = &shape;
                }
            }

            if (rotation_row < 1 || rotation_row > seq.rotation_library().size())
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (played[axis] == nullptr)
                        continue;
                    extend(
                        channels[axis],
                        played[axis]->times,
                        played[axis]->values,
                        played[axis]->trapezoid ? 0.0 : elapsed + played[axis]->delay,
                        grad_raster,
                        out.warnings);
                }
            }
            else
            {
                /* A rotation sends each axis's gradient onto all three, so
                 * what a rotated block plays on one axis is a sum of the
                 * three it was given. The waveforms are piecewise linear, so
                 * that sum is exact on the union of their corners: a linear
                 * combination of straight lines is a straight line between
                 * the same points. */
                double matrix[3][3];
                rotation_matrix(seq.rotation_library().row(rotation_row), matrix);

                /* Each input's corners moved to where the block plays them,
                 * once, so every later comparison is between times that were
                 * arrived at the same way. */
                union_times.clear();
                double loudest = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    absolute[axis].clear();
                    if (played[axis] == nullptr)
                        continue;
                    const double start =
                        played[axis]->trapezoid ? 0.0 : elapsed + played[axis]->delay;
                    absolute[axis].reserve(played[axis]->times.size());
                    for (size_t i = 0; i < played[axis]->times.size(); ++i)
                    {
                        absolute[axis].push_back(played[axis]->times[i] + start);
                        union_times.push_back(absolute[axis].back());
                        loudest = std::max(loudest, std::fabs(played[axis]->values[i]));
                    }
                }
                std::sort(union_times.begin(), union_times.end());
                union_times.erase(
                    std::unique(
                        union_times.begin(),
                        union_times.end(),
                        [](double a, double b) { return std::fabs(a - b) <= kEps; }),
                    union_times.end());

                /* A component too small to matter is not played at all, and
                 * an output axis that comes out silent is left alone rather
                 * than given a flat zero of its own. */
                const double floor = 1e-6;
                for (int into = 0; into < 3; ++into)
                {
                    combined.assign(union_times.size(), 0.0);
                    bool anything = false;
                    for (int from = 0; from < 3; ++from)
                    {
                        const double weight = matrix[into][from];
                        if (played[from] == nullptr || std::fabs(weight) < floor)
                            continue;
                        anything = true;
                        for (size_t i = 0; i < union_times.size(); ++i)
                            combined[i] += weight *
                                sampled(absolute[from], played[from]->values,
                                        union_times[i]);
                    }
                    if (!anything)
                        continue;

                    double peak = 0.0;
                    for (size_t i = 0; i < combined.size(); ++i)
                        peak = std::max(peak, std::fabs(combined[i]));
                    if (peak < floor * loudest)
                        continue;

                    extend(
                        channels[into],
                        union_times,
                        combined,
                        0.0,
                        grad_raster,
                        out.warnings);
                }
            }

            if (row[0] > 0)
            {
                const double* rf = seq.rf_library().row(row[0]);
                const double centre = rf[4];
                const double frequency = rf[8] + rf[6] * 1e-6 * larmor;
                const double phase = rf[7] * 1e-6 * larmor + rf[9];

                PulseMoment moment;
                moment.time = elapsed + rf[5] + centre;
                moment.frequency = frequency;
                moment.phase = phase + 2.0 * kPi * frequency * centre;

                const std::vector<char>& uses = seq.rf_uses();
                const char use = row[0] <= static_cast<int32_t>(uses.size())
                    ? uses[static_cast<size_t>(row[0]) - 1]
                    : 'u';
                out.pulses.push_back(moment);
                out.pulse_uses.push_back(use);
                out.pulse_blocks.push_back(index);

                if (options.append_rf)
                {
                    const std::vector<double>& magnitude = shapes[static_cast<int>(rf[1])];
                    const std::vector<double>& envelope_phase =
                        shapes[static_cast<int>(rf[2])];
                    const int time_shape = static_cast<int>(rf[3]);
                    const size_t count = magnitude.size();

                    std::vector<double> t(count);
                    if (time_shape > 0)
                    {
                        const std::vector<double>& ticks = shapes[time_shape];
                        for (size_t i = 0; i < count && i < ticks.size(); ++i)
                            t[i] = ticks[i] * rf_raster;
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                            t[i] = (static_cast<double>(i) + 0.5) * rf_raster;
                    }

                    std::vector<std::complex<double>> signal(count);
                    for (size_t i = 0; i < count; ++i)
                    {
                        const double turns = 2.0 * kPi *
                            (i < envelope_phase.size() ? envelope_phase[i] : 0.0);
                        const std::complex<double> envelope =
                            std::polar(rf[0] * magnitude[i], turns);
                        const double carrier =
                            phase + 2.0 * kPi * frequency * t[i];
                        signal[i] = envelope * std::polar(1.0, carrier);
                    }

                    const double start = elapsed + rf[5];
                    if (count != 0 && std::abs(signal.front()) > 0.0)
                    {
                        out.rf_times.push_back(start + t.front() - kEps);
                        out.rf_signal.push_back(0.0);
                    }
                    for (size_t i = 0; i < count; ++i)
                    {
                        out.rf_times.push_back(start + t[i]);
                        out.rf_signal.push_back(signal[i]);
                    }
                    if (count != 0 && std::abs(signal.back()) > 0.0)
                    {
                        out.rf_times.push_back(start + t.back() + kEps);
                        out.rf_signal.push_back(0.0);
                    }
                }
            }

            if (row[4] > 0)
            {
                const double* adc = seq.adc_library().row(row[4]);
                const int samples = static_cast<int>(std::llround(adc[0]));
                const double dwell = adc[1];
                const double start = elapsed + adc[2];
                const double frequency = adc[5] + adc[3] * 1e-6 * larmor;
                const double phase = adc[6] + adc[4] * 1e-6 * larmor;
                const std::vector<double>& modulation = shapes[static_cast<int>(adc[7])];

                out.window_frequency.push_back(adc[5]);
                out.window_phase.push_back(adc[6]);
                out.window_blocks.push_back(index);
                out.window_samples.push_back(samples);

                for (int i = 0; i < samples; ++i)
                {
                    const double within = dwell * (static_cast<double>(i) + 0.5);
                    const double shift = static_cast<size_t>(i) < modulation.size()
                        ? modulation[static_cast<size_t>(i)]
                        : 0.0;
                    out.adc_times.push_back(start + within);
                    out.adc_frequency.push_back(frequency);
                    out.adc_phase.push_back(phase + shift + frequency * within);
                    out.adc_modulation.push_back(shift);
                }
            }

            elapsed += durations[index - 1];
        }
        out.duration = elapsed;

        for (int axis = 0; axis < 3; ++axis)
        {
            const std::vector<double>& t = out.times[static_cast<size_t>(axis)];
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (t[i] <= t[i - 1])
                {
                    out.warnings.push_back(
                        "Warning: not all elements of the generated time vector are "
                        "unique and sorted in accending order!");
                    break;
                }
            }
        }

        return out;
    }

} // namespace pulseq
