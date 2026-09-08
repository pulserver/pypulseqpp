/**
 * @file waveforms.cpp
 * @brief Expanding a block table into what it plays.  See waveforms.hpp.
 */

#include "pulseq/waveforms.hpp"

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

        /**
         * Every shape decompressed at most once.
         *
         * A readout played a hundred thousand times names one shape, and
         * decompressing it per block is the whole cost of the pass.
         */
        class Shapes
        {
        public:
            explicit Shapes(const ShapeLibrary& library)
                : library_(library), cached_(static_cast<size_t>(library.size()) + 1)
            {
            }

            const std::vector<double>& operator[](int id)
            {
                if (id < 1 || id > library_.size())
                    return empty_;
                std::vector<double>& held = cached_[static_cast<size_t>(id)];
                if (held.empty())
                    held = decompress_shape(
                        library_.samples(id),
                        library_.num_compressed(id),
                        library_.num_uncompressed(id));
                return held;
            }

        private:
            const ShapeLibrary& library_;
            std::vector<std::vector<double>> cached_;
            std::vector<double> empty_;
        };

        /**
         * One gradient's corners, relative to the start of its own delay.
         *
         * A gradient plays the same shape every time it is played -- only
         * where it starts moves -- so the corners are worked out once per
         * gradient and offset per block. For a readout repeated a hundred
         * thousand times that turns the whole of `restore_shape_corners` into
         * one pass instead of a hundred thousand.
         */
        struct Corners
        {
            std::vector<double> times;
            std::vector<double> values;
            double delay = 0.0;
            bool known = false;
            bool empty_with_amplitude = false;

            /**
             * A trapezoid's ramps, kept so its corners can be added up from
             * where the block starts rather than offset from zero.
             *
             * `(start + rise) + flat` and `(rise + flat) + start` are not the
             * same double, and a corner one bit out lands on the other side
             * of `floor(t / raster)` -- which changes which raster points a
             * trajectory is followed through. Four additions is nothing; the
             * shape of a long waveform is what the cache is really for.
             */
            bool trapezoid = false;
            double ramps[3] = {0.0, 0.0, 0.0};
            double amplitude = 0.0;
        };

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

        /** A rotation matrix from a quaternion stored as w, x, y, z. */
        void rotation_matrix(const double* q, double into[3][3])
        {
            const double w = q[0];
            const double x = q[1];
            const double y = q[2];
            const double z = q[3];
            into[0][0] = 1.0 - 2.0 * (y * y + z * z);
            into[0][1] = 2.0 * (x * y - w * z);
            into[0][2] = 2.0 * (x * z + w * y);
            into[1][0] = 2.0 * (x * y + w * z);
            into[1][1] = 1.0 - 2.0 * (x * x + z * z);
            into[1][2] = 2.0 * (y * z - w * x);
            into[2][0] = 2.0 * (x * z - w * y);
            into[2][1] = 2.0 * (y * z + w * x);
            into[2][2] = 1.0 - 2.0 * (x * x + y * y);
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

    Waveforms waveforms_and_times(const Sequence& seq, const WaveformOptions& options)
    {
        Waveforms out;

        const double grad_raster = seq.grad_raster_time();
        const double rf_raster = seq.rf_raster_time();
        const double larmor = options.gamma * options.b0;

        Shapes shapes(seq.shape_library());

        /** Every gradient's corners, worked out the first time it is played. */
        std::vector<Corners> cached(static_cast<size_t>(seq.num_gradients()) + 1);
        const auto corners_of = [&](int32_t id) -> const Corners& {
            Corners& made = cached[static_cast<size_t>(id)];
            if (made.known)
                return made;
            made.known = true;

            if (seq.grad_kind(id) == GradKind::Trap)
            {
                const double* trap = seq.trap_library().row(seq.grad_row(id));
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

            const double* arb = seq.arb_library().row(seq.grad_row(id));
            const double amplitude = arb[0];
            const int shape = static_cast<int>(arb[3]);
            const int time_shape = static_cast<int>(arb[4]);
            made.delay = arb[5];

            const std::vector<double>& normalised = shapes[shape];
            std::vector<double> waveform(normalised.size());
            for (size_t i = 0; i < normalised.size(); ++i)
                waveform[i] = amplitude * normalised[i];

            if (time_shape == 0)
            {
                /* Stored at the centre of each raster interval: the corners
                 * in between have to be put back. */
                restore_shape_corners(
                    waveform, arb[1], arb[2], grad_raster, made.times, made.values);
                return made;
            }

            const std::vector<double>& ticks = shapes[time_shape];
            std::vector<double> tt(ticks.size());
            for (size_t i = 0; i < ticks.size(); ++i)
                tt[i] = ticks[i] * grad_raster;

            /* Times of its own, but starting half a raster in: the shape says
             * nothing about the edges, so the recorded first and last close it. */
            const bool starts_at_a_centre =
                !tt.empty() && std::fabs(tt[0] / grad_raster - 0.5) < 1e-6;
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
        };

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

        const int rotation_type = seq.find_extension_type_id("ROTATIONS");
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
             * different waveform on each, not this one moved. */
            int32_t rotation_row = 0;
            if (rotation_type > 0 && row[5] > 0)
            {
                int32_t node = row[5];
                const IntTable& links = seq.extensions_library();
                while (node > 0 && node <= links.size())
                {
                    const int32_t* link = links.row(node);
                    if (link[0] == rotation_type)
                        rotation_row = link[1];
                    node = link[2];
                }
            }

            const Corners* played[3] = {nullptr, nullptr, nullptr};
            Corners here[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0)
                    continue;
                const Corners& shape = corners_of(id);
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
                if (use == 'e' || use == 'u')
                    out.excitation.push_back(moment);
                else if (use == 'r')
                    out.refocusing.push_back(moment);

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
