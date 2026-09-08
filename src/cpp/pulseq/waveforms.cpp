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
        void extend(
            Channel& channel,
            const std::vector<double>& times,
            const std::vector<double>& values,
            double raster,
            std::vector<std::string>& warnings)
        {
            if (times.empty())
                return;

            std::vector<double>& into_t = *channel.times;
            std::vector<double>& into_v = *channel.values;

            if (into_t.empty())
            {
                into_t = times;
                into_v = values;
                return;
            }

            std::vector<double> piece_t = times;
            std::vector<double> piece_v = values;

            if (into_t.back() + raster < piece_t.front())
            {
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
                if (piece_v.front() != 0.0)
                {
                    if (std::fabs(piece_v.front()) > 1e-6)
                    {
                        warnings.push_back(
                            "waveforms_and_times(): forcing ramp-up to a non-zero "
                            "gradient sample on axis " +
                            std::to_string(channel.axis) + " at t=" +
                            microseconds(piece_t.front()) +
                            " us \ncheck your sequence, some calculations are probably "
                            "wrong. If using mr.makeArbitraryGrad() consider using "
                            "explicit values for 'first' and 'last' and setting them "
                            "correctly.");
                        piece_t.insert(piece_t.begin(), piece_t.front() - raster / 2.0);
                        piece_v.insert(piece_v.begin(), 0.0);
                    }
                    else
                    {
                        piece_v.front() = 0.0;
                    }
                }
            }

            if (into_t.back() < piece_t.front() - kEps)
            {
                into_t.insert(into_t.end(), piece_t.begin(), piece_t.end());
                into_v.insert(into_v.end(), piece_v.begin(), piece_v.end());
                return;
            }

            /* The pieces meet or overlap: keep only what comes after what is
             * already there. */
            if (piece_t.front() < into_t.back() - kEps)
                warnings.push_back(
                    "Warning: looks like rounding errors for some elements exceed the "
                    "acceptable tolerance!");

            size_t from = 0;
            while (from < piece_t.size() && piece_t[from] <= into_t.back() + kEps)
                ++from;
            if (from < piece_t.size())
            {
                into_t.insert(into_t.end(), piece_t.begin() + static_cast<long>(from), piece_t.end());
                into_v.insert(into_v.end(), piece_v.begin() + static_cast<long>(from), piece_v.end());
            }
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

        double elapsed = 0.0;
        for (int index = first; index <= last; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index - 1) * BLOCK_WIDTH;

            /* A rotation remaps the gradients onto other axes, which is a
             * different waveform on each, not this one moved. */
            bool rotated = false;
            if (rotation_type > 0 && row[5] > 0)
            {
                int32_t node = row[5];
                const IntTable& links = seq.extensions_library();
                while (node > 0 && node <= links.size())
                {
                    const int32_t* link = links.row(node);
                    if (link[0] == rotation_type)
                        rotated = true;
                    node = link[2];
                }
            }
            if (rotated)
                out.rotated_blocks.push_back(index);

            std::vector<double> piece_t;
            std::vector<double> piece_v;

            for (int axis = 0; axis < 3 && !rotated; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0)
                    continue;

                piece_t.clear();
                piece_v.clear();

                if (seq.grad_kind(id) == GradKind::Trap)
                {
                    const double* trap = seq.trap_library().row(seq.grad_row(id));
                    const double amplitude = trap[0];
                    const double rise = trap[1];
                    const double flat = trap[2];
                    const double fall = trap[3];
                    const double start = elapsed + trap[4];

                    if (std::fabs(flat) > kEps)
                    {
                        piece_t = {start, start + rise, start + rise + flat,
                                   start + rise + flat + fall};
                        piece_v = {0.0, amplitude, amplitude, 0.0};
                    }
                    else if (std::fabs(rise) > kEps && std::fabs(fall) > kEps)
                    {
                        piece_t = {start, start + rise, start + rise + fall};
                        piece_v = {0.0, amplitude, 0.0};
                    }
                    else
                    {
                        if (std::fabs(amplitude) > kEps)
                            out.warnings.push_back(
                                "\"empty\" gradient with non-zero magnitude detected in "
                                "block " +
                                std::to_string(index));
                        continue;
                    }
                }
                else
                {
                    const double* arb = seq.arb_library().row(seq.grad_row(id));
                    const double amplitude = arb[0];
                    const int shape = static_cast<int>(arb[3]);
                    const int time_shape = static_cast<int>(arb[4]);
                    const double start = elapsed + arb[5];

                    const std::vector<double>& normalised = shapes[shape];
                    std::vector<double> waveform(normalised.size());
                    for (size_t i = 0; i < normalised.size(); ++i)
                        waveform[i] = amplitude * normalised[i];

                    if (time_shape == 0)
                    {
                        /* Stored at the centre of each raster interval: the
                         * corners in between have to be put back. */
                        restore_shape_corners(
                            waveform, arb[1], arb[2], grad_raster, piece_t, piece_v);
                        for (size_t i = 0; i < piece_t.size(); ++i)
                            piece_t[i] += start;
                    }
                    else
                    {
                        const std::vector<double>& ticks = shapes[time_shape];
                        std::vector<double> tt(ticks.size());
                        for (size_t i = 0; i < ticks.size(); ++i)
                            tt[i] = ticks[i] * grad_raster;

                        /* Times of its own, but starting half a raster in:
                         * the shape says nothing about the edges, so the
                         * recorded first and last values close it. */
                        const bool starts_at_a_centre =
                            !tt.empty() &&
                            std::fabs(tt[0] / grad_raster - 0.5) < 1e-6;
                        if (starts_at_a_centre)
                        {
                            piece_t.push_back(start);
                            piece_v.push_back(arb[1]);
                        }
                        for (size_t i = 0; i < tt.size(); ++i)
                        {
                            piece_t.push_back(start + tt[i]);
                            piece_v.push_back(waveform[i]);
                        }
                        if (starts_at_a_centre)
                        {
                            piece_t.push_back(start + (tt.empty() ? 0.0 : tt.back()));
                            piece_v.push_back(arb[2]);
                        }
                    }
                }

                extend(channels[axis], piece_t, piece_v, grad_raster, out.warnings);
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
