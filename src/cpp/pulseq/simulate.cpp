/**
 * @file simulate.cpp
 * @brief A sequence's blocks played on isochromats, and the events of one
 *        block as the Bloch equation takes them.
 */

#include "pulseq/simulate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "pulseq/channels.hpp"
#include "pulseq/corners.hpp"

namespace pulseq
{

    namespace
    {

        constexpr double kPi = 3.14159265358979323846;

        /** A nanosecond: what two corner times have to differ by to be two. */
        constexpr double kEps = 1e-9;

        /** The piecewise-linear waveform through the corners at @p when, zero
         *  outside them; a time within kEps of a corner is that corner. */
        double sampled(const std::vector<double>& times, const std::vector<double>& values, double when)
        {
            const size_t count = times.size();
            if (count == 0 || when < times.front() - kEps || when > times.back() + kEps)
                return 0.0;
            const size_t after =
                static_cast<size_t>(std::lower_bound(times.begin(), times.end(), when - kEps) - times.begin());
            if (after >= count)
                return values.back();
            const double span = after > 0 ? times[after] - times[after - 1] : 0.0;
            if (times[after] <= when + kEps || !(span > 0.0))
                return values[after];
            return values[after - 1] + (when - times[after - 1]) / span * (values[after] - values[after - 1]);
        }

        /** The corners @p shape plays, timed from its block's start; none
         *  where it plays nothing. */
        void played_corners(const Corners& shape, std::vector<double>& times, std::vector<double>& values)
        {
            times.clear();
            values.clear();
            if (shape.empty_with_amplitude || shape.values.empty())
                return;
            values = shape.values;
            times.resize(values.size());
            double when = shape.delay;
            for (size_t i = 0; i < values.size(); ++i)
            {
                if (shape.trapezoid)
                    when += i > 0 ? shape.ramps[i - 1] : 0.0;
                times[i] = shape.trapezoid ? when : shape.times[i] + shape.delay;
            }
        }

        /** The waveform through the corners just before and just after
         *  @p when: a gradient is zero outside its corners, so it steps there
         *  from or to a value other than zero. */
        void limits(const std::vector<double>& times, const std::vector<double>& values, double when, double& before, double& after)
        {
            before = after = sampled(times, values, when);
            if (times.empty())
                return;
            if (std::fabs(when - times.front()) <= kEps)
                before = 0.0;
            if (std::fabs(when - times.back()) <= kEps)
                after = 0.0;
        }

        /** The sum @p row weights the given axes by, just before and just
         *  after @p when. */
        void mix(const GradientCorners& given, const double row[3], double when, double& before, double& after)
        {
            for (int from = 0; from < 3; ++from)
            {
                if (given.times[from].empty() || row[from] == 0.0)
                    continue;
                double left = 0.0;
                double right = 0.0;
                limits(given.times[from], given.values[from], when, left, right);
                before += row[from] * left;
                after += row[from] * right;
            }
        }

        void check_corners(const GradientCorners& given)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const std::vector<double>& times = given.times[axis];
                const std::vector<double>& values = given.values[axis];
                bool valid = times.size() == values.size();
                for (size_t i = 0; valid && i < times.size(); ++i)
                    valid = std::isfinite(times[i]) && std::isfinite(values[i]) && !(i > 0 && times[i] < times[i - 1]);
                if (!valid)
                    throw std::invalid_argument("a gradient's corners must be finite, at times in increasing order");
            }
        }

        /** Every corner time of the given axes, once. */
        std::vector<double> union_of(const GradientCorners& given)
        {
            std::vector<double> merged;
            for (int axis = 0; axis < 3; ++axis)
                merged.insert(merged.end(), given.times[axis].begin(), given.times[axis].end());
            std::sort(merged.begin(), merged.end());
            merged.erase(
                std::unique(merged.begin(), merged.end(), [](double a, double b) { return std::fabs(a - b) <= kEps; }),
                merged.end());
            return merged;
        }

        /** The gradients one block plays after its rotation, per axis, timed
         *  from the block's start. */
        void gradients_of(
            const Sequence& seq, const int32_t* row, CornerCache& corners, GradientCorners& given, GradientCorners& out)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                given.times[axis].clear();
                given.values[axis].clear();
                if (row[1 + axis] > 0)
                    played_corners(corners[row[1 + axis]], given.times[axis], given.values[axis]);
            }
            const int32_t rotation_row = row[BLOCK_ROTATION_COLUMN];
            if (rotation_row >= 1 && rotation_row <= seq.rotation_library().size())
            {
                double matrix[3][3];
                rotation_matrix(seq.rotation_library().row(rotation_row), matrix);
                rotate_gradients(matrix, given, out);
                return;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                out.times[axis].swap(given.times[axis]);
                out.values[axis].swap(given.values[axis]);
            }
        }

        /** An RF pulse's samples as Pulseq defines them: times from the
         *  pulse's start and the complex waveform, in Hz. */
        void waveform_of(
            const double* rf,
            ShapeCache& shapes,
            double raster,
            std::vector<double>& times,
            std::vector<std::complex<double>>& waveform)
        {
            const std::vector<double>& magnitude = shapes[static_cast<int>(rf[1])];
            const std::vector<double>& phase = shapes[static_cast<int>(rf[2])];
            const int time_shape = static_cast<int>(rf[3]);
            const std::vector<double>& ticks = shapes[time_shape > 0 ? time_shape : 0];
            const size_t count = magnitude.size();
            times.resize(count);
            waveform.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                times[i] =
                    time_shape > 0 ? (i < ticks.size() ? ticks[i] : 0.0) * raster : (static_cast<double>(i) + 0.5) * raster;
                waveform[i] = std::polar(rf[0] * magnitude[i], 2.0 * kPi * (i < phase.size() ? phase[i] : 0.0));
            }
        }

        /** Whether the first @p per times are the middles of equal intervals
         *  from the pulse's start, and so steps as they stand. */
        bool at_middles(const std::vector<double>& t, size_t per)
        {
            if (per < 2)
                return false;
            const double interval = t[1] - t[0];
            for (size_t i = 1; i < per; ++i)
                if (std::fabs(t[i] - t[i - 1] - interval) > 1e-9 * interval)
                    return false;
            return std::fabs(t[0] - 0.5 * interval) <= 1e-9 * interval;
        }

        /** The carrier Pulseq modulates a pulse with: its phase offset,
         *  advancing at its frequency offset from the pulse's start. */
        struct Carrier
        {
            double phase = 0.0;
            double frequency = 0.0;

            /** The pulse's field: the conjugate of Pulseq's waveform. */
            std::complex<double> field(std::complex<double> sample, double time) const
            {
                return std::conj(sample * std::polar(1.0, phase + 2.0 * kPi * frequency * time));
            }
        };

        void held_steps(
            const std::vector<double>& t,
            const std::vector<std::complex<double>>& waveform,
            size_t per,
            const Carrier& carrier,
            double delay,
            PulseSteps& out)
        {
            const double interval = t[1] - t[0];
            out.start = delay + t[0] - 0.5 * interval;
            out.step = interval;
            out.steps = per;
            out.rf.resize(out.channels * per);
            for (size_t c = 0; c < out.channels; ++c)
                for (size_t i = 0; i < per; ++i)
                    out.rf[c * per + i] = carrier.field(waveform[c * per + i], t[i]);
        }

        /** A channel's @p waveform at @p when, joined linearly between its
         *  samples; @p j is the sample at or before the previous time asked. */
        std::complex<double> joined(
            const std::vector<double>& t, const std::complex<double>* waveform, size_t per, size_t& j, double when)
        {
            while (j + 1 < per && t[j + 1] < when)
                ++j;
            if (j + 1 >= per || !(t[j + 1] > t[j]))
                return waveform[j];
            return waveform[j] + (when - t[j]) / (t[j + 1] - t[j]) * (waveform[j + 1] - waveform[j]);
        }

        /** Steps of the raster, or of the pulse's shortest interval where that
         *  is shorter, over which the joined samples are held. */
        void joined_steps(
            const std::vector<double>& t,
            const std::vector<std::complex<double>>& waveform,
            size_t per,
            const Carrier& carrier,
            double delay,
            double raster,
            PulseSteps& out)
        {
            double shortest = raster;
            for (size_t i = 1; i < per; ++i)
                shortest = t[i] > t[i - 1] ? std::min(shortest, t[i] - t[i - 1]) : shortest;
            const double span = t[per - 1] - t[0];
            const size_t steps = span > 0.0 ? static_cast<size_t>(std::max(1.0, std::ceil(span / shortest - 1e-9))) : 1;
            out.step = span > 0.0 ? span / static_cast<double>(steps) : raster;
            out.start = delay + (span > 0.0 ? t[0] : t[0] - 0.5 * raster);
            out.steps = steps;
            out.rf.resize(out.channels * steps);
            for (size_t c = 0; c < out.channels; ++c)
            {
                size_t j = 0;
                for (size_t s = 0; s < steps; ++s)
                {
                    const double when = span > 0.0 ? t[0] + (static_cast<double>(s) + 0.5) * out.step : t[0];
                    out.rf[c * steps + s] = carrier.field(joined(t, &waveform[c * per], per, j, when), when);
                }
            }
        }

        void check_pulse_times(const std::vector<double>& times, size_t per)
        {
            for (size_t i = 0; i < times.size(); ++i)
                if (!std::isfinite(times[i]) || (i % per != 0 && times[i] < times[i - 1]))
                    throw std::invalid_argument("an RF pulse's sample times must be finite and in increasing order");
        }

        /** Play a single-channel pulse on each of @p transmit_channels,
         *  weighted by the RF shim @p shim_id where it has one weight per
         *  channel, and at unit weight where there is none. */
        void shim(const Sequence& seq, int32_t shim_id, size_t transmit_channels, PulseSteps& pulse)
        {
            if (transmit_channels == 0 || pulse.channels != 1)
                return;
            const RaggedTable& shims = seq.rf_shim_library();
            const bool weighted = shim_id >= 1 && shim_id <= shims.size() &&
                static_cast<size_t>(shims.length(shim_id)) == 2 * transmit_channels;
            const std::vector<std::complex<double>> played = pulse.rf;
            pulse.channels = transmit_channels;
            pulse.rf.resize(transmit_channels * pulse.steps);
            for (size_t c = 0; c < transmit_channels; ++c)
            {
                const std::complex<double> weight = weighted
                    ? std::conj(std::polar(shims.row(shim_id)[2 * c], shims.row(shim_id)[2 * c + 1]))
                    : std::complex<double>(1.0, 0.0);
                for (size_t s = 0; s < pulse.steps; ++s)
                    pulse.rf[c * pulse.steps + s] = weight * played[s];
            }
        }

        /** Buffers reused from one block to the next. */
        struct Played
        {
            GradientCorners given;
            GradientCorners gradients;
            std::vector<double> rf_times;
            std::vector<std::complex<double>> waveform;
            PulseSteps pulse;
            AdcWindow window;
        };

        void pulse_of(
            const Sequence& seq, const int32_t* row, ShapeCache& shapes, double larmor, size_t transmit_channels, Played& into)
        {
            const double* rf = seq.rf_library().row(row[0]);
            waveform_of(rf, shapes, seq.rf_raster_time(), into.rf_times, into.waveform);
            pulse_steps(
                into.rf_times,
                into.waveform,
                rf[5],
                rf[9] + rf[7] * 1e-6 * larmor,
                rf[8] + rf[6] * 1e-6 * larmor,
                seq.rf_raster_time(),
                into.pulse);
            shim(seq, row[BLOCK_SHIM_COLUMN], transmit_channels, into.pulse);
        }

        void window_of(const double* adc, ShapeCache& shapes, double larmor, AdcWindow& out)
        {
            adc_window(
                static_cast<size_t>(std::llround(adc[0])),
                adc[1],
                adc[2],
                adc[6] + adc[4] * 1e-6 * larmor,
                adc[5] + adc[3] * 1e-6 * larmor,
                shapes[static_cast<int>(adc[7])],
                out);
        }

        size_t samples_in(const Sequence& seq, int first, int last)
        {
            size_t total = 0;
            for (int index = first; index <= last; ++index)
            {
                const int32_t adc = seq.block_events()[static_cast<size_t>(index - 1) * BLOCK_WIDTH + 4];
                total += adc > 0 ? static_cast<size_t>(std::llround(seq.adc_library().row(adc)[0])) : 0;
            }
            return total;
        }

        /** The block's events, pointing into @p played. */
        BlockEvents events_of(double duration, const Played& played, bool pulse)
        {
            BlockEvents block;
            block.duration = duration;
            for (int axis = 0; axis < 3; ++axis)
            {
                block.gradient_times[axis] = played.gradients.times[axis].data();
                block.gradient_values[axis] = played.gradients.values[axis].data();
                block.gradient_corners[axis] = played.gradients.times[axis].size();
            }
            if (pulse)
            {
                block.rf_start = played.pulse.start;
                block.rf_step = played.pulse.step;
                block.rf_steps = played.pulse.steps;
                block.rf_channels = played.pulse.channels;
                block.rf = played.pulse.rf.data();
            }
            block.adc_times = played.window.times.data();
            block.adc_samples = played.window.times.size();
            return block;
        }

    } // namespace

    void rotate_gradients(const double matrix[3][3], const GradientCorners& given, GradientCorners& out)
    {
        check_corners(given);
        const std::vector<double> merged = union_of(given);
        for (int into = 0; into < 3; ++into)
        {
            out.times[into].clear();
            out.values[into].clear();
            const double* row = matrix[into];
            bool anything = false;
            for (int from = 0; from < 3; ++from)
                anything = anything || (!given.times[from].empty() && row[from] != 0.0);
            for (size_t i = 0; anything && i < merged.size(); ++i)
            {
                double before = 0.0;
                double after = 0.0;
                mix(given, row, merged[i], before, after);
                if (before != after)
                {
                    out.times[into].push_back(merged[i]);
                    out.values[into].push_back(before);
                }
                out.times[into].push_back(merged[i]);
                out.values[into].push_back(after);
            }
        }
    }

    void pulse_steps(
        const std::vector<double>& times,
        const std::vector<std::complex<double>>& waveform,
        double delay,
        double phase,
        double frequency,
        double raster,
        PulseSteps& out)
    {
        if (times.size() != waveform.size())
            throw std::invalid_argument("an RF pulse needs one sample time per sample");
        if (!(raster > 0.0) || !std::isfinite(raster))
            throw std::invalid_argument("the RF raster must be positive");
        out.steps = 0;
        out.channels = rf_channels(times);
        out.rf.clear();
        const size_t per = times.size() / out.channels;
        if (per == 0)
            return;
        check_pulse_times(times, per);
        const Carrier carrier{phase, frequency};
        if (at_middles(times, per))
            held_steps(times, waveform, per, carrier, delay, out);
        else
            joined_steps(times, waveform, per, carrier, delay, raster, out);
    }

    void adc_window(
        size_t samples,
        double dwell,
        double delay,
        double phase,
        double frequency,
        const std::vector<double>& modulation,
        AdcWindow& out)
    {
        out.times.resize(samples);
        out.receiver.resize(samples);
        for (size_t k = 0; k < samples; ++k)
        {
            const double within = dwell * (static_cast<double>(k) + 0.5);
            out.times[k] = delay + within;
            out.receiver[k] = phase + (k < modulation.size() ? modulation[k] : 0.0) + 2.0 * kPi * frequency * within;
        }
    }

    std::vector<std::complex<double>> simulate(
        const Sequence& seq, Isochromats& isochromats, const SimulationOptions& options)
    {
        const int blocks = seq.num_blocks();
        const int first = std::max(options.first_block, 1);
        const int last = (options.last_block > 0 && options.last_block < blocks) ? options.last_block : blocks;
        const double larmor = options.gamma * options.b0;
        const size_t total = samples_in(seq, first, last);
        const size_t coils = isochromats.coils();
        std::vector<std::complex<double>> out(coils * total);

        ShapeCache shapes(seq.shape_library());
        CornerCache corners(seq);
        Played played;
        std::vector<std::complex<double>> signal;
        size_t offset = 0;
        for (int index = first; index <= last; ++index)
        {
            const int32_t* row = seq.block_events() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            gradients_of(seq, row, corners, played.given, played.gradients);
            if (row[0] > 0)
                pulse_of(seq, row, shapes, larmor, isochromats.transmit_channels(), played);
            played.window.times.clear();
            played.window.receiver.clear();
            if (row[4] > 0)
                window_of(seq.adc_library().row(row[4]), shapes, larmor, played.window);
            const BlockEvents block = events_of(seq.block_durations()[index - 1], played, row[0] > 0);

            signal.assign(coils * block.adc_samples, 0.0);
            isochromats.play(block, signal.data());
            for (size_t coil = 0; coil < coils; ++coil)
                for (size_t k = 0; k < block.adc_samples; ++k)
                    out[coil * total + offset + k] =
                        signal[coil * block.adc_samples + k] * std::polar(1.0, played.window.receiver[k]);
            offset += block.adc_samples;
        }
        return out;
    }

} // namespace pulseq
