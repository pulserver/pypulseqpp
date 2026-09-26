/**
 * @file simulate.cpp
 * @brief A sequence's blocks played on isochromats.
 */

#include "pulseq/simulate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

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

        /** The gradients one block plays after its rotation, per axis, timed
         *  from the block's start. */
        struct BlockGradients
        {
            std::vector<double> times[3];
            std::vector<double> values[3];
            std::vector<double> input_times[3];
            std::vector<double> input_values[3];
            bool present[3] = {false, false, false};
        };

        /** The corners @p shape plays, timed from its block's start; false
         *  where it plays nothing. */
        bool played_corners(const Corners& shape, std::vector<double>& times, std::vector<double>& values)
        {
            times.clear();
            values.clear();
            if (shape.empty_with_amplitude || shape.values.empty())
                return false;
            values = shape.values;
            times.resize(values.size());
            double when = shape.delay;
            for (size_t i = 0; i < values.size(); ++i)
            {
                if (shape.trapezoid)
                    when += i > 0 ? shape.ramps[i - 1] : 0.0;
                times[i] = shape.trapezoid ? when : shape.times[i] + shape.delay;
            }
            return true;
        }

        /** The sum of the three input axes each output axis plays, on the
         *  union of their corners, where the sum is exact. */
        void rotate(const double matrix[3][3], BlockGradients& out)
        {
            std::vector<double> merged;
            for (int axis = 0; axis < 3; ++axis)
                merged.insert(merged.end(), out.input_times[axis].begin(), out.input_times[axis].end());
            std::sort(merged.begin(), merged.end());
            merged.erase(
                std::unique(merged.begin(), merged.end(), [](double a, double b) { return std::fabs(a - b) <= kEps; }),
                merged.end());
            for (int into = 0; into < 3; ++into)
            {
                out.times[into].clear();
                out.values[into].clear();
                for (int from = 0; from < 3; ++from)
                {
                    const double weight = matrix[into][from];
                    if (!out.present[from] || weight == 0.0)
                        continue;
                    out.times[into] = merged;
                    out.values[into].resize(merged.size(), 0.0);
                    for (size_t i = 0; i < merged.size(); ++i)
                        out.values[into][i] +=
                            weight * sampled(out.input_times[from], out.input_values[from], merged[i]);
                }
            }
        }

        void gradients_of(const Sequence& seq, const int32_t* row, CornerCache& corners, BlockGradients& out)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                out.input_times[axis].clear();
                out.input_values[axis].clear();
                out.present[axis] =
                    id > 0 && played_corners(corners[id], out.input_times[axis], out.input_values[axis]);
            }
            const int32_t rotation_row = row[BLOCK_ROTATION_COLUMN];
            if (rotation_row >= 1 && rotation_row <= seq.rotation_library().size())
            {
                double matrix[3][3];
                rotation_matrix(seq.rotation_library().row(rotation_row), matrix);
                rotate(matrix, out);
                return;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                out.times[axis].swap(out.input_times[axis]);
                out.values[axis].swap(out.input_values[axis]);
            }
        }

        /** An RF pulse's samples: times from the pulse's start and complex
         *  envelope, channel after channel over one time base. */
        struct Samples
        {
            std::vector<double> times;
            std::vector<std::complex<double>> envelope;
            size_t channels = 1;
            size_t per = 0;
            /** Whether the samples are the middles of equal intervals from the
             *  pulse's start, and so steps as they stand. */
            bool on_steps = true;
        };

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

        void samples_of(const double* rf, ShapeCache& shapes, double raster, Samples& out)
        {
            const std::vector<double>& magnitude = shapes[static_cast<int>(rf[1])];
            const std::vector<double>& phase = shapes[static_cast<int>(rf[2])];
            const int time_shape = static_cast<int>(rf[3]);
            const std::vector<double>& ticks = shapes[time_shape > 0 ? time_shape : 0];
            const size_t count = magnitude.size();
            out.times.resize(count);
            out.envelope.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                out.times[i] =
                    time_shape > 0 ? (i < ticks.size() ? ticks[i] : 0.0) * raster : (static_cast<double>(i) + 0.5) * raster;
                out.envelope[i] = std::polar(rf[0] * magnitude[i], 2.0 * kPi * (i < phase.size() ? phase[i] : 0.0));
            }
            out.channels = time_shape > 0 ? rf_channels(out.times) : 1;
            out.per = count / out.channels;
            out.on_steps = time_shape <= 0 || at_middles(out.times, out.per);
        }

        /** The carrier Pulseq modulates a pulse with: its phase offset,
         *  advancing at its frequency offset from the pulse's start. */
        struct Carrier
        {
            double phase = 0.0;
            double frequency = 0.0;

            /** The pulse's field: the conjugate of Pulseq's waveform. */
            std::complex<double> field(std::complex<double> envelope, double time) const
            {
                return std::conj(envelope * std::polar(1.0, phase + 2.0 * kPi * frequency * time));
            }
        };

        /** One block's RF pulse, as the steps Isochromats::play takes. */
        struct Pulse
        {
            double start = 0.0;
            double step = 0.0;
            size_t steps = 0;
            size_t channels = 0;
            std::vector<std::complex<double>> rf;
        };

        void held_steps(const Samples& samples, const Carrier& carrier, double delay, Pulse& out)
        {
            const std::vector<double>& t = samples.times;
            const double interval = samples.per > 1 ? t[1] - t[0] : 2.0 * t[0];
            out.start = delay + t[0] - 0.5 * interval;
            out.step = interval;
            out.steps = samples.per;
            out.rf.resize(samples.channels * samples.per);
            for (size_t c = 0; c < samples.channels; ++c)
                for (size_t i = 0; i < samples.per; ++i)
                    out.rf[c * samples.per + i] = carrier.field(samples.envelope[c * samples.per + i], t[i]);
        }

        /** Channel @p c's envelope at @p when, joined linearly between its
         *  samples; @p j is the sample at or before the previous time asked. */
        std::complex<double> joined(const Samples& samples, size_t c, size_t& j, double when)
        {
            const std::vector<double>& t = samples.times;
            const std::complex<double>* envelope = &samples.envelope[c * samples.per];
            while (j + 1 < samples.per && t[j + 1] < when)
                ++j;
            if (j + 1 >= samples.per || !(t[j + 1] > t[j]))
                return envelope[j];
            return envelope[j] + (when - t[j]) / (t[j + 1] - t[j]) * (envelope[j + 1] - envelope[j]);
        }

        /** Steps of the RF raster, or of the pulse's shortest interval where
         *  that is shorter, over which the joined samples are held. */
        void joined_steps(const Samples& samples, const Carrier& carrier, double delay, double raster, Pulse& out)
        {
            const std::vector<double>& t = samples.times;
            double shortest = raster;
            for (size_t i = 1; i < samples.per; ++i)
                shortest = t[i] > t[i - 1] ? std::min(shortest, t[i] - t[i - 1]) : shortest;
            const double span = samples.per > 0 ? t[samples.per - 1] - t[0] : 0.0;
            const size_t steps = span > 0.0 ? static_cast<size_t>(std::max(1.0, std::ceil(span / shortest - 1e-9))) : 1;
            out.step = span > 0.0 ? span / static_cast<double>(steps) : raster;
            out.start = delay + (span > 0.0 ? t[0] : t[0] - 0.5 * raster);
            out.steps = steps;
            out.rf.resize(samples.channels * steps);
            for (size_t c = 0; c < samples.channels; ++c)
            {
                size_t j = 0;
                for (size_t s = 0; s < steps; ++s)
                {
                    const double when = span > 0.0 ? t[0] + (static_cast<double>(s) + 0.5) * out.step : t[0];
                    out.rf[c * steps + s] = carrier.field(joined(samples, c, j, when), when);
                }
            }
        }

        /** Play a single-channel pulse on each of @p transmit_channels,
         *  weighted by the RF shim @p shim_id where it has one weight per
         *  channel, and at unit weight where there is none. */
        void shim(const Sequence& seq, int32_t shim_id, size_t transmit_channels, Pulse& pulse)
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

        void pulse_of(
            const Sequence& seq,
            const int32_t* row,
            ShapeCache& shapes,
            double larmor,
            size_t transmit_channels,
            Samples& samples,
            Pulse& out)
        {
            const double* rf = seq.rf_library().row(row[0]);
            samples_of(rf, shapes, seq.rf_raster_time(), samples);
            out.steps = 0;
            out.channels = samples.channels;
            if (samples.per == 0)
                return;
            const Carrier carrier{rf[9] + rf[7] * 1e-6 * larmor, rf[8] + rf[6] * 1e-6 * larmor};
            if (samples.on_steps)
                held_steps(samples, carrier, rf[5], out);
            else
                joined_steps(samples, carrier, rf[5], seq.rf_raster_time(), out);
            shim(seq, row[BLOCK_SHIM_COLUMN], transmit_channels, out);
        }

        /** An ADC window's sample times from the block's start, and the phase
         *  each is demodulated by. */
        struct Window
        {
            std::vector<double> times;
            std::vector<double> receiver;
        };

        void window_of(const double* adc, ShapeCache& shapes, double larmor, Window& out)
        {
            const size_t samples = static_cast<size_t>(std::llround(adc[0]));
            const double dwell = adc[1];
            const double frequency = adc[5] + adc[3] * 1e-6 * larmor;
            const double phase = adc[6] + adc[4] * 1e-6 * larmor;
            const std::vector<double>& modulation = shapes[static_cast<int>(adc[7])];
            out.times.resize(samples);
            out.receiver.resize(samples);
            for (size_t k = 0; k < samples; ++k)
            {
                const double within = dwell * (static_cast<double>(k) + 0.5);
                out.times[k] = adc[2] + within;
                out.receiver[k] =
                    phase + (k < modulation.size() ? modulation[k] : 0.0) + 2.0 * kPi * frequency * within;
            }
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

    } // namespace

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
        BlockGradients gradients;
        Samples samples;
        Pulse pulse;
        Window window;
        std::vector<std::complex<double>> signal;
        size_t offset = 0;
        for (int index = first; index <= last; ++index)
        {
            const int32_t* row = seq.block_events() + static_cast<size_t>(index - 1) * BLOCK_WIDTH;
            BlockEvents block;
            block.duration = seq.block_durations()[index - 1];

            gradients_of(seq, row, corners, gradients);
            for (int axis = 0; axis < 3; ++axis)
            {
                block.gradient_times[axis] = gradients.times[axis].data();
                block.gradient_values[axis] = gradients.values[axis].data();
                block.gradient_corners[axis] = gradients.times[axis].size();
            }

            pulse.steps = 0;
            if (row[0] > 0)
                pulse_of(seq, row, shapes, larmor, isochromats.transmit_channels(), samples, pulse);
            block.rf_start = pulse.start;
            block.rf_step = pulse.step;
            block.rf_steps = pulse.steps;
            block.rf_channels = pulse.channels;
            block.rf = pulse.rf.data();

            window.times.clear();
            window.receiver.clear();
            if (row[4] > 0)
                window_of(seq.adc_library().row(row[4]), shapes, larmor, window);
            block.adc_times = window.times.data();
            block.adc_samples = window.times.size();

            signal.assign(coils * block.adc_samples, 0.0);
            isochromats.play(block, signal.data());
            for (size_t coil = 0; coil < coils; ++coil)
                for (size_t k = 0; k < block.adc_samples; ++k)
                    out[coil * total + offset + k] =
                        signal[coil * block.adc_samples + k] * std::polar(1.0, window.receiver[k]);
            offset += block.adc_samples;
        }
        return out;
    }

} // namespace pulseq
