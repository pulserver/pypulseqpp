/**
 * @file rf_safety.cpp
 * @brief RF power and VOP SAR.  See rf_safety.hpp.
 */

#include "pulseq/rf_safety.hpp"

#include "pulseq/channels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace pulseq
{

    namespace
    {

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /** A pulse's unit-amplitude envelope per channel, on a dt grid. */
        struct Resampled
        {
            size_t channels = 1;
            size_t samples = 0;
            /** Channel-major: values[c * samples + n]. */
            std::vector<std::complex<double>> values;
        };

        using ShapeKey = std::array<int, 3>;

        ShapeKey shapes_of(const double* rf)
        {
            return {static_cast<int>(rf[1]), static_cast<int>(rf[2]), static_cast<int>(rf[3])};
        }

        /**
         * Resample one RF row's envelope at amplitude 1.
         *
         * The shape duration is the core's: the last time-shape sample rounded
         * up to the RF raster, or the sample count times the raster.
         */
        Resampled resample(ShapeCache& shapes, const ShapeKey& key, double raster, double dt)
        {
            const std::vector<double>& magnitude = shapes[key[0]];
            const std::vector<double>& phase = shapes[key[1]];
            const size_t count = magnitude.size();

            std::vector<double> t(count);
            double shape_dur = 0.0;
            if (key[2] > 0)
            {
                const std::vector<double>& ticks = shapes[key[2]];
                for (size_t i = 0; i < count && i < ticks.size(); ++i)
                    t[i] = ticks[i] * raster;
                const double last = count ? t[count - 1] : 0.0;
                shape_dur = std::ceil(last / raster - 1e-6) * raster;
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    t[i] = (static_cast<double>(i) + 0.5) * raster;
                shape_dur = static_cast<double>(count) * raster;
            }

            Resampled out;
            out.channels = count ? rf_channels(t) : 1;
            out.samples = static_cast<size_t>(std::max(0.0, std::round(shape_dur / dt)));
            out.values.assign(out.channels * out.samples, std::complex<double>(0.0, 0.0));
            if (count == 0)
                return out;

            const size_t per_channel = count / out.channels;
            for (size_t c = 0; c < out.channels; ++c)
            {
                const size_t base = c * per_channel;
                std::complex<double>* into = out.values.data() + c * out.samples;
                size_t at = 0;
                for (size_t n = 0; n < out.samples; ++n)
                {
                    const double time = (static_cast<double>(n) + 0.5) * dt;
                    if (time < t[base] || time > t[base + per_channel - 1])
                        continue;
                    while (at + 1 < per_channel && t[base + at + 1] < time)
                        ++at;
                    const auto sample = [&](size_t i) {
                        const size_t k = base + i;
                        return std::polar(magnitude[k], kTwoPi * (k < phase.size() ? phase[k] : 0.0));
                    };
                    if (at + 1 >= per_channel)
                    {
                        into[n] = sample(per_channel - 1);
                        continue;
                    }
                    const double span = t[base + at + 1] - t[base + at];
                    const double f = span > 0.0 ? (time - t[base + at]) / span : 0.0;
                    into[n] = sample(at) * (1.0 - f) + sample(at + 1) * f;
                }
            }
            return out;
        }

        /** Energy and peak power of a unit-amplitude pulse, over its channels. */
        struct UnitPower
        {
            double energy = 0.0;
            double peak = 0.0;
        };

        UnitPower unit_power(const Resampled& pulse, double dt)
        {
            UnitPower out;
            for (size_t n = 0; n < pulse.samples; ++n)
            {
                double power = 0.0;
                for (size_t c = 0; c < pulse.channels; ++c)
                    power += std::norm(pulse.values[c * pulse.samples + n]);
                out.energy += power * dt;
                out.peak = std::max(out.peak, power);
            }
            return out;
        }

    } // namespace

    RfPower rf_power(const Sequence& seq, int first, int last, double window, double dt)
    {
        RfPower out;
        first = std::max(first, 1);
        last = std::min(last, seq.num_blocks());
        if (last < first)
            return out;

        ShapeCache shapes(seq.shape_library());
        std::map<ShapeKey, UnitPower> known;
        const double raster = seq.rf_raster_time();
        const int32_t* events = seq.block_events();
        const double* durations = seq.block_durations();
        const bool windowed = window > 0.0;

        double duration = 0.0;
        double energy = 0.0;
        double mean_square = 0.0;
        double energy_max = 0.0;
        double mean_square_max = 0.0;
        std::vector<double> kept(windowed ? static_cast<size_t>(last - first + 1) : 0, 0.0);
        double window_duration = 0.0;
        int window_start = first;

        for (int block = first; block <= last; ++block)
        {
            const int32_t* row = events + static_cast<size_t>(block - 1) * BLOCK_WIDTH;
            duration += durations[block - 1];
            if (row[0] > 0)
            {
                const double* rf = seq.rf_library().row(row[0]);
                const ShapeKey key = shapes_of(rf);
                auto found = known.find(key);
                if (found == known.end())
                    found = known.emplace(key, unit_power(resample(shapes, key, raster, dt), dt))
                                .first;
                const double scale = rf[0] * rf[0];
                const double e = scale * found->second.energy;
                energy += e;
                // MATLAB adds rms^2 * shape_dur, which is the energy again.
                mean_square += e;
                out.peak_power = std::max(out.peak_power, scale * found->second.peak);
                if (windowed)
                {
                    kept[static_cast<size_t>(block - first)] = e;
                    energy_max = std::max(energy_max, energy);
                    mean_square_max = std::max(mean_square_max, mean_square);
                }
            }
            if (windowed)
            {
                window_duration += durations[block - 1];
                while (window_duration > window)
                {
                    energy -= kept[static_cast<size_t>(window_start - first)];
                    mean_square -= kept[static_cast<size_t>(window_start - first)];
                    window_duration -= durations[window_start - 1];
                    ++window_start;
                }
            }
        }

        if (windowed)
        {
            out.energy = energy_max;
            out.mean_power = energy_max / window;
            out.rms = std::sqrt(mean_square_max / window);
        }
        else
        {
            out.energy = energy;
            out.mean_power = duration > 0.0 ? energy / duration : 0.0;
            out.rms = duration > 0.0 ? std::sqrt(mean_square / duration) : 0.0;
        }
        return out;
    }

    namespace
    {

        /** Re(tr(Q M)) for row-major Nc x Nc matrices. */
        double trace_product(const std::complex<double>* q, const std::vector<std::complex<double>>& m, size_t nc)
        {
            double sum = 0.0;
            for (size_t i = 0; i < nc; ++i)
                for (size_t j = 0; j < nc; ++j)
                    sum += (q[i * nc + j] * m[j * nc + i]).real();
            return sum;
        }

        /** Per-VOP and global energy of one pulse shape through one shim, at unit amplitude. */
        struct Deposit
        {
            std::vector<double> local;
            double global = 0.0;
        };

        Deposit deposit(
            const Resampled& pulse,
            const std::vector<std::complex<double>>& shim,
            const SarModel& model)
        {
            const size_t nc = static_cast<size_t>(model.channels);
            if (pulse.channels != 1 && pulse.channels != nc)
            {
                throw std::invalid_argument(
                    "a pulse plays " + std::to_string(pulse.channels) + " channels, and the VOPs describe " +
                    std::to_string(nc));
            }
            if (shim.size() != nc)
            {
                throw std::invalid_argument(
                    "an RF shim weighs " + std::to_string(shim.size()) + " channels, and the VOPs describe " +
                    std::to_string(nc));
            }

            /* M = sum over time of v v^H dt, with v_c = drive_c * s_c * b_c. */
            std::vector<std::complex<double>> m(nc * nc, std::complex<double>(0.0, 0.0));
            std::vector<std::complex<double>> v(nc);
            for (size_t n = 0; n < pulse.samples; ++n)
            {
                for (size_t c = 0; c < nc; ++c)
                {
                    const size_t from = pulse.channels == 1 ? 0 : c;
                    v[c] = model.drive[c] * shim[c] * pulse.values[from * pulse.samples + n];
                }
                for (size_t i = 0; i < nc; ++i)
                    for (size_t j = 0; j < nc; ++j)
                        m[i * nc + j] += v[i] * std::conj(v[j]) * model.dt;
            }

            Deposit out;
            const size_t count = model.vops.size() / (nc * nc);
            out.local.resize(count);
            for (size_t k = 0; k < count; ++k)
                out.local[k] = trace_product(model.vops.data() + k * nc * nc, m, nc);
            if (!model.global.empty())
                out.global = trace_product(model.global.data(), m, nc);
            return out;
        }

    } // namespace

    SarReport vop_sar(const Sequence& seq, const SarModel& model, int size, int start)
    {
        const size_t nc = static_cast<size_t>(model.channels);
        const size_t count = nc ? model.vops.size() / (nc * nc) : 0;
        const int blocks = seq.num_blocks();

        std::vector<std::pair<int, int>> bounds;
        if (size <= 0 || start < 1)
        {
            if (blocks > 0)
                bounds.emplace_back(1, blocks);
        }
        else
        {
            if (start > 1)
                bounds.emplace_back(1, std::min(start - 1, blocks));
            int from = start;
            for (; from + size - 1 <= blocks; from += size)
                bounds.emplace_back(from, from + size - 1);
            if (from <= blocks)
                bounds.emplace_back(from, blocks);
        }

        ShapeCache shapes(seq.shape_library());
        const double raster = seq.rf_raster_time();
        const int32_t* events = seq.block_events();
        const double* durations = seq.block_durations();
        const std::vector<std::complex<double>> ones(nc, std::complex<double>(1.0, 0.0));

        /* One deposit per pulse shape and shim, whatever amplitude plays them. */
        std::map<std::pair<ShapeKey, int>, Deposit> known;
        std::map<ShapeKey, Resampled> resampled;

        SarReport out;
        out.windows.reserve(bounds.size());
        double worst = -1.0;
        std::vector<double> energy(count);
        for (const auto& bound : bounds)
        {
            SarWindow window;
            window.first = bound.first;
            window.last = bound.second;
            std::fill(energy.begin(), energy.end(), 0.0);
            double global = 0.0;

            for (int block = bound.first; block <= bound.second; ++block)
            {
                const int32_t* row = events + static_cast<size_t>(block - 1) * BLOCK_WIDTH;
                window.duration += durations[block - 1];
                if (row[0] <= 0)
                    continue;
                const double* rf = seq.rf_library().row(row[0]);
                const ShapeKey key = shapes_of(rf);
                const int shim_id = row[BLOCK_SHIM_COLUMN];
                auto found = known.find({key, shim_id});
                if (found == known.end())
                {
                    auto pulse = resampled.find(key);
                    if (pulse == resampled.end())
                        pulse = resampled.emplace(key, resample(shapes, key, raster, model.dt)).first;

                    std::vector<std::complex<double>> shim;
                    if (shim_id >= 1 && shim_id <= seq.rf_shim_library().size())
                    {
                        const double* values = seq.rf_shim_library().row(shim_id);
                        const int length = seq.rf_shim_library().length(shim_id);
                        for (int c = 0; c + 1 < length; c += 2)
                            shim.push_back(std::polar(values[c], values[c + 1]));
                    }
                    else
                    {
                        shim = pulse->second.channels == 1 ? model.default_shim : ones;
                    }
                    found = known.emplace(std::make_pair(key, shim_id), deposit(pulse->second, shim, model))
                                .first;
                }
                const double scale = rf[0] * rf[0];
                const std::vector<double>& local = found->second.local;
                for (size_t k = 0; k < count; ++k)
                    energy[k] += scale * local[k];
                global += scale * found->second.global;
            }

            if (window.duration > 0.0)
            {
                for (size_t k = 0; k < count; ++k)
                {
                    const double sar = energy[k] / window.duration;
                    if (window.vop < 0 || sar > window.local)
                    {
                        window.local = sar;
                        window.vop = static_cast<int>(k);
                    }
                    if (!model.reference.empty())
                    {
                        const double against = model.reference[k];
                        const double ratio = against > 0.0
                            ? sar / against
                            : (sar > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
                        if (window.ratio_vop < 0 || ratio > window.ratio)
                        {
                            window.ratio = ratio;
                            window.ratio_vop = static_cast<int>(k);
                        }
                    }
                }
                window.global = global / window.duration;
                if (window.local > worst)
                {
                    worst = window.local;
                    out.worst.resize(count);
                    for (size_t k = 0; k < count; ++k)
                        out.worst[k] = energy[k] / window.duration;
                }
            }
            out.windows.push_back(window);
        }
        return out;
    }

} // namespace pulseq
