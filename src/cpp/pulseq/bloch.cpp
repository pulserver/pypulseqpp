/**
 * @file bloch.cpp
 * @brief Isochromats whose magnetisation the Bloch equation carries from one
 *        block to the next.
 */

#include "pulseq/bloch.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace pulseq
{

    /** The gradient area from a block's start, per axis, in 1/m. */
    class GradientAreas
    {
    public:
        explicit GradientAreas(const BlockEvents& block)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                times_[axis] = block.gradient_times[axis];
                values_[axis] = block.gradient_values[axis];
                count_[axis] =
                    (times_[axis] != nullptr && values_[axis] != nullptr) ? block.gradient_corners[axis] : 0;
                std::vector<double>& cumulative = cumulative_[axis];
                cumulative.assign(count_[axis], 0.0);
                for (size_t i = 1; i < count_[axis]; ++i)
                    cumulative[i] = cumulative[i - 1] +
                        0.5 * (values_[axis][i] + values_[axis][i - 1]) * (times_[axis][i] - times_[axis][i - 1]);
            }
        }

        void at(double time, double area[3]) const
        {
            for (int axis = 0; axis < 3; ++axis)
                area[axis] = along(axis, time);
        }

    private:
        double along(int axis, double time) const
        {
            const size_t count = count_[axis];
            if (count == 0 || time <= times_[axis][0])
                return 0.0;
            const double* times = times_[axis];
            if (time >= times[count - 1])
                return cumulative_[axis][count - 1];
            const size_t after = static_cast<size_t>(std::upper_bound(times, times + count, time) - times);
            const size_t before = after - 1;
            const double span = times[after] - times[before];
            const double elapsed = time - times[before];
            const double start = values_[axis][before];
            const double slope = span > 0.0 ? (values_[axis][after] - start) / span : 0.0;
            return cumulative_[axis][before] + start * elapsed + 0.5 * slope * elapsed * elapsed;
        }

        const double* times_[3] = {nullptr, nullptr, nullptr};
        const double* values_[3] = {nullptr, nullptr, nullptr};
        size_t count_[3] = {0, 0, 0};
        std::vector<double> cumulative_[3];
    };

    namespace
    {

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /** Positions along the gradient of an RF pulse closer than this, in m,
         *  share one computation of the pulse. */
        constexpr double kQuantum = 1e-12;

        /** Tolerance, in s, on event times against the block's bounds. */
        constexpr double kTimeTolerance = 1e-12;

        /** Isochromats a worker takes at the least. */
        constexpr size_t kChunk = 4096;

        /** Isochromats an ADC window is read over together. */
        constexpr size_t kTile = 256;

        /** Groupings kept for reuse, by the field a pulse plays under. */
        constexpr size_t kGroupings = 4;

        /** On what of an isochromat's position the field during a pulse
         *  depends: nothing, its position along one direction, or any. */
        constexpr int kNoGradient = 0;
        constexpr int kOneDirection = 1;
        constexpr int kAnyDirection = 2;

        /** How many workers parallel() runs for @p count items. */
        size_t workers_for(size_t count, size_t threads, size_t least)
        {
            return std::max<size_t>(1, std::min(threads, (count + least - 1) / least));
        }

        /**
         * Run body(worker, first, last) over [0, count) on up to @p threads
         * threads, each given at least @p least items. The body must not throw.
         */
        void parallel(
            size_t count,
            size_t threads,
            size_t least,
            const std::function<void(size_t, size_t, size_t)>& body)
        {
            if (count == 0)
                return;
            const size_t workers = workers_for(count, threads, least);
            if (workers == 1)
            {
                body(0, 0, count);
                return;
            }
            const size_t chunk = (count + workers - 1) / workers;
            std::vector<std::thread> pool;
            pool.reserve(workers - 1);
            for (size_t worker = 1; worker < workers && worker * chunk < count; ++worker)
                pool.emplace_back(body, worker, worker * chunk, std::min(count, (worker + 1) * chunk));
            body(0, 0, std::min(count, chunk));
            for (std::thread& thread : pool)
                thread.join();
        }

        uint64_t bits_of(double value)
        {
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof bits);
            return bits;
        }

        bool same_bits(const double a[3], const double b[3])
        {
            return std::memcmp(a, b, 3 * sizeof(double)) == 0;
        }

        double rate(double time)
        {
            return std::isinf(time) ? 0.0 : 1.0 / time;
        }

        void fill(std::vector<double>& values, size_t count, double value, const char* name)
        {
            if (values.empty())
                values.assign(count, value);
            else if (values.size() != count)
                throw std::invalid_argument(
                    std::string(name) + " holds " + std::to_string(values.size()) + " values for " +
                    std::to_string(count) + " isochromats");
        }

        void require_finite(const std::vector<double>& values, const char* name)
        {
            for (const double value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument(std::string(name) + " must be finite");
        }

        void require_positive(const std::vector<double>& values)
        {
            for (const double value : values)
                if (!(value > 0.0))
                    throw std::invalid_argument("relaxation times must be positive");
        }

        /** Check @p values holds @p per finite sensitivities per isochromat,
         *  none where @p per is zero. */
        void require_sensitivities(
            const std::vector<std::complex<double>>& values, size_t count, size_t per, const char* name)
        {
            if (values.size() != count * per)
                throw std::invalid_argument(
                    std::string(name) + " must hold one row of sensitivities per isochromat");
            for (const std::complex<double>& value : values)
                if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                    throw std::invalid_argument("sensitivities must be finite");
        }

        /**
         * The rotation of the magnetisation about (wx, wy, wz) by its length,
         * in rad, clockwise seen from its tip: R = cI + (1 - c) n n^T - s [n]x.
         */
        void rotation(double wx, double wy, double wz, double r[3][3])
        {
            const double angle = std::sqrt(wx * wx + wy * wy + wz * wz);
            if (!(angle > 1e-300))
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        r[i][j] = i == j ? 1.0 : 0.0;
                return;
            }
            const double nx = wx / angle;
            const double ny = wy / angle;
            const double nz = wz / angle;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const double half = std::sin(0.5 * angle);
            const double k = 2.0 * half * half;
            r[0][0] = c + k * nx * nx;
            r[0][1] = k * nx * ny + s * nz;
            r[0][2] = k * nx * nz - s * ny;
            r[1][0] = k * ny * nx - s * nz;
            r[1][1] = c + k * ny * ny;
            r[1][2] = k * ny * nz + s * nx;
            r[2][0] = k * nz * nx + s * ny;
            r[2][1] = k * nz * ny - s * nx;
            r[2][2] = c + k * nz * nz;
        }

        /**
         * Compose one step, a rotation between two half steps of relaxation,
         * after the affine map M -> A M + c M0 accumulated so far.
         */
        void compose(const double turn[3][3], const double decay[3], double recovery, double a[3][3], double c[3])
        {
            double step[3][3];
            double shift[3];
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    step[row][col] = decay[row] * turn[row][col] * decay[col];
                shift[row] = decay[row] * turn[row][2] * recovery + (row == 2 ? recovery : 0.0);
            }
            double next[3][3];
            double moved[3];
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    next[row][col] =
                        step[row][0] * a[0][col] + step[row][1] * a[1][col] + step[row][2] * a[2][col];
                moved[row] = step[row][0] * c[0] + step[row][1] * c[1] + step[row][2] * c[2] + shift[row];
            }
            std::memcpy(a, next, sizeof next);
            std::memcpy(c, moved, sizeof moved);
        }

        /** The gradient area over each step of a pulse, and on what of an
         *  isochromat's position the field it sees depends. */
        struct PulseGradient
        {
            int mode = kNoGradient;
            /** For kOneDirection the direction; for kAnyDirection, 1 on each
             *  axis a gradient plays on. */
            double direction[3] = {0.0, 0.0, 0.0};
            /** Per step, the gradient area in 1/m. */
            std::vector<double> delta;
            /** Per step, the mean gradient along the direction, in Hz/m. */
            std::vector<double> along;
        };

        /** Whether every step's area lies along the gradient's direction,
         *  filling the mean gradient along it. */
        bool along_one_direction(PulseGradient& gradient, size_t steps, double dt)
        {
            gradient.along.resize(steps);
            const double* direction = gradient.direction;
            for (size_t i = 0; i < steps; ++i)
            {
                const double* d = &gradient.delta[3 * i];
                const double projection = d[0] * direction[0] + d[1] * direction[1] + d[2] * direction[2];
                double off = 0.0;
                double size = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double rest = d[axis] - projection * direction[axis];
                    off += rest * rest;
                    size += d[axis] * d[axis];
                }
                if (off > 1e-18 * size)
                    return false;
                gradient.along[i] = projection / dt;
            }
            return true;
        }

        /** Fill each step's gradient area, and return the step of the largest. */
        size_t step_areas(const BlockEvents& block, const GradientAreas& areas, PulseGradient& gradient, double& largest)
        {
            gradient.delta.resize(3 * block.rf_steps);
            double previous[3];
            areas.at(block.rf_start, previous);
            largest = 0.0;
            size_t loudest = 0;
            for (size_t i = 0; i < block.rf_steps; ++i)
            {
                double next[3];
                areas.at(block.rf_start + static_cast<double>(i + 1) * block.rf_step, next);
                double norm = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double d = next[axis] - previous[axis];
                    gradient.delta[3 * i + axis] = d;
                    norm += d * d;
                    previous[axis] = next[axis];
                }
                if (norm > largest)
                {
                    largest = norm;
                    loudest = i;
                }
            }
            return loudest;
        }

        PulseGradient pulse_gradient(const BlockEvents& block, const GradientAreas& areas)
        {
            PulseGradient gradient;
            double largest = 0.0;
            const size_t loudest = step_areas(block, areas, gradient, largest);
            if (!(largest > 0.0))
                return gradient;

            const double norm = std::sqrt(largest);
            for (int axis = 0; axis < 3; ++axis)
                gradient.direction[axis] = gradient.delta[3 * loudest + axis] / norm;
            gradient.mode = kOneDirection;
            if (along_one_direction(gradient, block.rf_steps, block.rf_step))
                return gradient;

            gradient.mode = kAnyDirection;
            for (int axis = 0; axis < 3; ++axis)
            {
                gradient.direction[axis] = 0.0;
                for (size_t i = 0; i < block.rf_steps; ++i)
                    if (gradient.delta[3 * i + axis] != 0.0)
                        gradient.direction[axis] = 1.0;
            }
            return gradient;
        }

        /** The field along z isochromat @p r sees over step @p i, in Hz. */
        double step_field(const IsochromatProperties& p, size_t r, const PulseGradient& gradient, size_t i, double dt)
        {
            const double* d = gradient.direction;
            if (gradient.mode == kOneDirection)
                return p.off_resonance[r] + (p.x[r] * d[0] + p.y[r] * d[1] + p.z[r] * d[2]) * gradient.along[i];
            if (gradient.mode == kAnyDirection)
            {
                const double* delta = &gradient.delta[3 * i];
                return p.off_resonance[r] + (p.x[r] * delta[0] + p.y[r] * delta[1] + p.z[r] * delta[2]) / dt;
            }
            return p.off_resonance[r];
        }

        /** Write the affine map a pulse applies to isochromat @p r, A then c,
         *  to @p map. */
        void pulse_map(
            const IsochromatProperties& p,
            size_t r,
            const BlockEvents& block,
            const PulseGradient& gradient,
            const std::vector<std::complex<double>>& summed,
            double* map)
        {
            const size_t steps = block.rf_steps;
            const double dt = block.rf_step;
            const double e1 = std::exp(-0.5 * dt * rate(p.t1[r]));
            const double e2 = std::exp(-0.5 * dt * rate(p.t2[r]));
            const double decay[3] = {e2, e2, e1};
            const std::complex<double>* transmit =
                p.transmit_channels ? &p.transmit[r * p.transmit_channels] : nullptr;

            double a[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
            double c[3] = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < steps; ++i)
            {
                std::complex<double> b1 = transmit ? 0.0 : summed[i];
                for (size_t k = 0; transmit && k < block.rf_channels; ++k)
                    b1 += transmit[k] * block.rf[k * steps + i];
                const double bz = step_field(p, r, gradient, i, dt);
                double turn[3][3];
                rotation(kTwoPi * dt * b1.real(), kTwoPi * dt * b1.imag(), kTwoPi * dt * bz, turn);
                compose(turn, decay, 1.0 - e1, a, c);
            }
            std::memcpy(map, a, 9 * sizeof(double));
            std::memcpy(map + 9, c, 3 * sizeof(double));
        }

        /** The gradient area and time from one ADC sample to the next. */
        struct SampleSteps
        {
            std::vector<double> area;
            std::vector<double> time;
            /** One increment throughout, which makes an isochromat's turn per
             *  sample one complex factor. */
            bool uniform = true;
        };

        SampleSteps sample_steps(const double* times, size_t samples, const GradientAreas& areas)
        {
            SampleSteps steps;
            steps.area.assign(3 * samples, 0.0);
            steps.time.assign(samples, 0.0);
            double previous[3];
            areas.at(times[0], previous);
            for (size_t k = 1; k < samples; ++k)
            {
                double next[3];
                areas.at(times[k], next);
                steps.time[k] = times[k] - times[k - 1];
                for (int axis = 0; axis < 3; ++axis)
                {
                    steps.area[3 * k + axis] = next[axis] - previous[axis];
                    previous[axis] = next[axis];
                    if (k > 1 &&
                        std::fabs(steps.area[3 * k + axis] - steps.area[3 + axis]) >
                            1e-12 * (1.0 + std::fabs(next[axis])))
                        steps.uniform = false;
                }
                if (k > 1 && std::fabs(steps.time[k] - steps.time[1]) > 1e-12 * steps.time[1] + 1e-18)
                    steps.uniform = false;
            }
            return steps;
        }

        /** Receive sensitivities, coil-major: entry c * count + i. */
        struct Receive
        {
            const std::vector<double>& re;
            const std::vector<double>& im;
            size_t coils;
            size_t count;
        };

        /** One worker's reading of an ADC window, a tile of isochromats at a time. */
        class WindowReader
        {
        public:
            WindowReader(const IsochromatProperties& p, const SampleSteps& steps, size_t samples, const Receive& receive)
                : p_(p), steps_(steps), samples_(samples), coils_(receive.coils), count_(receive.count),
                  re_(receive.re.empty() ? nullptr : receive.re.data()), im_(receive.im.empty() ? nullptr : receive.im.data())
            {
            }

            void tile(size_t first, size_t size, double* mx, double* my, double* sum) const
            {
                double zr[kTile], zi[kTile], wr[kTile], wi[kTile];
                std::copy(mx + first, mx + first + size, zr);
                std::copy(my + first, my + first + size, zi);
                if (steps_.uniform && samples_ > 1)
                    for (size_t j = 0; j < size; ++j)
                        turn(first + j, &steps_.area[3], steps_.time[1], wr[j], wi[j]);
                for (size_t k = 0; k < samples_; ++k)
                {
                    if (k > 0)
                    {
                        if (!steps_.uniform)
                            for (size_t j = 0; j < size; ++j)
                                turn(first + j, &steps_.area[3 * k], steps_.time[k], wr[j], wi[j]);
                        rotate(zr, zi, wr, wi, size);
                    }
                    accumulate(zr, zi, first, size, k, sum);
                }
                std::copy(zr, zr + size, mx + first);
                std::copy(zi, zi + size, my + first);
            }

        private:
            static void rotate(double* zr, double* zi, const double* wr, const double* wi, size_t size)
            {
                for (size_t j = 0; j < size; ++j)
                {
                    const double r = zr[j] * wr[j] - zi[j] * wi[j];
                    zi[j] = zr[j] * wi[j] + zi[j] * wr[j];
                    zr[j] = r;
                }
            }

            void turn(size_t i, const double* area, double time, double& re, double& im) const
            {
                const double phase = -kTwoPi *
                    (p_.x[i] * area[0] + p_.y[i] * area[1] + p_.z[i] * area[2] + p_.off_resonance[i] * time);
                const double e2 = std::exp(-time * rate(p_.t2[i]));
                re = e2 * std::cos(phase);
                im = e2 * std::sin(phase);
            }

            void accumulate(const double* zr, const double* zi, size_t first, size_t size, size_t k, double* sum) const
            {
                if (re_ == nullptr)
                {
                    sum[2 * k] += std::accumulate(zr, zr + size, 0.0);
                    sum[2 * k + 1] += std::accumulate(zi, zi + size, 0.0);
                    return;
                }
                for (size_t coil = 0; coil < coils_; ++coil)
                {
                    const double* gr = re_ + coil * count_ + first;
                    const double* gi = im_ + coil * count_ + first;
                    double sr = 0.0;
                    double si = 0.0;
                    for (size_t j = 0; j < size; ++j)
                    {
                        sr += gr[j] * zr[j] - gi[j] * zi[j];
                        si += gr[j] * zi[j] + gi[j] * zr[j];
                    }
                    sum[2 * (coil * samples_ + k)] += sr;
                    sum[2 * (coil * samples_ + k) + 1] += si;
                }
            }

            const IsochromatProperties& p_;
            const SampleSteps& steps_;
            size_t samples_;
            size_t coils_;
            size_t count_;
            const double* re_;
            const double* im_;
        };

        void check_gradients(const BlockEvents& block)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const size_t count = block.gradient_corners[axis];
                const double* times = block.gradient_times[axis];
                const double* values = block.gradient_values[axis];
                if (count > 0 && (times == nullptr || values == nullptr))
                    throw std::invalid_argument("a gradient's corners are missing");
                for (size_t i = 0; i < count; ++i)
                    if (!std::isfinite(times[i]) || !std::isfinite(values[i]) || (i > 0 && times[i] < times[i - 1]))
                        throw std::invalid_argument(
                            "a gradient's corners must be finite, at times in increasing order");
            }
        }

        double pulse_end(const BlockEvents& block)
        {
            return block.rf_start + static_cast<double>(block.rf_steps) * block.rf_step;
        }

        bool within(double time, double duration)
        {
            return std::isfinite(time) && time >= -kTimeTolerance && time <= duration + kTimeTolerance;
        }

        void check_channels(const BlockEvents& block, size_t transmit_channels)
        {
            if (transmit_channels != 0 && block.rf_channels != transmit_channels)
                throw std::invalid_argument(
                    "an RF pulse of " + std::to_string(block.rf_channels) + " channels, and transmit sensitivities of " +
                    std::to_string(transmit_channels));
        }

        void check_pulse(const BlockEvents& block, size_t transmit_channels)
        {
            if (block.rf_steps == 0)
                return;
            if (block.rf == nullptr || block.rf_channels == 0)
                throw std::invalid_argument("an RF pulse needs at least one channel of samples");
            if (!(block.rf_step > 0.0) || !std::isfinite(block.rf_step))
                throw std::invalid_argument("an RF pulse's step must be positive");
            if (!within(block.rf_start, block.duration) || !within(pulse_end(block), block.duration))
                throw std::invalid_argument("an RF pulse must play within its block");
            check_channels(block, transmit_channels);
            const double* values = reinterpret_cast<const double*>(block.rf);
            if (!std::all_of(values, values + 2 * block.rf_steps * block.rf_channels, [](double v) { return std::isfinite(v); }))
                throw std::invalid_argument("an RF pulse's samples must be finite");
        }

        /** Check the ADC samples, and return how many come before the pulse. */
        size_t check_samples(const BlockEvents& block)
        {
            const bool pulse = block.rf_steps > 0;
            const double start = block.rf_start;
            const double end = pulse_end(block);
            size_t before = 0;
            for (size_t k = 0; k < block.adc_samples; ++k)
            {
                const double t = block.adc_times[k];
                if (!within(t, block.duration) || (k > 0 && t < block.adc_times[k - 1]))
                    throw std::invalid_argument("ADC samples must be in increasing order within their block");
                if (pulse && t > start + kTimeTolerance && t < end - kTimeTolerance)
                    throw std::invalid_argument("an ADC sample falls inside the RF pulse");
                before = (pulse && t <= start + kTimeTolerance) ? k + 1 : before;
            }
            return before;
        }

        /** Each isochromat's position on which the field during a pulse
         *  depends, in units of the quantum. */
        std::vector<std::array<double, 3>> quantised(const IsochromatProperties& p, int mode, const double direction[3])
        {
            const size_t count = p.x.size();
            std::vector<std::array<double, 3>> positions(count, {0.0, 0.0, 0.0});
            for (size_t i = 0; i < count && mode == kOneDirection; ++i)
                positions[i][0] = std::floor(
                    (p.x[i] * direction[0] + p.y[i] * direction[1] + p.z[i] * direction[2]) / kQuantum + 0.5);
            const std::vector<double>* axes[3] = {&p.x, &p.y, &p.z};
            for (int axis = 0; axis < 3 && mode == kAnyDirection; ++axis)
                for (size_t i = 0; i < count && direction[axis] != 0.0; ++i)
                    positions[i][axis] = std::floor((*axes[axis])[i] / kQuantum + 0.5);
            return positions;
        }

    } // namespace

    Isochromats::Isochromats(IsochromatProperties properties, size_t threads)
        : properties_(std::move(properties))
    {
        check();
        threads_ = threads != 0 ? threads : std::max(1u, std::thread::hardware_concurrency());
        lay_out_receive();
        classify();
        reset();
    }

    void Isochromats::check()
    {
        IsochromatProperties& p = properties_;
        count_ = p.x.size();
        if (p.y.size() != count_ || p.z.size() != count_)
            throw std::invalid_argument("x, y and z must hold one position per isochromat");
        if (count_ > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("too many isochromats");
        fill(p.proton_density, count_, 1.0, "proton_density");
        fill(p.t1, count_, std::numeric_limits<double>::infinity(), "t1");
        fill(p.t2, count_, std::numeric_limits<double>::infinity(), "t2");
        fill(p.off_resonance, count_, 0.0, "off_resonance");
        for (const auto& named : {std::make_pair(&p.x, "x"), std::make_pair(&p.y, "y"), std::make_pair(&p.z, "z"),
                                  std::make_pair(&p.proton_density, "proton_density"),
                                  std::make_pair(&p.off_resonance, "off_resonance")})
            require_finite(*named.first, named.second);
        require_positive(p.t1);
        require_positive(p.t2);
        require_sensitivities(p.transmit, count_, p.transmit_channels, "transmit");
        require_sensitivities(p.receive, count_, p.coils, "receive");
        coils_ = p.coils == 0 ? 1 : p.coils;
    }

    void Isochromats::lay_out_receive()
    {
        IsochromatProperties& p = properties_;
        receive_re_.resize(count_ * p.coils);
        receive_im_.resize(count_ * p.coils);
        for (size_t i = 0; i < count_; ++i)
            for (size_t c = 0; c < p.coils; ++c)
            {
                receive_re_[c * count_ + i] = p.receive[i * p.coils + c].real();
                receive_im_[c * count_ + i] = p.receive[i * p.coils + c].imag();
            }
        p.receive.clear();
        p.receive.shrink_to_fit();
    }

    void Isochromats::classify()
    {
        /* Isochromats equal in everything but position see one field during
         * a pulse wherever their positions along its gradient agree. */
        const IsochromatProperties& p = properties_;
        const size_t width = 3 + 2 * p.transmit_channels;
        std::vector<uint64_t> keys(count_ * width);
        for (size_t i = 0; i < count_; ++i)
        {
            uint64_t* key = &keys[i * width];
            key[0] = bits_of(p.off_resonance[i]);
            key[1] = bits_of(p.t1[i]);
            key[2] = bits_of(p.t2[i]);
            for (size_t c = 0; c < p.transmit_channels; ++c)
            {
                key[3 + 2 * c] = bits_of(p.transmit[i * p.transmit_channels + c].real());
                key[4 + 2 * c] = bits_of(p.transmit[i * p.transmit_channels + c].imag());
            }
        }
        const auto row = [&](uint32_t i) { return keys.cbegin() + static_cast<std::ptrdiff_t>(i * width); };
        std::vector<uint32_t> order(count_);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return std::lexicographical_compare(row(a), row(a) + width, row(b), row(b) + width);
        });
        class_of_.assign(count_, 0);
        uint32_t current = 0;
        for (size_t n = 1; n < count_; ++n)
        {
            if (!std::equal(row(order[n]), row(order[n]) + width, row(order[n - 1])))
                ++current;
            class_of_[order[n]] = current;
        }
    }

    void Isochromats::reset()
    {
        const std::lock_guard<std::mutex> held(mutex_);
        mx_.assign(count_, 0.0);
        my_.assign(count_, 0.0);
        mz_ = properties_.proton_density;
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
        elapsed_ = 0.0;
    }

    void Isochromats::magnetization(double* into)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        flush();
        for (size_t i = 0; i < count_; ++i)
        {
            into[3 * i] = mx_[i];
            into[3 * i + 1] = my_[i];
            into[3 * i + 2] = mz_[i];
        }
    }

    void Isochromats::set_magnetization(const double* from)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        for (size_t i = 0; i < count_; ++i)
        {
            mx_[i] = from[3 * i];
            my_[i] = from[3 * i + 1];
            mz_[i] = from[3 * i + 2];
        }
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
    }

    void Isochromats::advance(const GradientAreas& areas, double& now, double to)
    {
        double from_area[3];
        double to_area[3];
        areas.at(now, from_area);
        areas.at(to, to_area);
        for (int axis = 0; axis < 3; ++axis)
            pending_area_[axis] += to_area[axis] - from_area[axis];
        pending_time_ += to - now;
        now = to;
    }

    void Isochromats::flush()
    {
        const double time = pending_time_;
        const double ax = pending_area_[0];
        const double ay = pending_area_[1];
        const double az = pending_area_[2];
        if (time == 0.0 && ax == 0.0 && ay == 0.0 && az == 0.0)
            return;
        const IsochromatProperties& p = properties_;
        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t i = first; i < last; ++i)
            {
                const double phase = -kTwoPi * (p.x[i] * ax + p.y[i] * ay + p.z[i] * az + p.off_resonance[i] * time);
                const double e2 = std::exp(-time * rate(p.t2[i]));
                const double e1 = std::exp(-time * rate(p.t1[i]));
                const double c = e2 * std::cos(phase);
                const double s = e2 * std::sin(phase);
                const double x = mx_[i];
                const double y = my_[i];
                mx_[i] = c * x - s * y;
                my_[i] = s * x + c * y;
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
    }

    const Isochromats::Grouping& Isochromats::grouping(int mode, const double direction[3])
    {
        for (const Grouping& held : groupings_)
            if (held.mode == mode && same_bits(held.direction, direction))
                return held;

        const std::vector<std::array<double, 3>> positions = quantised(properties_, mode, direction);
        const auto less = [&](uint32_t i, uint32_t j) {
            return class_of_[i] != class_of_[j] ? class_of_[i] < class_of_[j] : positions[i] < positions[j];
        };
        std::vector<uint32_t> order(count_);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), less);

        Grouping made;
        made.mode = mode;
        std::copy(direction, direction + 3, made.direction);
        made.group_of.assign(count_, 0);
        for (size_t n = 0; n < count_; ++n)
        {
            if (n == 0 || less(order[n - 1], order[n]))
                made.representative.push_back(order[n]);
            made.group_of[order[n]] = static_cast<uint32_t>(made.representative.size() - 1);
        }

        if (groupings_.size() >= kGroupings)
            groupings_.erase(groupings_.begin());
        groupings_.push_back(std::move(made));
        return groupings_.back();
    }

    void Isochromats::excite(const BlockEvents& block, const GradientAreas& areas)
    {
        const IsochromatProperties& p = properties_;
        const PulseGradient gradient = pulse_gradient(block, areas);
        const Grouping& groups = grouping(gradient.mode, gradient.direction);

        std::vector<std::complex<double>> summed;
        if (p.transmit_channels == 0)
        {
            summed.assign(block.rf_steps, 0.0);
            for (size_t c = 0; c < block.rf_channels; ++c)
                for (size_t i = 0; i < block.rf_steps; ++i)
                    summed[i] += block.rf[c * block.rf_steps + i];
        }

        /* Each group's pulse as one affine map, M -> A M + c * proton density. */
        const size_t count = groups.representative.size();
        std::vector<double> maps(12 * count);
        parallel(count, threads_, 8, [&](size_t, size_t first, size_t last) {
            for (size_t g = first; g < last; ++g)
                pulse_map(p, groups.representative[g], block, gradient, summed, &maps[12 * g]);
        });

        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t i = first; i < last; ++i)
            {
                const double* map = &maps[12 * groups.group_of[i]];
                const double x = mx_[i];
                const double y = my_[i];
                const double z = mz_[i];
                const double density = p.proton_density[i];
                mx_[i] = map[0] * x + map[1] * y + map[2] * z + map[9] * density;
                my_[i] = map[3] * x + map[4] * y + map[5] * z + map[10] * density;
                mz_[i] = map[6] * x + map[7] * y + map[8] * z + map[11] * density;
            }
        });
    }

    void Isochromats::acquire(
        const BlockEvents& block, const GradientAreas& areas, size_t first, size_t last, std::complex<double>* signal)
    {
        const IsochromatProperties& p = properties_;
        const size_t samples = last - first;
        const double* times = block.adc_times + first;
        const SampleSteps steps = sample_steps(times, samples, areas);
        const double span = times[samples - 1] - times[0];
        const Receive receive{receive_re_, receive_im_, coils_, count_};

        const size_t workers = workers_for(count_, threads_, kChunk);
        std::vector<std::vector<double>> sums(workers, std::vector<double>(2 * coils_ * samples, 0.0));
        parallel(count_, threads_, kChunk, [&](size_t worker, size_t begin, size_t end) {
            WindowReader reader(p, steps, samples, receive);
            for (size_t tile = begin; tile < end; tile += kTile)
                reader.tile(tile, std::min(kTile, end - tile), mx_.data(), my_.data(), sums[worker].data());
            for (size_t i = begin; i < end; ++i)
            {
                const double e1 = std::exp(-span * rate(p.t1[i]));
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });

        for (size_t coil = 0; coil < coils_; ++coil)
            for (size_t k = 0; k < samples; ++k)
            {
                std::complex<double> total = 0.0;
                for (const std::vector<double>& sum : sums)
                    total += std::complex<double>(sum[2 * (coil * samples + k)], sum[2 * (coil * samples + k) + 1]);
                signal[coil * block.adc_samples + first + k] = total;
            }
    }

    void Isochromats::play(const BlockEvents& block, std::complex<double>* signal)
    {
        if (!(block.duration >= 0.0) || !std::isfinite(block.duration))
            throw std::invalid_argument("a block's duration must be finite and not negative");
        check_gradients(block);
        check_pulse(block, properties_.transmit_channels);
        const size_t before = check_samples(block);
        const size_t samples = block.adc_samples;
        if (samples > 0 && signal == nullptr)
            throw std::invalid_argument("no room for the signal");

        const std::lock_guard<std::mutex> held(mutex_);
        const GradientAreas areas(block);
        double now = 0.0;
        if (before > 0)
        {
            advance(areas, now, block.adc_times[0]);
            flush();
            acquire(block, areas, 0, before, signal);
            now = block.adc_times[before - 1];
        }
        if (block.rf_steps > 0)
        {
            advance(areas, now, block.rf_start);
            flush();
            excite(block, areas);
            now = block.rf_start + static_cast<double>(block.rf_steps) * block.rf_step;
        }
        if (before < samples)
        {
            advance(areas, now, block.adc_times[before]);
            flush();
            acquire(block, areas, before, samples, signal);
            now = block.adc_times[samples - 1];
        }
        advance(areas, now, block.duration);
        elapsed_ += block.duration;
    }

} // namespace pulseq
