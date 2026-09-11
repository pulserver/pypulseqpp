/**
 * @file resonance.cpp
 * @brief Windowed gradient spectrum against forbidden bands.  See resonance.hpp.
 */

#include "pulseq/resonance.hpp"

#include "pulseq/corners.hpp"
#include "pulseq/rfft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace pulseq
{

    namespace
    {

        constexpr double kPi = 3.14159265358979323846;
        /* Fraction of a raster interval a time may miss a sample centre by. */
        constexpr double kRasterEps = 1e-9;

        /** First sample whose centre (n + 1/2) dt is not before @p time. */
        int64_t first_sample_from(double time, double dt)
        {
            return static_cast<int64_t>(std::ceil(time / dt - 0.5 - kRasterEps));
        }

        /** One gradient's corners, timed from the start of its block. */
        struct Played
        {
            const double* times = nullptr;
            const double* values = nullptr;
            size_t count = 0;
            double offset = 0.0;
            size_t at = 0;

            double when(size_t i) const
            {
                return times[i] + offset;
            }

            /** Value at @p t, zero outside the corners; @p t must not decrease. */
            double value(double t)
            {
                if (count == 0 || t < when(0) || t > when(count - 1))
                    return 0.0;
                while (at + 1 < count && when(at + 1) <= t)
                    ++at;
                if (at + 1 >= count)
                    return values[count - 1];
                const double span = when(at + 1) - when(at);
                if (span <= 0.0)
                    return values[at + 1];
                return values[at] + (values[at + 1] - values[at]) * (t - when(at)) / span;
            }
        };

        /** One band on one axis, as bins. */
        struct Guard
        {
            size_t reading = 0;
            int64_t lo = 1;
            int64_t hi = 0;
            double threshold = 0.0;
        };

    } // namespace

    ResonanceReport mech_resonance(
        const Sequence& seq,
        const std::vector<ForbiddenBand>& bands,
        const ResonanceOptions& options)
    {
        ResonanceReport out;
        const double dt = seq.grad_raster_time();
        const int64_t width = std::max<int64_t>(1, std::llround(options.window / dt));
        const int64_t step = std::max<int64_t>(1, std::llround(options.stride / dt));
        const int64_t length = width * std::max(1, options.oversampling);
        const int64_t nyquist = length / 2;
        out.window = static_cast<double>(width) * dt;
        out.stride = static_cast<double>(step) * dt;
        out.frequency_step = 1.0 / (static_cast<double>(length) * dt);

        std::vector<Guard> guards[3];
        for (size_t b = 0; b < bands.size(); ++b)
        {
            const ForbiddenBand& band = bands[b];
            double lo_at = band.f_min / out.frequency_step;
            double hi_at = band.f_max / out.frequency_step;
            int64_t lo = static_cast<int64_t>(std::ceil(lo_at - kRasterEps));
            int64_t hi = static_cast<int64_t>(std::floor(hi_at + kRasterEps));
            if (lo > hi)
                lo = hi = std::llround(0.5 * (lo_at + hi_at));
            lo = std::max<int64_t>(lo, 0);
            hi = std::min(hi, nyquist);

            for (int axis = 0; axis < 3; ++axis)
            {
                if (band.axis >= 0 && band.axis != axis)
                    continue;
                BandReading reading;
                reading.band = static_cast<int>(b);
                reading.axis = axis;
                Guard guard;
                guard.reading = out.readings.size();
                guard.lo = lo;
                guard.hi = hi;
                guard.threshold = band.threshold;
                out.readings.push_back(reading);
                if (lo <= hi)
                    guards[axis].push_back(guard);
            }
        }

        out.band_violations.assign(bands.size(), 0);
        /* A window violating a band on two axes is one violating window. */
        std::vector<int64_t> last_violated(bands.size(), -1);

        RealFft fft(static_cast<size_t>(length), options.mkl_runtime);
        out.backend = fft.backend();

        std::vector<double> taper(static_cast<size_t>(width));
        double taper_sum = 0.0;
        for (int64_t i = 0; i < width; ++i)
        {
            taper[static_cast<size_t>(i)] =
                0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) / width));
            taper_sum += taper[static_cast<size_t>(i)];
        }
        const double scale = taper_sum > 0.0 ? 2.0 / taper_sum : 0.0;

        std::vector<double> tapered(static_cast<size_t>(length), 0.0);
        std::vector<std::complex<double>> spectrum(static_cast<size_t>(nyquist) + 1);

        /* Samples [base, base + buffer.size()) of each physical axis. Only a
         * window's worth behind the next window start is kept, so a long scan
         * or a long delay never holds its timeline. */
        std::vector<double> buffer[3];
        int64_t base = 0;
        int64_t filled = 0;
        int64_t next = 0;

        const auto judge = [&](int64_t window) {
            const int64_t start = window * step;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (guards[axis].empty())
                    continue;
                const double* x = buffer[axis].data() + (start - base);
                double mean = 0.0;
                double low = x[0];
                double high = x[0];
                for (int64_t i = 0; i < width; ++i)
                {
                    mean += x[i];
                    low = std::min(low, x[i]);
                    high = std::max(high, x[i]);
                }
                /* A constant window has nothing left once its mean is gone. */
                if (high == low)
                    continue;
                mean /= static_cast<double>(width);
                for (int64_t i = 0; i < width; ++i)
                    tapered[static_cast<size_t>(i)] = (x[i] - mean) * taper[static_cast<size_t>(i)];
                fft.forward(tapered.data(), spectrum.data());

                for (const Guard& guard : guards[axis])
                {
                    double strongest = 0.0;
                    int64_t where = guard.lo;
                    for (int64_t k = guard.lo; k <= guard.hi; ++k)
                    {
                        const double magnitude = std::abs(spectrum[static_cast<size_t>(k)]);
                        if (magnitude > strongest)
                        {
                            strongest = magnitude;
                            where = k;
                        }
                    }
                    const double amplitude = strongest * scale;
                    BandReading& reading = out.readings[guard.reading];
                    if (amplitude > guard.threshold)
                    {
                        ++reading.violations;
                        const size_t band = static_cast<size_t>(reading.band);
                        if (last_violated[band] != window)
                        {
                            last_violated[band] = window;
                            ++out.band_violations[band];
                        }
                    }
                    if (amplitude > reading.peak)
                    {
                        reading.peak = amplitude;
                        reading.frequency = static_cast<double>(where) * out.frequency_step;
                        reading.window = window;
                        reading.window_start = static_cast<double>(start) * dt;
                    }
                }
            }
        };

        const auto grow_to = [&](int64_t end) {
            for (int axis = 0; axis < 3; ++axis)
                buffer[axis].resize(static_cast<size_t>(end - base), 0.0);
        };

        const auto judge_complete = [&]() {
            while (next * step + width <= filled)
                judge(next++);
            /* Dropped in bulk, so the front of the buffer moves at most once
             * per few windows rather than once per block. */
            const int64_t keep_from = next * step;
            if (keep_from - base > 4 * width)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    buffer[axis].erase(
                        buffer[axis].begin(),
                        buffer[axis].begin() + static_cast<std::ptrdiff_t>(keep_from - base));
                }
                base = keep_from;
            }
        };

        CornerCache corners(seq, dt);
        std::vector<double> trapezoid_times[3];
        const int32_t* events = seq.block_events();
        const double* durations = seq.block_durations();
        const int blocks = seq.num_blocks();

        double elapsed = 0.0;
        for (int index = 0; index < blocks; ++index)
        {
            const int32_t* row = events + static_cast<size_t>(index) * BLOCK_WIDTH;
            const double ends = elapsed + durations[index];
            const int64_t from = filled;
            const int64_t to = std::max(from, first_sample_from(ends, dt));

            Played played[3];
            bool any = false;
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0)
                    continue;
                const Corners& shape = corners[id];
                /* No ramps and no flat top plays for no time. */
                if (shape.empty_with_amplitude || shape.values.empty())
                    continue;
                Played& p = played[axis];
                p.values = shape.values.data();
                p.count = shape.values.size();
                if (shape.trapezoid)
                {
                    shape.at(0.0, trapezoid_times[axis]);
                    p.times = trapezoid_times[axis].data();
                }
                else
                {
                    p.times = shape.times.data();
                    p.offset = shape.delay;
                }
                any = true;
            }

            double turn[3][3];
            const int32_t rotation = row[BLOCK_ROTATION_COLUMN];
            double own[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
            if (rotation >= 1 && rotation <= seq.rotation_library().size())
                rotation_matrix(seq.rotation_library().row(rotation), own);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    turn[i][j] = options.rotation[i][0] * own[0][j] +
                        options.rotation[i][1] * own[1][j] + options.rotation[i][2] * own[2][j];

            /* A block is written in pieces that end where the next window
             * does, so a delay of minutes is judged as it is sampled. */
            for (int64_t n = from; n < to;)
            {
                const int64_t piece = std::min(to, std::max(n + 1, next * step + width));
                grow_to(piece);
                if (any)
                {
                    for (int64_t m = n; m < piece; ++m)
                    {
                        const double t = (static_cast<double>(m) + 0.5) * dt - elapsed;
                        double logical[3];
                        for (int axis = 0; axis < 3; ++axis)
                            logical[axis] = played[axis].value(t);
                        const size_t at = static_cast<size_t>(m - base);
                        for (int axis = 0; axis < 3; ++axis)
                        {
                            buffer[axis][at] = turn[axis][0] * logical[0] +
                                turn[axis][1] * logical[1] + turn[axis][2] * logical[2];
                        }
                    }
                }
                filled = piece;
                n = piece;
                judge_complete();
            }
            elapsed = ends;
        }

        if (filled > 0)
        {
            const int64_t windows =
                filled <= width ? 1 : (filled - width + step - 1) / step + 1;
            while (next < windows)
            {
                filled = std::max(filled, next * step + width);
                grow_to(filled);
                judge_complete();
            }
            out.windows = windows;
        }
        return out;
    }

} // namespace pulseq
