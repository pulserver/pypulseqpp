/**
 * @file resonance.cpp
 * @brief Windowed gradient spectrum against forbidden bands.  See resonance.hpp.
 *
 * Only the bins a band reads are computed, as direct sums over the samples.
 * A window's taper and mean are applied to those sums afterwards, so a block's
 * contribution to any window is its own sums turned in phase. Sums are taken
 * per logical channel and combined through the block's rotation; a
 * trapezoid's are taken per unit amplitude, so one phase-encode table shares
 * them, and a channel with few corners is summed in closed form, a straight
 * stretch at a time.
 */

#include "pulseq/resonance.hpp"

#include "pulseq/raster.hpp"
#include "pulseq/rfft.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_map>

namespace pulseq
{

    namespace
    {

        using Complex = std::complex<double>;

        constexpr double kPi = 3.14159265358979323846;
        /* Fraction of a bin a band edge may miss a bin by. */
        constexpr double kBinEps = 1e-9;

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11 && defined(__x86_64__) && \
    defined(__linux__)
/* Resolved once per process to the widest vector unit the processor has. */
#define PULSEQ_RESONANCE_CLONES \
    __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define PULSEQ_RESONANCE_CLONES
#endif

        /** Samples between exact phases, so that rotating one by multiplication
         * gathers no more than this many roundings. */
        constexpr int64_t kReseed = 256;

        /** Corners up to which a channel is summed a straight stretch at a time
         * rather than a sample at a time. */
        constexpr size_t kLineCorners = 16;

        /* Fraction of a raster interval a corner may miss a sample centre by. */
        constexpr double kRasterEps = 1e-9;

        /* The loops below run over bins, one array per real quantity, so that
         * they vectorise; each table is indexed by (k t) mod L. */

        /** (index * t) mod length, for t already reduced mod length; exact in doubles. */
        PULSEQ_RESONANCE_CLONES
        void phase_indices(size_t bins, const double* __restrict index, double t, double length,
                           double inverse, int32_t* __restrict out)
        {
            for (size_t i = 0; i < bins; ++i)
            {
                const double product = index[i] * t;
                double rest = product - length * std::floor(product * inverse);
                rest += rest < 0.0 ? length : 0.0;
                rest -= rest >= length ? length : 0.0;
                out[i] = static_cast<int32_t>(rest);
            }
        }

        /** sum += (c + i s) * part. */
        PULSEQ_RESONANCE_CLONES
        void turned_add(size_t bins, const double* __restrict at_re, const double* __restrict at_im,
                        const double* __restrict part_re, const double* __restrict part_im,
                        double* __restrict sum_re, double* __restrict sum_im)
        {
            for (size_t i = 0; i < bins; ++i)
            {
                const double c = at_re[i];
                const double s = at_im[i];
                sum_re[i] += c * part_re[i] - s * part_im[i];
                sum_im[i] += c * part_im[i] + s * part_re[i];
            }
        }

        /**
         * sum += r^j (a sum_{m<n} r^m + b sum_{m<n} m r^m), r = exp(-2 pi i k / L),
         * given r^j and r^n; zero at k = 0, which the caller adds.
         */
        PULSEQ_RESONANCE_CLONES
        void line_add(size_t bins, const double* __restrict at_re, const double* __restrict at_im,
                      const double* __restrict span_re, const double* __restrict span_im,
                      const double* __restrict r_re, const double* __restrict r_im,
                      const double* __restrict rise_re, const double* __restrict rise_im,
                      const double* __restrict ramp_re, const double* __restrict ramp_im, double a,
                      double b, double n, double* __restrict sum_re, double* __restrict sum_im)
        {
            for (size_t i = 0; i < bins; ++i)
            {
                const double rn_re = span_re[i];
                const double rn_im = span_im[i];
                /* sum r^m = (1 - r^n) / (1 - r). */
                const double one_re = 1.0 - rn_re;
                const double flat_re = one_re * rise_re[i] + rn_im * rise_im[i];
                const double flat_im = one_re * rise_im[i] - rn_im * rise_re[i];
                /* sum m r^m = r (1 - n r^(n-1) + (n - 1) r^n) / (1 - r)^2. */
                const double back_re = rn_re * r_re[i] + rn_im * r_im[i];
                const double back_im = rn_im * r_re[i] - rn_re * r_im[i];
                const double in_re = 1.0 - n * back_re + (n - 1.0) * rn_re;
                const double in_im = -n * back_im + (n - 1.0) * rn_im;
                const double v_re = a * flat_re + b * (ramp_re[i] * in_re - ramp_im[i] * in_im);
                const double v_im = a * flat_im + b * (ramp_re[i] * in_im + ramp_im[i] * in_re);
                const double c = at_re[i];
                const double s = at_im[i];
                sum_re[i] += c * v_re - s * v_im;
                sum_im[i] += c * v_im + s * v_re;
            }
        }

        /** sum += weight * part. */
        PULSEQ_RESONANCE_CLONES
        void weighted_add(size_t bins, double weight, const double* __restrict part_re,
                          const double* __restrict part_im, double* __restrict sum_re,
                          double* __restrict sum_im)
        {
            for (size_t i = 0; i < bins; ++i)
            {
                sum_re[i] += weight * part_re[i];
                sum_im[i] += weight * part_im[i];
            }
        }

        /**
         * Add @p count samples to sums at @p bins bins whose phases start at
         * (re, im) and turn by (step_re, step_im) per sample.
         */
        PULSEQ_RESONANCE_CLONES
        void accumulate(int64_t count, const double* __restrict v, size_t bins,
                        double* __restrict re, double* __restrict im,
                        const double* __restrict step_re, const double* __restrict step_im,
                        double* __restrict sum_re, double* __restrict sum_im)
        {
            for (int64_t j = 0; j < count; ++j)
            {
                const double x = v[j];
                for (size_t i = 0; i < bins; ++i)
                {
                    sum_re[i] += x * re[i];
                    sum_im[i] += x * im[i];
                    const double r = re[i] * step_re[i] - im[i] * step_im[i];
                    im[i] = re[i] * step_im[i] + im[i] * step_re[i];
                    re[i] = r;
                }
            }
        }

        /** One band on one axis, as bins, with where each bin and its taper
         * neighbours sit among that axis's sums. */
        struct Guard
        {
            size_t reading = 0;
            int64_t lo = 1;
            int64_t hi = 0;
            double threshold = 0.0;
            std::vector<size_t> at;
            std::vector<size_t> below;
            std::vector<size_t> above;
        };

        /**
         * Sums of samples against exp(-2 pi i k t / L) at one axis's bins, and
         * the extremes of the samples summed. A sum over samples t0, t0 + 1, ...
         * is referred to t0 when it belongs to a block and to sample 0 when it
         * belongs to an interval of the timeline.
         */
        struct Sums
        {
            std::vector<double> re;
            std::vector<double> im;
            double low[3];
            double high[3];

            void reset(size_t bins)
            {
                re.assign(bins, 0.0);
                im.assign(bins, 0.0);
                for (int axis = 0; axis < 3; ++axis)
                {
                    low[axis] = std::numeric_limits<double>::infinity();
                    high[axis] = -std::numeric_limits<double>::infinity();
                }
            }
        };

        /** The timeline between two consecutive window edges. */
        struct Interval
        {
            int64_t begin = 0;
            int64_t end = 0;
            Sums sums;
        };

        /**
         * What makes one logical channel play one waveform, up to its amplitude,
         * over a block's samples: a trapezoid's timing, or an arbitrary
         * gradient's id, with the samples and their offset as in BlockKey.
         */
        struct ChannelKey
        {
            int32_t gradient;
            int64_t timing[4];
            int64_t samples;
            int64_t offset;

            bool operator==(const ChannelKey& other) const
            {
                return gradient == other.gradient && timing[0] == other.timing[0] &&
                    timing[1] == other.timing[1] && timing[2] == other.timing[2] &&
                    timing[3] == other.timing[3] && samples == other.samples &&
                    offset == other.offset;
            }
        };

        struct ChannelKeyHash
        {
            size_t operator()(const ChannelKey& key) const
            {
                uint64_t h = 1469598103934665603ull;
                const auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
                mix(static_cast<uint32_t>(key.gradient));
                for (int64_t v : key.timing)
                    mix(static_cast<uint64_t>(v));
                mix(static_cast<uint64_t>(key.samples));
                mix(static_cast<uint64_t>(key.offset));
                return static_cast<size_t>(h);
            }
        };

        class Spectra
        {
        public:
            Spectra(int64_t length, const std::vector<int64_t> (&bins)[3])
                : length_(length)
            {
                turn_re_.resize(static_cast<size_t>(length));
                turn_im_.resize(static_cast<size_t>(length));
                for (int64_t m = 0; m < length; ++m)
                {
                    const double angle = -2.0 * kPi * static_cast<double>(m) / length;
                    turn_re_[static_cast<size_t>(m)] = std::cos(angle);
                    turn_im_[static_cast<size_t>(m)] = std::sin(angle);
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    first_[axis] = count_;
                    for (int64_t k : bins[axis])
                    {
                        const int64_t r = k % length_;
                        const int64_t index = r < 0 ? r + length_ : r;
                        if (index == 0)
                            zero_.push_back(index_.size());
                        follows_.push_back(!index_.empty() &&
                                           index == (static_cast<int64_t>(index_.back()) + 1) % length_);
                        index_.push_back(static_cast<double>(index));
                        const Complex unit = turn_at(index);
                        const Complex rise = index == 0 ? Complex() : 1.0 / (1.0 - unit);
                        const Complex ramp = unit * rise * rise;
                        r_re_.push_back(unit.real());
                        r_im_.push_back(unit.imag());
                        rise_re_.push_back(rise.real());
                        rise_im_.push_back(rise.imag());
                        ramp_re_.push_back(ramp.real());
                        ramp_im_.push_back(ramp.imag());
                    }
                    count_ += bins[axis].size();
                    last_[axis] = count_;
                }
                at_.resize(count_);
                at_re_.resize(count_);
                at_im_.resize(count_);
                span_re_.resize(count_);
                span_im_.resize(count_);
                re_.resize(count_);
                im_.resize(count_);
                sum_re_.resize(count_);
                sum_im_.resize(count_);
            }

            size_t bins() const
            {
                return count_;
            }

            size_t first(int axis) const
            {
                return first_[axis];
            }

            /** exp(-2 pi i m / L). */
            Complex turn_at(int64_t m) const
            {
                return {turn_re_[static_cast<size_t>(m)], turn_im_[static_cast<size_t>(m)]};
            }

            /** (k t) mod L for every bin, into @p out. */
            void phases(int64_t t, int32_t* out) const
            {
                phase_indices(count_, index_.data(), static_cast<double>(t % length_),
                              static_cast<double>(length_), 1.0 / static_cast<double>(length_), out);
            }

            /** exp(-2 pi i k t / L) for every bin, into (re, im). A bin one
             * above the previous turns t further, by addition mod L. */
            void turns(int64_t t, double* re, double* im) const
            {
                const int32_t step = static_cast<int32_t>(t % length_);
                const int32_t length = static_cast<int32_t>(length_);
                int32_t at = 0;
                for (size_t i = 0; i < count_; ++i)
                {
                    if (follows_[i])
                    {
                        at += step;
                        at -= at >= length ? length : 0;
                    }
                    else
                    {
                        at = static_cast<int32_t>(
                            (static_cast<int64_t>(index_[i]) * step) % length_);
                    }
                    re[i] = turn_re_[static_cast<size_t>(at)];
                    im[i] = turn_im_[static_cast<size_t>(at)];
                }
            }

            /** Add samples t, t + 1, ... of each axis to sums referred to sample 0. */
            void add(Sums& sums, int64_t t, int64_t count, const double* const (&x)[3]) const
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double* v = x[axis];
                    for (int64_t j = 0; j < count; ++j)
                    {
                        sums.low[axis] = std::min(sums.low[axis], v[j]);
                        sums.high[axis] = std::max(sums.high[axis], v[j]);
                    }
                }
                std::fill(sum_re_.begin(), sum_re_.end(), 0.0);
                std::fill(sum_im_.begin(), sum_im_.end(), 0.0);
                for (int64_t done = 0; done < count; done += kReseed)
                {
                    const int64_t chunk = std::min(kReseed, count - done);
                    phases(t + done, at_.data());
                    for (size_t i = 0; i < count_; ++i)
                    {
                        re_[i] = turn_re_[static_cast<size_t>(at_[i])];
                        im_[i] = turn_im_[static_cast<size_t>(at_[i])];
                    }
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const size_t f = first_[axis];
                        accumulate(chunk, x[axis] + done, last_[axis] - f, &re_[f], &im_[f],
                                   &r_re_[f], &r_im_[f], &sum_re_[f], &sum_im_[f]);
                    }
                }
                for (size_t i = 0; i < count_; ++i)
                {
                    sums.re[i] += sum_re_[i];
                    sums.im[i] += sum_im_[i];
                }
            }

            /**
             * Add samples j, j + 1, ..., j + n - 1 valued a + b (i - j) on every
             * axis to sums referred to sample 0, in closed form.
             */
            void add_line(Sums& sums, int64_t j, int64_t n, double a, double b) const
            {
                const double last = a + b * static_cast<double>(n - 1);
                for (int axis = 0; axis < 3; ++axis)
                {
                    sums.low[axis] = std::min({sums.low[axis], a, last});
                    sums.high[axis] = std::max({sums.high[axis], a, last});
                }
                turns(j, at_re_.data(), at_im_.data());
                turns(n, span_re_.data(), span_im_.data());
                const double count = static_cast<double>(n);
                line_add(count_, at_re_.data(), at_im_.data(), span_re_.data(), span_im_.data(),
                         r_re_.data(), r_im_.data(), rise_re_.data(), rise_im_.data(),
                         ramp_re_.data(), ramp_im_.data(), a, b, count, sums.re.data(),
                         sums.im.data());
                for (size_t i : zero_)
                    sums.re[i] += a * count + b * 0.5 * count * (count - 1.0);
            }

            /** Add sums referred to t to sums referred to sample 0. */
            void add_from(Sums& sums, int64_t t, const Sums& part) const
            {
                turns(t, at_re_.data(), at_im_.data());
                turned_add(count_, at_re_.data(), at_im_.data(), part.re.data(), part.im.data(),
                           sums.re.data(), sums.im.data());
                for (int axis = 0; axis < 3; ++axis)
                {
                    sums.low[axis] = std::min(sums.low[axis], part.low[axis]);
                    sums.high[axis] = std::max(sums.high[axis], part.high[axis]);
                }
            }

        private:
            int64_t length_;
            std::vector<double> turn_re_;
            std::vector<double> turn_im_;
            /** k mod L per bin. */
            std::vector<double> index_;
            /** Whether a bin's k mod L is the previous bin's plus one. */
            std::vector<char> follows_;
            /** Bins at k = 0 mod L, where the closed forms divide by zero. */
            std::vector<size_t> zero_;
            /** r = exp(-2 pi i k / L), 1 / (1 - r) and r / (1 - r)^2 per bin. */
            std::vector<double> r_re_;
            std::vector<double> r_im_;
            std::vector<double> rise_re_;
            std::vector<double> rise_im_;
            std::vector<double> ramp_re_;
            std::vector<double> ramp_im_;
            size_t first_[3] = {0, 0, 0};
            size_t last_[3] = {0, 0, 0};
            size_t count_ = 0;
            mutable std::vector<int32_t> at_;
            mutable std::vector<double> at_re_;
            mutable std::vector<double> at_im_;
            mutable std::vector<double> span_re_;
            mutable std::vector<double> span_im_;
            mutable std::vector<double> re_;
            mutable std::vector<double> im_;
            mutable std::vector<double> sum_re_;
            mutable std::vector<double> sum_im_;
        };

        /** The FFT of one window of every axis, for a diagnostic. */
        void keep_spectrum(
            const Sequence& seq,
            const ResonanceOptions& options,
            int64_t width,
            int64_t step,
            int64_t length,
            const std::vector<double>& taper,
            double scale,
            ResonanceReport& out)
        {
            const int64_t start = options.keep_spectrum * step;
            const int64_t nyquist = length / 2;
            std::vector<double> samples[3];
            for (int axis = 0; axis < 3; ++axis)
                samples[axis].assign(static_cast<size_t>(width), 0.0);

            PhysicalRaster raster(seq, options.rotation);
            int64_t count;
            while ((count = raster.enter_block()) > 0 && raster.position() < start + width)
            {
                const int64_t at = raster.position();
                if (at + count <= start)
                {
                    raster.skip(count);
                    continue;
                }
                if (at < start)
                {
                    raster.skip(start - at);
                    continue;
                }
                const int64_t wanted = std::min(count, start + width - at);
                const size_t into = static_cast<size_t>(at - start);
                raster.read(wanted, &samples[0][into], &samples[1][into], &samples[2][into]);
            }

            RealFft fft(static_cast<size_t>(length), options.mkl_runtime);
            std::vector<double> tapered(static_cast<size_t>(length), 0.0);
            std::vector<Complex> spectrum(static_cast<size_t>(nyquist) + 1);
            out.spectrum.assign(3, std::vector<double>(static_cast<size_t>(nyquist) + 1, 0.0));
            out.spectrum_start = static_cast<double>(start) * seq.grad_raster_time();
            for (int axis = 0; axis < 3; ++axis)
            {
                const std::vector<double>& x = samples[axis];
                const auto [low, high] = std::minmax_element(x.begin(), x.end());
                if (*low == *high)
                    continue;
                double mean = 0.0;
                for (double v : x)
                    mean += v;
                mean /= static_cast<double>(width);
                for (int64_t i = 0; i < width; ++i)
                {
                    tapered[static_cast<size_t>(i)] =
                        (x[static_cast<size_t>(i)] - mean) * taper[static_cast<size_t>(i)];
                }
                fft.forward(tapered.data(), spectrum.data());
                for (int64_t k = 0; k <= nyquist; ++k)
                {
                    out.spectrum[static_cast<size_t>(axis)][static_cast<size_t>(k)] =
                        std::abs(spectrum[static_cast<size_t>(k)]) * scale;
                }
            }
        }

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
        const int64_t taper_shift = std::max(1, options.oversampling);
        const int64_t length = width * taper_shift;
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
            int64_t lo = static_cast<int64_t>(std::ceil(lo_at - kBinEps));
            int64_t hi = static_cast<int64_t>(std::floor(hi_at + kBinEps));
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

        std::vector<double> taper(static_cast<size_t>(width));
        double taper_sum = 0.0;
        for (int64_t i = 0; i < width; ++i)
        {
            taper[static_cast<size_t>(i)] =
                0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) / width));
            taper_sum += taper[static_cast<size_t>(i)];
        }
        const double scale = taper_sum > 0.0 ? 2.0 / taper_sum : 0.0;

        out.backend = RealFft(static_cast<size_t>(length), options.mkl_runtime).backend();

        /* Each axis's bins: the window mean's (0), and every guarded bin with
         * the two the Hann taper mixes into it, taper_shift bins either side. */
        std::vector<int64_t> bins[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            if (guards[axis].empty())
                continue;
            bins[axis].push_back(0);
            for (const Guard& guard : guards[axis])
            {
                for (int64_t k = guard.lo - taper_shift; k <= guard.hi + taper_shift; ++k)
                    bins[axis].push_back(k);
            }
            std::sort(bins[axis].begin(), bins[axis].end());
            bins[axis].erase(std::unique(bins[axis].begin(), bins[axis].end()), bins[axis].end());
        }
        const Spectra spectra(length, bins);
        const auto where = [&](int axis, int64_t k) {
            const auto found = std::lower_bound(bins[axis].begin(), bins[axis].end(), k);
            return spectra.first(axis) + static_cast<size_t>(found - bins[axis].begin());
        };
        for (int axis = 0; axis < 3; ++axis)
        {
            for (Guard& guard : guards[axis])
            {
                for (int64_t k = guard.lo; k <= guard.hi; ++k)
                {
                    guard.at.push_back(where(axis, k));
                    guard.below.push_back(where(axis, k - taper_shift));
                    guard.above.push_back(where(axis, k + taper_shift));
                }
            }
        }
        const size_t total_bins = spectra.bins();

        /* sum_{n < width} exp(-2 pi i k n / L): the untapered window's response
         * to a constant, which the window mean is removed through. */
        std::vector<Complex> flat(total_bins);
        {
            std::vector<int32_t> unit_at(total_bins);
            std::vector<int32_t> width_at(total_bins);
            spectra.phases(1, unit_at.data());
            spectra.phases(width, width_at.data());
            for (size_t i = 0; i < total_bins; ++i)
            {
                flat[i] = unit_at[i] == 0
                    ? Complex(static_cast<double>(width), 0.0)
                    : (1.0 - spectra.turn_at(width_at[i])) / (1.0 - spectra.turn_at(unit_at[i]));
            }
        }
        size_t mean_at[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!bins[axis].empty())
                mean_at[axis] = where(axis, 0);
        }

        std::deque<Interval> done;
        Interval current;
        current.sums.reset(total_bins);
        const auto next_edge = [&](int64_t at) {
            const int64_t start = (at / step + 1) * step;
            const int64_t end = at < width ? width : ((at - width) / step + 1) * step + width;
            return std::min(start, end);
        };
        current.end = next_edge(0);

        std::vector<double> sum_re(total_bins);
        std::vector<double> sum_im(total_bins);
        std::vector<double> back_re(total_bins);
        std::vector<double> back_im(total_bins);
        std::vector<double> plain_re(total_bins);
        std::vector<double> plain_im(total_bins);
        int64_t next = 0;
        int64_t samples = -1;

        const auto judge = [&](int64_t window) {
            const int64_t start = window * step;
            std::fill(sum_re.begin(), sum_re.end(), 0.0);
            std::fill(sum_im.begin(), sum_im.end(), 0.0);
            double low[3];
            double high[3];
            for (int axis = 0; axis < 3; ++axis)
            {
                /* Past the end of the sequence the window is zero-filled. */
                const bool padded = samples >= 0 && start + width > samples;
                low[axis] = padded ? 0.0 : std::numeric_limits<double>::infinity();
                high[axis] = padded ? 0.0 : -std::numeric_limits<double>::infinity();
            }
            for (const Interval& interval : done)
            {
                if (interval.begin < start || interval.begin >= start + width)
                    continue;
                weighted_add(total_bins, 1.0, interval.sums.re.data(), interval.sums.im.data(),
                             sum_re.data(), sum_im.data());
                for (int axis = 0; axis < 3; ++axis)
                {
                    low[axis] = std::min(low[axis], interval.sums.low[axis]);
                    high[axis] = std::max(high[axis], interval.sums.high[axis]);
                }
            }
            spectra.turns(start, back_re.data(), back_im.data());
            for (int axis = 0; axis < 3; ++axis)
            {
                /* A constant window has nothing left once its mean is gone. */
                if (guards[axis].empty() || !(high[axis] > low[axis]))
                    continue;
                const double mean = sum_re[mean_at[axis]] / static_cast<double>(width);
                const size_t first = spectra.first(axis);
                const size_t last = first + bins[axis].size();
                /* Referred to the window's start, less the mean's response. */
                for (size_t i = first; i < last; ++i)
                {
                    plain_re[i] = back_re[i] * sum_re[i] + back_im[i] * sum_im[i] -
                        mean * flat[i].real();
                    plain_im[i] = back_re[i] * sum_im[i] - back_im[i] * sum_re[i] -
                        mean * flat[i].imag();
                }

                for (const Guard& guard : guards[axis])
                {
                    double strongest = 0.0;
                    int64_t found = guard.lo;
                    for (size_t j = 0; j < guard.at.size(); ++j)
                    {
                        /* The Hann taper, applied to the untapered bins. */
                        const double re = 0.5 * plain_re[guard.at[j]] -
                            0.25 * (plain_re[guard.below[j]] + plain_re[guard.above[j]]);
                        const double im = 0.5 * plain_im[guard.at[j]] -
                            0.25 * (plain_im[guard.below[j]] + plain_im[guard.above[j]]);
                        const double power = re * re + im * im;
                        if (power > strongest)
                        {
                            strongest = power;
                            found = guard.lo + static_cast<int64_t>(j);
                        }
                    }
                    const double amplitude = std::sqrt(strongest) * scale;
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
                        reading.frequency = static_cast<double>(found) * out.frequency_step;
                        reading.window = window;
                        reading.window_start = static_cast<double>(start) * dt;
                    }
                }
            }
        };

        /* Intervals judged and dropped, kept for their storage. */
        std::vector<Interval> spare;
        const auto close_interval = [&]() {
            const int64_t end = current.end;
            done.push_back(std::move(current));
            if (spare.empty())
            {
                current = Interval();
            }
            else
            {
                current = std::move(spare.back());
                spare.pop_back();
            }
            current.sums.reset(total_bins);
            current.begin = end;
            current.end = next_edge(end);
            while (next * step + width <= end)
                judge(next++);
            while (!done.empty() && done.front().begin < next * step)
            {
                spare.push_back(std::move(done.front()));
                done.pop_front();
            }
        };

        /* Each logical channel's sums over a block, per unit amplitude, by
         * linearity: the block's sums on a physical axis are the channels'
         * turned by its rotation and scaled by their amplitudes. A channel's
         * sums are taken once per distinct bin, whichever axes read it. */
        std::vector<int64_t> distinct[3];
        for (int axis = 0; axis < 3; ++axis)
            distinct[0].insert(distinct[0].end(), bins[axis].begin(), bins[axis].end());
        std::sort(distinct[0].begin(), distinct[0].end());
        distinct[0].erase(std::unique(distinct[0].begin(), distinct[0].end()), distinct[0].end());
        const Spectra channel_spectra(length, distinct);
        const size_t channel_bins = channel_spectra.bins();
        /* Each axis's bins, as positions among the distinct ones. */
        std::vector<size_t> source(total_bins);
        for (int axis = 0; axis < 3; ++axis)
        {
            for (size_t i = 0; i < bins[axis].size(); ++i)
            {
                source[spectra.first(axis) + i] = static_cast<size_t>(
                    std::lower_bound(distinct[0].begin(), distinct[0].end(), bins[axis][i]) -
                    distinct[0].begin());
            }
        }

        std::unordered_map<ChannelKey, std::unique_ptr<Sums>, ChannelKeyHash> memory;
        Sums part;
        Sums channel_part;
        Sums rest;
        std::vector<double> channel_samples;

        /* Channel c's samples [from, from + n) of the current block, divided by
         * amplitude, into @p out, referred to the first of them. A channel
         * with few corners is summed a straight stretch at a time. */
        const auto channel_sums = [&](const PhysicalRaster& raster, int c, int64_t from, int64_t n,
                                      double amplitude, Sums& out) {
            out.reset(channel_bins);
            const PhysicalRaster::ChannelCorners corners = raster.played(c);
            if (corners.count > kLineCorners)
            {
                channel_samples.resize(static_cast<size_t>(n));
                raster.sample_channel(c, from, n, channel_samples.data());
                for (double& v : channel_samples)
                    v /= amplitude;
                const double* const x[3] = {channel_samples.data(), channel_samples.data(),
                                            channel_samples.data()};
                channel_spectra.add(out, 0, n, x);
                return;
            }
            const double t0 =
                (static_cast<double>(raster.position() + from) + 0.5) * dt - raster.block_start();
            const auto first_at = [&](double when) {
                const double j = std::ceil((when + corners.offset - t0) / dt - kRasterEps);
                return static_cast<int64_t>(std::clamp(j, 0.0, static_cast<double>(n)));
            };
            int64_t covered = 0;
            int64_t begin = first_at(corners.times[0]);
            for (size_t i = 0; i + 1 < corners.count; ++i)
            {
                const int64_t end = first_at(corners.times[i + 1]);
                if (end > begin)
                {
                    const double span = corners.times[i + 1] - corners.times[i];
                    const double slope = (corners.values[i + 1] - corners.values[i]) / span;
                    const double start = t0 + static_cast<double>(begin) * dt - corners.offset;
                    const double a = corners.values[i] + slope * (start - corners.times[i]);
                    channel_spectra.add_line(out, begin, end - begin, a / amplitude,
                                             slope * dt / amplitude);
                    covered += end - begin;
                }
                begin = std::max(begin, end);
            }
            if (covered < n)
            {
                out.low[0] = std::min(out.low[0], 0.0);
                out.high[0] = std::max(out.high[0], 0.0);
            }
        };

        /* Channel c's sums over the whole block, remembered from the second
         * time its waveform appears: most that appear once never appear again. */
        const auto whole_sums = [&](const PhysicalRaster& raster, int c, const BlockKey& block,
                                    const Corners& corners, double amplitude) -> const Sums* {
            ChannelKey key = {corners.trapezoid ? 0 : block.gradient[c], {0, 0, 0, 0},
                              block.samples, block.offset};
            if (corners.trapezoid)
            {
                const double unit = dt * kOffsetStep;
                key.timing[0] = std::llround(corners.delay / unit);
                for (int i = 0; i < 3; ++i)
                    key.timing[1 + i] = std::llround(corners.ramps[i] / unit);
            }
            auto found = memory.find(key);
            if (found != memory.end() && found->second)
                return found->second.get();
            if (found == memory.end())
            {
                memory.emplace(key, nullptr);
                return nullptr;
            }
            channel_sums(raster, c, 0, block.samples, amplitude, channel_part);
            found->second = std::make_unique<Sums>(channel_part);
            return found->second.get();
        };

        /* Samples [from, from + n) of the current block into part, referred to
         * the first of them. */
        const auto block_sums = [&](const PhysicalRaster& raster, int64_t count, int64_t from,
                                    int64_t n) {
            part.reset(total_bins);
            const double(&turn)[3][3] = raster.rotation();
            const BlockKey block = raster.block_key(count);
            double low[3] = {0.0, 0.0, 0.0};
            double high[3] = {0.0, 0.0, 0.0};
            for (int c = 0; c < 3; ++c)
            {
                const Corners* corners = raster.channel(c);
                if (!corners || (corners->trapezoid && corners->amplitude == 0.0))
                    continue;
                const double amplitude = corners->trapezoid ? corners->amplitude : 1.0;
                const Sums* whole = whole_sums(raster, c, block, *corners, amplitude);
                const Sums* unit = &channel_part;
                if (from == 0 && n == count)
                {
                    if (whole)
                        unit = whole;
                    else
                        channel_sums(raster, c, 0, n, amplitude, channel_part);
                }
                else if (whole && 2 * n > count && (from == 0 || from + n == count))
                {
                    /* The longer end of a remembered block, as the whole less
                     * the shorter end: W = P + r^n R for a prefix P of n samples,
                     * and S = r^-from (W - P') for the suffix after P'. */
                    channel_part.reset(channel_bins);
                    if (from == 0)
                    {
                        channel_sums(raster, c, n, count - n, -amplitude, rest);
                        channel_spectra.add_from(channel_part, 0, *whole);
                        channel_spectra.add_from(channel_part, n, rest);
                    }
                    else
                    {
                        channel_sums(raster, c, 0, from, -amplitude, rest);
                        for (size_t i = 0; i < channel_bins; ++i)
                        {
                            rest.re[i] += whole->re[i];
                            rest.im[i] += whole->im[i];
                        }
                        channel_spectra.add_from(channel_part, length - from % length, rest);
                    }
                    channel_part.low[0] = whole->low[0];
                    channel_part.high[0] = whole->high[0];
                }
                else
                {
                    channel_sums(raster, c, from, n, amplitude, channel_part);
                }
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double weight = turn[axis][c] * amplitude;
                    if (weight == 0.0)
                        continue;
                    const size_t first = spectra.first(axis);
                    const size_t last = first + bins[axis].size();
                    for (size_t i = first; i < last; ++i)
                    {
                        part.re[i] += weight * unit->re[source[i]];
                        part.im[i] += weight * unit->im[source[i]];
                    }
                    const double a = weight * unit->low[0];
                    const double b = weight * unit->high[0];
                    low[axis] += std::min(a, b);
                    high[axis] += std::max(a, b);
                }
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                part.low[axis] = low[axis];
                part.high[axis] = high[axis];
            }
        };

        PhysicalRaster raster(seq, options.rotation);
        int64_t count;
        while ((count = raster.enter_block()) > 0)
        {
            int64_t at = raster.position();
            for (int64_t from = 0; from < count;)
            {
                const int64_t n = std::min(count - from, current.end - at);
                block_sums(raster, count, from, n);
                spectra.add_from(current.sums, at, part);
                at += n;
                from += n;
                if (at == current.end)
                    close_interval();
            }
            raster.skip(count);
        }

        samples = raster.position();
        if (samples > 0)
        {
            const int64_t windows =
                samples <= width ? 1 : (samples - width + step - 1) / step + 1;
            current.end = samples;
            done.push_back(std::move(current));
            while (next < windows)
                judge(next++);
            out.windows = windows;
        }
        if (options.keep_spectrum >= 0 && options.keep_spectrum < out.windows)
            keep_spectrum(seq, options, width, step, length, taper, scale, out);
        return out;
    }

} // namespace pulseq
