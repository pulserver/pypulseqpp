// Shinnar-Le Roux design kernels. The root-flip search follows SigPy's
// `sigpy.mri.rf.slr.root_flip` (BSD 3-Clause; see
// LICENSES/SigPy-BSD-3-Clause.txt). Each candidate is built and inverted as
// `pypulseqpp._slr._flipped_pulse` does, step for step.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "slr.h"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;

    constexpr double kPi = 3.14159265358979323846;

    /// Most flippable roots accepted; the search tries all 2^count patterns.
    constexpr std::size_t kMaxCandidates = 30;

    /// Radix-2 FFT of one fixed power-of-two size, with NumPy's sign and scaling.
    class Fft
    {
    public:
        explicit Fft(std::size_t size) : size_(size), reversed_(size), twiddle_(size / 2)
        {
            std::size_t bits = 0;
            while ((std::size_t{1} << bits) < size)
            {
                ++bits;
            }
            for (std::size_t i = 0; i < size; ++i)
            {
                std::size_t r = 0;
                for (std::size_t b = 0; b < bits; ++b)
                {
                    if (i & (std::size_t{1} << b))
                    {
                        r |= std::size_t{1} << (bits - 1 - b);
                    }
                }
                reversed_[i] = r;
            }
            for (std::size_t k = 0; k < size / 2; ++k)
            {
                twiddle_[k] = std::polar(
                    1.0, -2.0 * kPi * static_cast<double>(k) / static_cast<double>(size));
            }
        }

        std::size_t size() const { return size_; }

        void forward(std::vector<Complex>& x) const { transform(x, false); }

        void inverse(std::vector<Complex>& x) const
        {
            transform(x, true);
            const double scale = 1.0 / static_cast<double>(size_);
            for (auto& value : x)
            {
                value *= scale;
            }
        }

    private:
        void transform(std::vector<Complex>& x, bool conjugate) const
        {
            for (std::size_t i = 0; i < size_; ++i)
            {
                if (i < reversed_[i])
                {
                    std::swap(x[i], x[reversed_[i]]);
                }
            }
            for (std::size_t length = 2; length <= size_; length <<= 1)
            {
                const std::size_t half = length / 2;
                const std::size_t stride = size_ / length;
                for (std::size_t start = 0; start < size_; start += length)
                {
                    for (std::size_t j = 0; j < half; ++j)
                    {
                        const Complex w =
                            conjugate ? std::conj(twiddle_[j * stride]) : twiddle_[j * stride];
                        const Complex u = x[start + j];
                        const Complex v = x[start + j + half] * w;
                        x[start + j] = u + v;
                        x[start + j + half] = u - v;
                    }
                }
            }
        }

        std::size_t size_;
        std::vector<std::size_t> reversed_;
        std::vector<Complex> twiddle_;
    };

    /// Per-thread scratch; `roots` is rewritten for each pattern tried.
    struct Workspace
    {
        Workspace(std::size_t n, std::size_t padded, const std::vector<Complex>& roots)
            : roots(roots), beta(n), alpha(n), spectrum(padded), cepstrum(padded)
        {
        }

        std::vector<Complex> roots;
        std::vector<Complex> beta;
        std::vector<Complex> alpha;
        std::vector<Complex> spectrum;
        std::vector<Complex> cepstrum;
    };

    /// Monic polynomial with these roots, highest power first, as `np.poly`
    /// builds it, right-aligned in `coefficients`.
    void polynomial(const std::vector<Complex>& roots, std::vector<Complex>& coefficients)
    {
        std::fill(coefficients.begin(), coefficients.end(), Complex{});
        const std::size_t offset = coefficients.size() - roots.size() - 1;
        Complex* c = coefficients.data() + offset;
        c[0] = 1.0;
        for (std::size_t k = 0; k < roots.size(); ++k)
        {
            for (std::size_t j = k + 1; j > 0; --j)
            {
                c[j] -= roots[k] * c[j - 1];
            }
        }
    }

    /// Peak |rf|, in rad per sample, of the pulse whose beta has `w.roots`
    /// scaled to a peak |beta| of `target` over the spectrum. Overwrites the
    /// other buffers of `w`.
    double pulse_peak(double target, const Fft& fft, Workspace& w)
    {
        const std::size_t n = w.beta.size();
        const std::size_t padded = fft.size();
        polynomial(w.roots, w.beta);

        std::fill(w.spectrum.begin(), w.spectrum.end(), Complex{});
        std::copy(w.beta.begin(), w.beta.end(), w.spectrum.begin());
        fft.forward(w.spectrum);
        double peak = 0.0;
        for (const auto& value : w.spectrum)
        {
            peak = std::max(peak, std::abs(value));
        }
        const double scale = target / peak;
        for (auto& value : w.beta)
        {
            value *= scale;
        }

        // alpha is the minimum-phase polynomial with |alpha|^2 + |beta|^2 = 1:
        // the log magnitude's causal part, exponentiated.
        for (std::size_t i = 0; i < padded; ++i)
        {
            const double b = std::abs(w.spectrum[i]) * scale;
            const double magnitude = std::sqrt(std::max(0.0, 1.0 - b * b));
            w.cepstrum[i] = std::log(std::max(magnitude, std::numeric_limits<double>::min()));
        }
        fft.forward(w.cepstrum);
        const std::size_t half = padded / 2;
        for (std::size_t i = 1; i < half; ++i)
        {
            w.cepstrum[i] *= 2.0;
        }
        for (std::size_t i = half + 1; i < padded; ++i)
        {
            w.cepstrum[i] = 0.0;
        }
        fft.inverse(w.cepstrum);
        for (auto& value : w.cepstrum)
        {
            value = std::exp(value);
        }
        fft.forward(w.cepstrum);
        for (std::size_t i = 0; i < n; ++i)
        {
            w.alpha[i] = w.cepstrum[n - 1 - i] / static_cast<double>(padded);
        }

        // The inverse SLR transform: peel one hard pulse off the end at a time.
        double rf_peak = 0.0;
        std::vector<Complex>& a = w.alpha;
        std::vector<Complex>& b = w.beta;
        for (std::size_t index = n; index-- > 0;)
        {
            const Complex ratio = b[index] / a[index];
            const double cosine = std::sqrt(1.0 / (1.0 + std::norm(ratio)));
            const Complex sine = std::conj(cosine * ratio);
            rf_peak = std::max(rf_peak, 2.0 * std::atan2(std::abs(sine), cosine));
            for (std::size_t j = 0; j < index; ++j)
            {
                const Complex a_next = cosine * a[j + 1] + sine * b[j + 1];
                const Complex b_next = -std::conj(sine) * a[j] + cosine * b[j];
                a[j] = a_next;
                b[j] = b_next;
            }
        }
        return rf_peak;
    }

    struct Best
    {
        double peak = std::numeric_limits<double>::infinity();
        std::uint64_t pattern = 0;
    };

    /// The lowest-peak pattern among `first, first + stride, ...`; bit j flips
    /// root `candidates[j]`, and ties keep the lower pattern.
    Best search(const std::vector<Complex>& roots,
                const std::vector<std::size_t>& candidates,
                double target,
                std::size_t n,
                const Fft& fft,
                std::uint64_t first,
                std::uint64_t stride)
    {
        Workspace w(n, fft.size(), roots);
        Best best;
        const std::uint64_t count = std::uint64_t{1} << candidates.size();
        for (std::uint64_t pattern = first; pattern < count; pattern += stride)
        {
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const Complex r = roots[candidates[j]];
                w.roots[candidates[j]] = (pattern >> j) & 1u ? 1.0 / std::conj(r) : r;
            }
            const double peak = pulse_peak(target, fft, w);
            if (peak < best.peak)
            {
                best = {peak, pattern};
            }
        }
        return best;
    }

    std::size_t padded_length(std::size_t n)
    {
        std::size_t padded = 1;
        while (padded < 16 * n)
        {
            padded <<= 1;
        }
        return padded;
    }

} // namespace

void pypulseqpp_bind_slr(py::module_& module)
{
    module.doc() = "Compiled Shinnar-Le Roux design kernels";
    module.def(
        "root_flip_search",
        [](py::array_t<Complex, py::array::c_style | py::array::forcecast> roots,
           py::array_t<bool, py::array::c_style | py::array::forcecast> candidates,
           double target,
           std::size_t n,
           std::size_t threads)
        {
            if (roots.ndim() != 1 || candidates.ndim() != 1 || roots.size() != candidates.size())
            {
                throw std::invalid_argument("roots and candidates must be 1-D and of one length");
            }
            if (!(target > 0.0 && target < 1.0))
            {
                throw std::invalid_argument("target must lie in (0, 1)");
            }
            const auto count = static_cast<std::size_t>(roots.size());
            if (n < count + 1)
            {
                throw std::invalid_argument("n must exceed the number of roots");
            }
            std::vector<Complex> root_values(roots.data(), roots.data() + count);
            std::vector<std::size_t> flippable;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (candidates.data()[i])
                {
                    flippable.push_back(i);
                }
            }
            if (flippable.size() > kMaxCandidates)
            {
                throw std::invalid_argument("too many flippable roots for an exhaustive search");
            }

            Best best;
            {
                py::gil_scoped_release release;
                const Fft fft(padded_length(n));
                const std::uint64_t patterns = std::uint64_t{1} << flippable.size();
                std::size_t workers = threads ? threads : std::thread::hardware_concurrency();
                workers = static_cast<std::size_t>(
                    std::max<std::uint64_t>(1, std::min<std::uint64_t>(workers, patterns)));
                std::vector<Best> found(workers);
                std::vector<std::thread> pool;
                for (std::size_t t = 1; t < workers; ++t)
                {
                    pool.emplace_back(
                        [&, t]
                        { found[t] = search(root_values, flippable, target, n, fft, t, workers); });
                }
                found[0] = search(root_values, flippable, target, n, fft, 0, workers);
                for (auto& worker : pool)
                {
                    worker.join();
                }
                // Lowest peak, and the lowest pattern among equal peaks, so the
                // answer does not depend on how the patterns were shared out.
                for (const auto& candidate : found)
                {
                    if (candidate.peak < best.peak ||
                        (candidate.peak == best.peak && candidate.pattern < best.pattern))
                    {
                        best = candidate;
                    }
                }
            }

            py::array_t<bool> flips(static_cast<py::ssize_t>(count));
            auto* output = flips.mutable_data();
            std::fill(output, output + count, false);
            for (std::size_t j = 0; j < flippable.size(); ++j)
            {
                output[flippable[j]] = ((best.pattern >> j) & 1u) != 0;
            }
            return py::make_tuple(flips, best.peak);
        },
        py::arg("roots"),
        py::arg("candidates"),
        py::arg("target"),
        py::arg("n"),
        py::arg("threads") = 0,
        R"doc(Find which roots of an SLR beta polynomial to flip for the lowest peak RF.

Every subset of the flippable roots is tried. Flipping root r to 1/conj(r)
keeps the shape of |beta| on the unit circle (the slice profile) and changes
only its phase, and so how the pulse's energy spreads in time. Among equal
peaks the lowest bit pattern wins, so the result does not depend on threads.

Parameters
----------
roots : complex array
    Roots of beta, Leja-ordered so the polynomial rebuilds accurately.
candidates : bool array
    Which roots may flip.
target : float
    Peak |beta| over the spectrum each candidate is scaled to.
n : int
    Samples in the pulse.
threads : int, optional
    Workers; 0 uses every core.

Returns
-------
flips : bool array
    Which roots the best pulse flips.
peak : float
    Its peak |rf|, in radians per sample.
)doc");
}
