// Small-tip parallel-transmit design kernels: the spatial-domain method of
// Grissom et al. (Magn Reson Med 56:620, 2006) solved by conjugate gradients
// with magnitude-least-squares phase updates, RF shimming by magnitude least
// squares, and greedy spokes selection (Grissom et al., Magn Reson Med
// 68:1553, 2012). They follow SigPy's sigpy.mri.rf.ptx and shim (BSD 3-Clause;
// see LICENSES/SigPy-BSD-3-Clause.txt).
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ptx.h"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;
    using Vector = std::vector<Complex>;
    using ComplexArray = py::array_t<Complex, py::array::c_style | py::array::forcecast>;
    using RealArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

    constexpr double kTwoPi = 6.283185307179586476925286766559;

    /// Run ``body(first, last)`` over ``[0, count)``, one contiguous chunk per worker.
    void parallel(size_t count, size_t workers, const std::function<void(size_t, size_t)>& body)
    {
        workers = workers ? workers : std::thread::hardware_concurrency();
        workers = std::max<size_t>(1, std::min(workers, count));
        if (workers == 1)
        {
            body(0, count);
            return;
        }
        const size_t chunk = (count + workers - 1) / workers;
        std::vector<std::thread> pool;
        for (size_t first = chunk; first < count; first += chunk)
        {
            pool.emplace_back(body, first, std::min(count, first + chunk));
        }
        body(0, std::min(count, chunk));
        for (auto& worker : pool)
        {
            worker.join();
        }
    }

    double squared_norm(const Vector& values)
    {
        double sum = 0.0;
        for (const auto& value : values)
        {
            sum += std::norm(value);
        }
        return sum;
    }

    Complex phasor(const Complex& value)
    {
        const double magnitude = std::abs(value);
        return magnitude > 0.0 ? value / magnitude : Complex(1.0, 0.0);
    }

    /**
     * The small-tip model. Position ``s`` ends with transverse magnetisation
     * ``-i scale sum_c S[c,s] sum_j b[c,j] exp(-2 pi i (x_s . k_j + df_s (t_j - end)))``,
     * ``k_j`` being the gradient moment still to come after sample ``j``.
     */
    struct Model
    {
        size_t channels = 0;
        size_t samples = 0;
        size_t positions = 0;
        size_t dims = 0;
        Vector sens;            // (channels, positions)
        std::vector<double> x;  // (positions, dims), m
        std::vector<double> k;  // (samples, dims), cycles/m
        std::vector<double> df; // (positions), Hz; empty for none
        std::vector<double> t;  // (samples), s
        double end = 0.0;
        double scale = 1.0;
        size_t threads = 0;

        Complex phase(size_t s, size_t j) const
        {
            double cycles = 0.0;
            for (size_t d = 0; d < dims; ++d)
            {
                cycles += x[s * dims + d] * k[j * dims + d];
            }
            if (!df.empty())
            {
                cycles += df[s] * (t[j] - end);
            }
            return std::polar(1.0, -kTwoPi * cycles);
        }

        void forward(const Vector& b, Vector& m) const
        {
            parallel(positions, threads, [&](size_t first, size_t last) {
                for (size_t s = first; s < last; ++s)
                {
                    Complex sum = 0.0;
                    for (size_t j = 0; j < samples; ++j)
                    {
                        Complex field = 0.0;
                        for (size_t c = 0; c < channels; ++c)
                        {
                            field += sens[c * positions + s] * b[c * samples + j];
                        }
                        sum += field * phase(s, j);
                    }
                    m[s] = Complex(0.0, -scale) * sum;
                }
            });
        }

        void adjoint(const Vector& m, Vector& b) const
        {
            parallel(samples, threads, [&](size_t first, size_t last) {
                Vector sum(channels);
                for (size_t j = first; j < last; ++j)
                {
                    std::fill(sum.begin(), sum.end(), Complex(0.0));
                    for (size_t s = 0; s < positions; ++s)
                    {
                        const Complex seen = std::conj(phase(s, j)) * m[s];
                        for (size_t c = 0; c < channels; ++c)
                        {
                            sum[c] += std::conj(sens[c * positions + s]) * seen;
                        }
                    }
                    for (size_t c = 0; c < channels; ++c)
                    {
                        b[c * samples + j] = Complex(0.0, scale) * sum[c];
                    }
                }
            });
        }
    };

    /// ``A^H W A + lambda I``, applied without forming it.
    struct Normal
    {
        const Model& model;
        const std::vector<double>& weight;
        double lambda;
        mutable Vector scratch;

        void apply(const Vector& b, Vector& out) const
        {
            model.forward(b, scratch);
            for (size_t s = 0; s < scratch.size(); ++s)
            {
                scratch[s] *= weight[s];
            }
            model.adjoint(scratch, out);
            for (size_t i = 0; i < out.size(); ++i)
            {
                out[i] += lambda * b[i];
            }
        }
    };

    /// Conjugate gradients on the normal equations, warm-started from ``b``.
    void conjugate_gradient(
        const Normal& normal, const Vector& rhs, Vector& b, int iterations, double tolerance)
    {
        Vector residual(b.size());
        Vector product(b.size());
        normal.apply(b, product);
        for (size_t i = 0; i < b.size(); ++i)
        {
            residual[i] = rhs[i] - product[i];
        }
        Vector direction = residual;
        double current = squared_norm(residual);
        const double stop =
            tolerance * tolerance * std::max(squared_norm(rhs), std::numeric_limits<double>::min());
        for (int iteration = 0; iteration < iterations && current > stop; ++iteration)
        {
            normal.apply(direction, product);
            double curvature = 0.0;
            for (size_t i = 0; i < b.size(); ++i)
            {
                curvature += std::real(std::conj(direction[i]) * product[i]);
            }
            if (!(curvature > 0.0))
            {
                break;
            }
            const double step = current / curvature;
            for (size_t i = 0; i < b.size(); ++i)
            {
                b[i] += step * direction[i];
                residual[i] -= step * product[i];
            }
            const double next = squared_norm(residual);
            const double ratio = next / current;
            for (size_t i = 0; i < b.size(); ++i)
            {
                direction[i] = residual[i] + ratio * direction[i];
            }
            current = next;
        }
    }

    /**
     * Least squares against ``target`` under the model, with the target's
     * phase exchanged for the achieved one ``phase_updates`` times: magnitude
     * least squares when the phase of the profile is free.
     */
    Vector design(
        const Model& model,
        const Vector& target,
        const std::vector<double>& weight,
        double lambda,
        int iterations,
        double tolerance,
        int phase_updates,
        Vector b)
    {
        const Normal normal{model, weight, lambda, Vector(model.positions)};
        Vector wanted = target;
        Vector weighted(model.positions);
        Vector achieved(model.positions);
        Vector rhs(b.size());
        for (int round = 0; round <= phase_updates; ++round)
        {
            if (round > 0)
            {
                model.forward(b, achieved);
                for (size_t s = 0; s < model.positions; ++s)
                {
                    wanted[s] = std::abs(target[s]) * phasor(achieved[s]);
                }
            }
            for (size_t s = 0; s < model.positions; ++s)
            {
                weighted[s] = weight[s] * wanted[s];
            }
            model.adjoint(weighted, rhs);
            conjugate_gradient(normal, rhs, b, iterations, tolerance);
        }
        return b;
    }

    /// Solve ``H u = r`` for a Hermitian positive-definite ``n x n`` ``H`` by Cholesky.
    Vector solve_hermitian(Vector h, Vector r, size_t n)
    {
        double trace = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            trace += h[i * n + i].real();
        }
        // A semi-definite system -- a channel that reaches nothing -- is
        // steadied by a trace-relative ridge too small to move a full one.
        const double ridge = 1e-12 * std::max(trace / static_cast<double>(n), 1e-300);
        for (size_t i = 0; i < n; ++i)
        {
            h[i * n + i] += ridge;
        }
        for (size_t j = 0; j < n; ++j)
        {
            double diagonal = h[j * n + j].real();
            for (size_t k = 0; k < j; ++k)
            {
                diagonal -= std::norm(h[j * n + k]);
            }
            if (!(diagonal > 0.0))
            {
                throw std::runtime_error("the normal equations are not positive definite");
            }
            const double root = std::sqrt(diagonal);
            h[j * n + j] = root;
            for (size_t i = j + 1; i < n; ++i)
            {
                Complex sum = h[i * n + j];
                for (size_t k = 0; k < j; ++k)
                {
                    sum -= h[i * n + k] * std::conj(h[j * n + k]);
                }
                h[i * n + j] = sum / root;
            }
        }
        for (size_t i = 0; i < n; ++i)
        {
            Complex sum = r[i];
            for (size_t k = 0; k < i; ++k)
            {
                sum -= h[i * n + k] * r[k];
            }
            r[i] = sum / h[i * n + i].real();
        }
        for (size_t i = n; i-- > 0;)
        {
            Complex sum = r[i];
            for (size_t k = i + 1; k < n; ++k)
            {
                sum -= std::conj(h[k * n + i]) * r[k];
            }
            r[i] = sum / h[i * n + i].real();
        }
        return r;
    }

    /// A dense system matrix, column-major: ``data[column * rows + row]``.
    struct Columns
    {
        size_t rows = 0;
        size_t cols = 0;
        Vector data;

        Vector apply(const Vector& u) const
        {
            Vector out(rows);
            for (size_t c = 0; c < cols; ++c)
            {
                for (size_t r = 0; r < rows; ++r)
                {
                    out[r] += data[c * rows + r] * u[c];
                }
            }
            return out;
        }

        /// Weighted least squares with a ridge: ``(A^H W A + lambda I) u = A^H W y``.
        Vector fit(const std::vector<double>& weight, const Vector& y, double lambda) const
        {
            Vector h(cols * cols);
            Vector rhs(cols);
            for (size_t i = 0; i < cols; ++i)
            {
                const Complex* column_i = &data[i * rows];
                for (size_t j = 0; j <= i; ++j)
                {
                    const Complex* column_j = &data[j * rows];
                    Complex sum = 0.0;
                    for (size_t r = 0; r < rows; ++r)
                    {
                        sum += std::conj(column_i[r]) * weight[r] * column_j[r];
                    }
                    h[i * cols + j] = sum;
                    h[j * cols + i] = std::conj(sum);
                }
                h[i * cols + i] += lambda;
                Complex sum = 0.0;
                for (size_t r = 0; r < rows; ++r)
                {
                    sum += std::conj(column_i[r]) * weight[r] * y[r];
                }
                rhs[i] = sum;
            }
            return solve_hermitian(std::move(h), std::move(rhs), cols);
        }

        double cost(const std::vector<double>& weight, const Vector& y, const Vector& u, double lambda)
            const
        {
            const Vector achieved = apply(u);
            double sum = lambda * squared_norm(u);
            for (size_t r = 0; r < rows; ++r)
            {
                sum += weight[r] * std::norm(achieved[r] - y[r]);
            }
            return sum;
        }
    };

    struct MagnitudeFit
    {
        Vector u;
        Vector target; // the magnitude at the phase finally settled on
    };

    /// Magnitude least squares by variable exchange, from the phases in ``seed``.
    MagnitudeFit magnitude_fit(
        const Columns& a,
        const std::vector<double>& weight,
        const std::vector<double>& magnitude,
        const Vector& seed,
        double lambda,
        double tolerance,
        int rounds)
    {
        MagnitudeFit found;
        found.target.resize(a.rows);
        for (size_t r = 0; r < a.rows; ++r)
        {
            found.target[r] = magnitude[r] * phasor(seed[r]);
        }
        found.u = a.fit(weight, found.target, lambda);
        double cost = a.cost(weight, found.target, found.u, lambda);
        for (int round = 0; round < rounds; ++round)
        {
            const Vector achieved = a.apply(found.u);
            for (size_t r = 0; r < a.rows; ++r)
            {
                found.target[r] = magnitude[r] * phasor(achieved[r]);
            }
            found.u = a.fit(weight, found.target, lambda);
            const double next = a.cost(weight, found.target, found.u, lambda);
            const bool settled = std::fabs(cost - next) <= tolerance * cost;
            cost = next;
            if (settled)
            {
                break;
            }
        }
        return found;
    }

    /// Columns ``S[c,s] exp(-2 pi i x_s . k_j)``, spoke-major: column ``j * channels + c``.
    Columns spoke_columns(
        const Vector& sens,
        const std::vector<double>& x,
        size_t channels,
        size_t positions,
        const std::vector<std::array<double, 2>>& spokes)
    {
        Columns a;
        a.rows = positions;
        a.cols = spokes.size() * channels;
        a.data.resize(a.rows * a.cols);
        for (size_t j = 0; j < spokes.size(); ++j)
        {
            for (size_t s = 0; s < positions; ++s)
            {
                const double cycles = x[2 * s] * spokes[j][0] + x[2 * s + 1] * spokes[j][1];
                const Complex turn = std::polar(1.0, -kTwoPi * cycles);
                for (size_t c = 0; c < channels; ++c)
                {
                    a.data[(j * channels + c) * positions + s] = sens[c * positions + s] * turn;
                }
            }
        }
        return a;
    }

    // -- argument handling ---------------------------------------------------

    Vector complex_values(const ComplexArray& array)
    {
        return Vector(array.data(), array.data() + array.size());
    }

    std::vector<double> real_values(const RealArray& array)
    {
        return std::vector<double>(array.data(), array.data() + array.size());
    }

    void require(bool condition, const char* message)
    {
        if (!condition)
        {
            throw std::invalid_argument(message);
        }
    }

    Model make_model(
        const ComplexArray& sens,
        const RealArray& positions,
        const RealArray& kspace,
        const RealArray& times,
        double end,
        double scale,
        const RealArray& off_resonance,
        size_t threads)
    {
        require(sens.ndim() == 2, "sens must be (channels, positions)");
        require(positions.ndim() == 2 && positions.shape(0) == sens.shape(1),
                "positions must be (positions, dims), one row per column of sens");
        require(kspace.ndim() == 2 && kspace.shape(1) == positions.shape(1),
                "kspace must be (samples, dims) in the dimensions of positions");
        require(times.ndim() == 1 && times.shape(0) == kspace.shape(0),
                "times must hold one time per kspace sample");
        require(off_resonance.size() == 0 || off_resonance.size() == positions.shape(0),
                "off_resonance must be empty or one value per position");
        Model model;
        model.channels = static_cast<size_t>(sens.shape(0));
        model.positions = static_cast<size_t>(sens.shape(1));
        model.samples = static_cast<size_t>(kspace.shape(0));
        model.dims = static_cast<size_t>(positions.shape(1));
        model.sens = complex_values(sens);
        model.x = real_values(positions);
        model.k = real_values(kspace);
        model.t = real_values(times);
        model.df = real_values(off_resonance);
        model.end = end;
        model.scale = scale;
        model.threads = threads;
        return model;
    }

    ComplexArray as_array(const Vector& values, std::vector<py::ssize_t> shape)
    {
        ComplexArray out(shape);
        std::copy(values.begin(), values.end(), out.mutable_data());
        return out;
    }

    const char* kModelDoc = R"doc(
The small-tip model: position s ends with transverse magnetisation
-i scale sum_c sens[c,s] sum_j b[c,j] exp(-2 pi i (positions[s] . kspace[j] + off_resonance[s] (times[j] - end))),
kspace[j] being the gradient moment still to come after sample j, in cycles/m,
and positions in m. With b in Hz and scale = 2 pi dwell, the result is in the
Bloch simulator's Mx + i My.
)doc";

} // namespace

void pypulseqpp_bind_ptx(py::module_& module)
{
    module.doc() = "Compiled small-tip parallel-transmit design kernels";
    const RealArray none(0);

    module.def(
        "forward",
        [](const ComplexArray& b, const ComplexArray& sens, const RealArray& positions,
           const RealArray& kspace, const RealArray& times, double end, double scale,
           const RealArray& off_resonance, size_t threads) {
            const Model model =
                make_model(sens, positions, kspace, times, end, scale, off_resonance, threads);
            require(b.size() == static_cast<py::ssize_t>(model.channels * model.samples),
                    "b must be (channels, samples)");
            const Vector values = complex_values(b);
            Vector m(model.positions);
            {
                py::gil_scoped_release release;
                model.forward(values, m);
            }
            return as_array(m, {static_cast<py::ssize_t>(model.positions)});
        },
        py::arg("b"), py::arg("sens"), py::arg("positions"), py::arg("kspace"), py::arg("times"),
        py::arg("end"), py::arg("scale"), py::arg("off_resonance") = none, py::arg("threads") = 0,
        (std::string("Apply the small-tip model to per-channel waveforms.\n") + kModelDoc).c_str());

    module.def(
        "adjoint",
        [](const ComplexArray& m, const ComplexArray& sens, const RealArray& positions,
           const RealArray& kspace, const RealArray& times, double end, double scale,
           const RealArray& off_resonance, size_t threads) {
            const Model model =
                make_model(sens, positions, kspace, times, end, scale, off_resonance, threads);
            require(m.size() == static_cast<py::ssize_t>(model.positions),
                    "m must hold one value per position");
            const Vector values = complex_values(m);
            Vector b(model.channels * model.samples);
            {
                py::gil_scoped_release release;
                model.adjoint(values, b);
            }
            return as_array(b, {static_cast<py::ssize_t>(model.channels),
                                static_cast<py::ssize_t>(model.samples)});
        },
        py::arg("m"), py::arg("sens"), py::arg("positions"), py::arg("kspace"), py::arg("times"),
        py::arg("end"), py::arg("scale"), py::arg("off_resonance") = none, py::arg("threads") = 0,
        "Apply the adjoint of the small-tip model.");

    module.def(
        "design",
        [](const ComplexArray& target, const RealArray& weight, const ComplexArray& sens,
           const RealArray& positions, const RealArray& kspace, const RealArray& times, double end,
           double scale, const RealArray& off_resonance, double regularization, int iterations,
           double tolerance, int phase_updates, const ComplexArray& initial, size_t threads) {
            const Model model =
                make_model(sens, positions, kspace, times, end, scale, off_resonance, threads);
            require(target.size() == static_cast<py::ssize_t>(model.positions)
                        && weight.size() == target.size(),
                    "target and weight must hold one value per position");
            require(regularization >= 0.0 && iterations >= 0 && phase_updates >= 0,
                    "regularization, iterations and phase_updates must be non-negative");
            const size_t unknowns = model.channels * model.samples;
            require(initial.size() == 0 || initial.size() == static_cast<py::ssize_t>(unknowns),
                    "initial must be empty or (channels, samples)");
            const Vector wanted = complex_values(target);
            const std::vector<double> weights = real_values(weight);
            Vector b = initial.size() ? complex_values(initial) : Vector(unknowns);
            {
                py::gil_scoped_release release;
                b = design(model, wanted, weights, regularization, iterations, tolerance,
                           phase_updates, std::move(b));
            }
            return as_array(b, {static_cast<py::ssize_t>(model.channels),
                                static_cast<py::ssize_t>(model.samples)});
        },
        py::arg("target"), py::arg("weight"), py::arg("sens"), py::arg("positions"),
        py::arg("kspace"), py::arg("times"), py::arg("end"), py::arg("scale"),
        py::arg("off_resonance") = none, py::arg("regularization") = 0.0,
        py::arg("iterations") = 30, py::arg("tolerance") = 1e-6, py::arg("phase_updates") = 0,
        py::arg("initial") = ComplexArray(0), py::arg("threads") = 0,
        R"doc(Return per-channel waveforms (channels, samples) matching a target under the model.

Minimises sum_s weight[s] |A b - target|^2 + regularization |b|^2 by conjugate
gradients, ``iterations`` at a time; with ``phase_updates`` > 0 the target's
phase is then exchanged for the achieved one that many times, each round
warm-started from the last -- magnitude least squares.
)doc");

    module.def(
        "rf_shim",
        [](const ComplexArray& sens, const RealArray& magnitude, const RealArray& weight,
           double regularization, double tolerance, int rounds) {
            require(sens.ndim() == 2, "sens must be (channels, positions)");
            const size_t channels = static_cast<size_t>(sens.shape(0));
            const size_t positions = static_cast<size_t>(sens.shape(1));
            require(magnitude.size() == static_cast<py::ssize_t>(positions)
                        && weight.size() == magnitude.size(),
                    "magnitude and weight must hold one value per position");
            Columns a;
            a.rows = positions;
            a.cols = channels;
            a.data = complex_values(sens); // row c of sens is column c
            const std::vector<double> wanted = real_values(magnitude);
            const std::vector<double> weights = real_values(weight);
            MagnitudeFit found;
            {
                py::gil_scoped_release release;
                // Start from the most efficient drive: the leading eigenvector
                // of A^H W A, by power iteration.
                Vector h(channels * channels);
                for (size_t i = 0; i < channels; ++i)
                {
                    for (size_t j = 0; j < channels; ++j)
                    {
                        Complex sum = 0.0;
                        for (size_t r = 0; r < positions; ++r)
                        {
                            sum += std::conj(a.data[i * positions + r]) * weights[r]
                                   * a.data[j * positions + r];
                        }
                        h[i * channels + j] = sum;
                    }
                }
                Vector v(channels, Complex(1.0, 0.0));
                for (int iteration = 0; iteration < 200; ++iteration)
                {
                    Vector next(channels);
                    for (size_t i = 0; i < channels; ++i)
                    {
                        for (size_t j = 0; j < channels; ++j)
                        {
                            next[i] += h[i * channels + j] * v[j];
                        }
                    }
                    const double size = std::sqrt(squared_norm(next));
                    if (!(size > 0.0))
                    {
                        break;
                    }
                    for (auto& value : next)
                    {
                        value /= size;
                    }
                    v = std::move(next);
                }
                found = magnitude_fit(a, weights, wanted, a.apply(v), regularization, tolerance,
                                      rounds);
            }
            return as_array(found.u, {static_cast<py::ssize_t>(channels)});
        },
        py::arg("sens"), py::arg("magnitude"), py::arg("weight"), py::arg("regularization") = 0.0,
        py::arg("tolerance") = 1e-6, py::arg("rounds") = 100,
        R"doc(Return per-channel weights whose combined field has the magnitude asked for.

Magnitude least squares on sum_c weights[c] sens[c,s], started from the leading
eigenvector of the weighted normal matrix and stopped when the cost moves by
less than ``tolerance`` of itself.
)doc");

    module.def(
        "spokes",
        [](const ComplexArray& sens, const RealArray& positions, const RealArray& magnitude,
           const RealArray& weight, const RealArray& candidates, int count, double regularization,
           double tolerance, int rounds, size_t threads) {
            require(sens.ndim() == 2, "sens must be (channels, positions)");
            const size_t channels = static_cast<size_t>(sens.shape(0));
            const size_t points = static_cast<size_t>(sens.shape(1));
            require(positions.ndim() == 2 && positions.shape(0) == sens.shape(1)
                        && positions.shape(1) == 2,
                    "positions must be (positions, 2)");
            require(magnitude.size() == static_cast<py::ssize_t>(points)
                        && weight.size() == magnitude.size(),
                    "magnitude and weight must hold one value per position");
            require(candidates.ndim() == 2 && candidates.shape(1) == 2,
                    "candidates must be (count, 2)");
            require(count >= 1 && static_cast<py::ssize_t>(count) <= candidates.shape(0) + 1,
                    "count must lie between 1 and one more than the candidates");
            const Vector s = complex_values(sens);
            const std::vector<double> x = real_values(positions);
            const std::vector<double> wanted = real_values(magnitude);
            const std::vector<double> weights = real_values(weight);
            const std::vector<double> grid = real_values(candidates);
            const size_t offered = static_cast<size_t>(candidates.shape(0));

            std::vector<std::array<double, 2>> chosen = {{0.0, 0.0}};
            MagnitudeFit found;
            {
                py::gil_scoped_release release;
                std::vector<bool> taken(offered, false);
                Vector seed(points, Complex(1.0, 0.0));
                for (size_t round = 0;; ++round)
                {
                    const Columns a = spoke_columns(s, x, channels, points, chosen);
                    found = magnitude_fit(a, weights, wanted, seed, regularization, tolerance,
                                          rounds);
                    seed = found.target;
                    if (chosen.size() == static_cast<size_t>(count))
                    {
                        break;
                    }
                    const Vector achieved = a.apply(found.u);
                    Vector residual(points);
                    for (size_t r = 0; r < points; ++r)
                    {
                        residual[r] = found.target[r] - achieved[r];
                    }
                    // The spoke that would carry most of what is left.
                    std::vector<double> score(offered, -1.0);
                    parallel(offered, threads, [&](size_t first, size_t last) {
                        for (size_t i = first; i < last; ++i)
                        {
                            if (taken[i])
                            {
                                continue;
                            }
                            const Columns single = spoke_columns(
                                s, x, channels, points, {{grid[2 * i], grid[2 * i + 1]}});
                            score[i] = std::sqrt(squared_norm(single.fit(weights, residual, 0.0)));
                        }
                    });
                    const size_t best = static_cast<size_t>(
                        std::max_element(score.begin(), score.end()) - score.begin());
                    taken[best] = true;
                    const std::array<double, 2> spoke = {grid[2 * best], grid[2 * best + 1]};
                    // Alternate ends, so the trajectory grows outward from the centre.
                    if (round % 2 != 0)
                    {
                        chosen.push_back(spoke);
                    }
                    else
                    {
                        chosen.insert(chosen.begin(), spoke);
                    }
                }
            }
            RealArray where({static_cast<py::ssize_t>(chosen.size()), py::ssize_t{2}});
            for (size_t j = 0; j < chosen.size(); ++j)
            {
                where.mutable_data()[2 * j] = chosen[j][0];
                where.mutable_data()[2 * j + 1] = chosen[j][1];
            }
            // Spoke-major columns come back as (channels, spokes).
            ComplexArray amounts({static_cast<py::ssize_t>(channels),
                                  static_cast<py::ssize_t>(chosen.size())});
            for (size_t j = 0; j < chosen.size(); ++j)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    amounts.mutable_data()[c * chosen.size() + j] = found.u[j * channels + c];
                }
            }
            return py::make_tuple(where, amounts);
        },
        py::arg("sens"), py::arg("positions"), py::arg("magnitude"), py::arg("weight"),
        py::arg("candidates"), py::arg("count"), py::arg("regularization") = 0.0,
        py::arg("tolerance") = 0.01, py::arg("rounds") = 100, py::arg("threads") = 0,
        R"doc(Choose spoke positions greedily and their per-channel weights by magnitude least squares.

Starts from a spoke at the k-space centre; each round adds the candidate whose
own least-squares fit to the residual is largest, alternately after and before
the spokes already chosen. Returns (kspace (count, 2), weights (channels,
count)): each weight is the rotation, in rad, the spoke's sub-pulse carries on
that channel.
)doc");
}
