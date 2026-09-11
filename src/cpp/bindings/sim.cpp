// Relaxation-free Bloch simulation in hard-pulse steps. Each position's steps
// compose as SU(2) elements and are converted to one 3x3 rotation at the end.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "sim.h"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;

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

    /**
     * The active rotation, row-major, of ``U = [[a, -conj(b)], [b, conj(a)]]``:
     * ``R_ij = Tr(sigma_i U sigma_j U^H) / 2``.
     */
    void rotation_of(Complex a, Complex b, double* r)
    {
        const Complex i(0.0, 1.0);
        const Complex u[2][2] = {{a, -std::conj(b)}, {b, std::conj(a)}};
        const Complex sigma[3][2][2] = {
            {{0.0, 1.0}, {1.0, 0.0}},
            {{0.0, -i}, {i, 0.0}},
            {{1.0, 0.0}, {0.0, -1.0}},
        };
        for (int j = 0; j < 3; ++j)
        {
            Complex us[2][2] = {};
            for (int p = 0; p < 2; ++p)
                for (int q = 0; q < 2; ++q)
                    for (int k = 0; k < 2; ++k)
                        us[p][q] += u[p][k] * sigma[j][k][q];
            Complex v[2][2] = {};
            for (int p = 0; p < 2; ++p)
                for (int q = 0; q < 2; ++q)
                    for (int k = 0; k < 2; ++k)
                        v[p][q] += us[p][k] * std::conj(u[q][k]);
            for (int row = 0; row < 3; ++row)
            {
                Complex trace = 0.0;
                for (int p = 0; p < 2; ++p)
                    for (int k = 0; k < 2; ++k)
                        trace += sigma[row][p][k] * v[k][p];
                r[row * 3 + j] = 0.5 * trace.real();
            }
        }
    }

    /**
     * Net rotation at each position, written row-major to ``out`` as
     * ``(positions, 3, 3)``. Step ``k`` turns right-handedly about
     * ``2 pi dt (Re b1, Im b1, bz)`` by that vector's length; a step with no
     * field is skipped. ``b1_rows == 1`` shares one b1 row across positions;
     * ``bz_cols == 1`` holds bz constant over the steps.
     */
    void rotations(const Complex* b1,
                   size_t b1_rows,
                   const double* bz,
                   size_t bz_cols,
                   size_t positions,
                   size_t steps,
                   double dt,
                   double* out,
                   size_t threads)
    {
        const double scale = kTwoPi * dt;
        parallel(positions, threads, [&](size_t first, size_t last) {
            for (size_t p = first; p < last; ++p)
            {
                const Complex* field = b1 + (b1_rows == 1 ? 0 : p * steps);
                const double* along = bz + p * bz_cols;
                Complex a(1.0, 0.0);
                Complex b(0.0, 0.0);
                for (size_t k = 0; k < steps; ++k)
                {
                    const double wx = scale * field[k].real();
                    const double wy = scale * field[k].imag();
                    const double wz = scale * along[bz_cols == 1 ? 0 : k];
                    const double angle = std::sqrt(wx * wx + wy * wy + wz * wz);
                    if (angle <= 1e-15)
                    {
                        continue;
                    }
                    const double c = std::cos(0.5 * angle);
                    const double s = std::sin(0.5 * angle) / angle;
                    const Complex step_a(c, -s * wz);
                    const Complex step_b(s * wy, -s * wx);
                    const Complex next_a = step_a * a - std::conj(step_b) * b;
                    const Complex next_b = step_b * a + std::conj(step_a) * b;
                    a = next_a;
                    b = next_b;
                }
                rotation_of(a, b, out + 9 * p);
            }
        });
    }

} // namespace

void pypulseqpp_bind_sim(py::module_& module)
{
    module.doc() = "Compiled Bloch simulation";
    module.def(
        "rotations",
        [](py::array_t<Complex, py::array::c_style | py::array::forcecast> b1,
           py::array_t<double, py::array::c_style | py::array::forcecast> bz,
           double dt,
           size_t threads) {
            if (bz.ndim() != 2)
            {
                throw std::invalid_argument("bz must be (positions, steps) or (positions, 1)");
            }
            const size_t positions = static_cast<size_t>(bz.shape(0));
            size_t rows = 1;
            size_t steps = 0;
            if (b1.ndim() == 1)
            {
                steps = static_cast<size_t>(b1.shape(0));
            }
            else if (b1.ndim() == 2)
            {
                rows = static_cast<size_t>(b1.shape(0));
                steps = static_cast<size_t>(b1.shape(1));
            }
            else
            {
                throw std::invalid_argument("b1 must be (steps,) or (positions, steps)");
            }
            if (rows != 1 && rows != positions)
            {
                throw std::invalid_argument("b1 holds fields for a different number of positions than bz");
            }
            const size_t columns = static_cast<size_t>(bz.shape(1));
            if (columns != 1 && columns != steps)
            {
                throw std::invalid_argument("bz must hold one value per step, or one throughout");
            }
            py::array_t<double> out({static_cast<py::ssize_t>(positions), py::ssize_t{3}, py::ssize_t{3}});
            const Complex* field = b1.data();
            const double* along = bz.data();
            double* result = out.mutable_data();
            {
                py::gil_scoped_release release;
                rotations(field, rows, along, columns, positions, steps, dt, result, threads);
            }
            return out;
        },
        py::arg("b1"),
        py::arg("bz"),
        py::arg("dt"),
        py::arg("threads") = 0,
        R"doc(Return each position's net rotation, (positions, 3, 3), over a field in Hz.

Step k turns right-handedly about 2 pi dt (Re b1[k], Im b1[k], bz[k]) by that
vector's length. b1 is (steps,) shared by every position or (positions, steps);
bz is (positions, steps) or (positions, 1). threads = 0 uses every core.
)doc");
}
