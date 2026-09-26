// Relaxation-free Bloch simulation in hard-pulse steps. Each position's steps
// compose as SU(2) elements and are converted to one 3x3 rotation at the end.
// Beside it, the isochromats the Bloch equation carries from block to block.
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

#include "pulseq/bloch.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/simulate.hpp"
#include "pulseqpp_events.h"
#include "sim.h"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;
    using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using Complexes = py::array_t<Complex, py::array::c_style | py::array::forcecast>;

    std::vector<double> column(const Doubles& values, size_t count, const char* name)
    {
        if (values.ndim() != 1 || static_cast<size_t>(values.shape(0)) != count)
            throw std::invalid_argument(std::string(name) + " must hold one value per isochromat");
        return std::vector<double>(values.data(), values.data() + count);
    }

    /** Sensitivities as (isochromats, channels), or none. */
    std::vector<Complex> sensitivities(const py::object& given, size_t count, size_t& channels, const char* name)
    {
        channels = 0;
        if (given.is_none())
            return {};
        const Complexes values = py::cast<Complexes>(given);
        if (values.ndim() != 2 || static_cast<size_t>(values.shape(0)) != count || values.shape(1) < 1)
            throw std::invalid_argument(
                std::string(name) + " must be (isochromats, channels) with at least one channel");
        channels = static_cast<size_t>(values.shape(1));
        return std::vector<Complex>(values.data(), values.data() + values.size());
    }

    pulseq::Isochromats* make_isochromats(
        const Doubles& positions,
        const Doubles& proton_density,
        const Doubles& t1,
        const Doubles& t2,
        const Doubles& off_resonance,
        const py::object& transmit,
        const py::object& receive,
        size_t threads)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3)
            throw std::invalid_argument("positions must be (isochromats, 3)");
        const size_t count = static_cast<size_t>(positions.shape(0));
        pulseq::IsochromatProperties properties;
        properties.x.resize(count);
        properties.y.resize(count);
        properties.z.resize(count);
        const double* xyz = positions.data();
        for (size_t i = 0; i < count; ++i)
        {
            properties.x[i] = xyz[3 * i];
            properties.y[i] = xyz[3 * i + 1];
            properties.z[i] = xyz[3 * i + 2];
        }
        properties.proton_density = column(proton_density, count, "proton_density");
        properties.t1 = column(t1, count, "t1");
        properties.t2 = column(t2, count, "t2");
        properties.off_resonance = column(off_resonance, count, "off_resonance");
        properties.transmit = sensitivities(transmit, count, properties.transmit_channels, "transmit");
        properties.receive = sensitivities(receive, count, properties.coils, "receive");
        py::gil_scoped_release unlocked;
        return new pulseq::Isochromats(std::move(properties), threads);
    }

    py::array_t<double> magnetization(pulseq::Isochromats& self)
    {
        py::array_t<double> out({static_cast<py::ssize_t>(self.size()), static_cast<py::ssize_t>(3)});
        double* into = out.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.magnetization(into);
        }
        return out;
    }

    void set_magnetization(pulseq::Isochromats& self, const Doubles& values)
    {
        if (values.ndim() != 2 || static_cast<size_t>(values.shape(0)) != self.size() || values.shape(1) != 3)
            throw std::invalid_argument("the magnetisation must be (isochromats, 3)");
        self.set_magnetization(values.data());
    }

    py::array_t<Complex> simulate_sequence(
        const pulseqpp_events::BoundSequence& sequence,
        pulseq::Isochromats& isochromats,
        int first_block,
        int last_block,
        double b0,
        double gamma)
    {
        pulseq::SimulationOptions options;
        options.first_block = first_block;
        options.last_block = last_block;
        options.b0 = b0;
        options.gamma = gamma;
        std::vector<Complex> samples;
        {
            py::gil_scoped_release unlocked;
            samples = pulseq::simulate(sequence, isochromats, options);
        }
        const size_t coils = isochromats.coils();
        py::array_t<Complex> out(
            {static_cast<py::ssize_t>(coils), static_cast<py::ssize_t>(samples.size() / coils)});
        std::copy(samples.begin(), samples.end(), out.mutable_data());
        return out;
    }

    py::array_t<Complex> play(
        pulseq::Isochromats& self,
        double duration,
        const py::sequence& gradients,
        double rf_start,
        double rf_step,
        const py::object& rf,
        const py::object& adc)
    {
        if (py::len(gradients) != 3)
            throw std::invalid_argument("gradients must hold the three axes");
        pulseq::BlockEvents block;
        block.duration = duration;
        std::vector<Doubles> held;
        held.reserve(3);
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const py::object given = gradients[axis];
            if (given.is_none())
                continue;
            held.push_back(py::cast<Doubles>(given));
            const Doubles& corners = held.back();
            if (corners.ndim() != 2 || corners.shape(0) != 2)
                throw std::invalid_argument("a gradient must be a (2, n) array of time over amplitude");
            const size_t count = static_cast<size_t>(corners.shape(1));
            block.gradient_times[axis] = corners.data();
            block.gradient_values[axis] = corners.data() + count;
            block.gradient_corners[axis] = count;
        }
        Complexes samples;
        if (!rf.is_none())
        {
            samples = py::cast<Complexes>(rf);
            if (samples.ndim() != 2)
                throw std::invalid_argument("an RF pulse must be (channels, steps)");
            block.rf_start = rf_start;
            block.rf_step = rf_step;
            block.rf_channels = static_cast<size_t>(samples.shape(0));
            block.rf_steps = static_cast<size_t>(samples.shape(1));
            block.rf = samples.data();
        }
        Doubles times;
        if (!adc.is_none())
        {
            times = py::cast<Doubles>(adc);
            if (times.ndim() != 1)
                throw std::invalid_argument("ADC sample times must be one-dimensional");
            block.adc_times = times.data();
            block.adc_samples = static_cast<size_t>(times.shape(0));
        }
        py::array_t<Complex> signal(
            {static_cast<py::ssize_t>(self.coils()), static_cast<py::ssize_t>(block.adc_samples)});
        Complex* out = signal.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.play(block, out);
        }
        return signal;
    }

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

    py::class_<pulseq::Isochromats>(module, "Isochromats")
        .def(py::init(&make_isochromats),
             py::arg("positions"),
             py::arg("proton_density"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("off_resonance"),
             py::arg("transmit") = py::none(),
             py::arg("receive") = py::none(),
             py::arg("threads") = 0)
        .def_property_readonly("size", &pulseq::Isochromats::size)
        .def_property_readonly("coils", &pulseq::Isochromats::coils)
        .def_property_readonly("transmit_channels", &pulseq::Isochromats::transmit_channels)
        .def_property_readonly("elapsed", &pulseq::Isochromats::elapsed)
        .def("reset", &pulseq::Isochromats::reset)
        .def("magnetization", &magnetization)
        .def("set_magnetization", &set_magnetization)
        .def("play",
             &play,
             py::arg("duration"),
             py::arg("gradients"),
             py::arg("rf_start") = 0.0,
             py::arg("rf_step") = 0.0,
             py::arg("rf") = py::none(),
             py::arg("adc") = py::none());

    module.def(
        "simulate",
        &simulate_sequence,
        py::arg("sequence"),
        py::arg("isochromats"),
        py::arg("first_block") = 1,
        py::arg("last_block") = 0,
        py::arg("b0") = 1.5,
        py::arg("gamma") = 42576000.0,
        "Play the blocks of a sequence on isochromats; every ADC sample, (coils, samples).");
}
