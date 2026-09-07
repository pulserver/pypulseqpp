/**
 * @file module.cpp
 * @brief The compiled sequence core, bound as `pypulseqpp._ext`.
 *
 * The binding is deliberately thin: it hands NumPy rows straight to the
 * library's registration calls and returns the ids they mint. Every design
 * decision about what a row means lives on the Python side, and every
 * decision about how rows are stored lives in C++.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "pulseq/sequence.hpp"
#include "pulseq/shape.hpp"
#include "pulseq/types.hpp"
#include "pulseq/write.hpp"

namespace py = pybind11;

namespace
{

    using Row = py::array_t<double, py::array::c_style | py::array::forcecast>;

    /** The one row of @p width a registration call expects, or an error. */
    const double* row_of(const Row& values, int width, const char* what)
    {
        if (values.ndim() != 1 || values.shape(0) != width)
            throw std::invalid_argument(
                std::string(what) + " takes exactly " + std::to_string(width) +
                " values, got " + std::to_string(values.size()));
        return values.data();
    }

    /**
     * `add_block(rf, gx, gy, gz, adc, ext, duration)`, without argument parsing.
     *
     * A design loop makes this call once per block and a protocol-scale scan
     * has millions of them, so it is the one place where pybind11's argument
     * handling is worth going around: METH_FASTCALL hands the arguments over
     * as a C array of borrowed references, which is what building a block
     * wanted in the first place. Nothing is allocated, and no tuple is built.
     */
    PyObject* add_block_fast(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
    {
        if (nargs != 7)
        {
            PyErr_SetString(
                PyExc_TypeError,
                "add_block() takes exactly 7 arguments "
                "(rf, gx, gy, gz, adc, ext, duration)");
            return nullptr;
        }
        try
        {
            pulseq::Block block;
            block.rf = static_cast<int32_t>(PyLong_AsLong(args[0]));
            block.gx = static_cast<int32_t>(PyLong_AsLong(args[1]));
            block.gy = static_cast<int32_t>(PyLong_AsLong(args[2]));
            block.gz = static_cast<int32_t>(PyLong_AsLong(args[3]));
            block.adc = static_cast<int32_t>(PyLong_AsLong(args[4]));
            block.ext = static_cast<int32_t>(PyLong_AsLong(args[5]));
            block.duration = PyFloat_AsDouble(args[6]);
            if (PyErr_Occurred())
                return nullptr;

            pulseq::Sequence& sequence = py::cast<pulseq::Sequence&>(py::handle(self));
            return PyLong_FromLong(sequence.add_block(block));
        }
        catch (py::error_already_set& raised)
        {
            raised.restore();
            return nullptr;
        }
        catch (const std::exception& raised)
        {
            PyErr_SetString(PyExc_ValueError, raised.what());
            return nullptr;
        }
    }

    PyMethodDef add_block_fast_def = {
        "add_block",
        reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(add_block_fast)),
        METH_FASTCALL,
        PyDoc_STR("add_block(rf, gx, gy, gz, adc, ext, duration) -> int")};

    /** A `[DEFINITIONS]` value from whatever Python handed over. */
    pulseq::Definition definition_from(const py::object& value)
    {
        if (py::isinstance<py::str>(value))
            return pulseq::Definition(value.cast<std::string>());
        if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value))
            return pulseq::Definition(value.cast<double>());
        return pulseq::Definition(value.cast<std::vector<double>>());
    }

} // namespace

PYBIND11_MODULE(_ext, module)
{
    module.doc() = "Compiled sequence core for pypulseqpp";

    py::class_<pulseq::Block>(module, "Block", "One block's event ids and its duration.")
        .def(
            py::init([](int32_t rf, int32_t gx, int32_t gy, int32_t gz, int32_t adc, int32_t ext,
                        double duration) {
                pulseq::Block block;
                block.rf = rf;
                block.gx = gx;
                block.gy = gy;
                block.gz = gz;
                block.adc = adc;
                block.ext = ext;
                block.duration = duration;
                return block;
            }),
            py::arg("rf") = 0, py::arg("gx") = 0, py::arg("gy") = 0, py::arg("gz") = 0,
            py::arg("adc") = 0, py::arg("ext") = 0, py::arg("duration") = 0.0)
        .def_readwrite("rf", &pulseq::Block::rf)
        .def_readwrite("gx", &pulseq::Block::gx)
        .def_readwrite("gy", &pulseq::Block::gy)
        .def_readwrite("gz", &pulseq::Block::gz)
        .def_readwrite("adc", &pulseq::Block::adc)
        .def_readwrite("ext", &pulseq::Block::ext)
        .def_readwrite("duration", &pulseq::Block::duration);

    py::class_<pulseq::SoftDelay>(module, "SoftDelay", "One `[DELAYS]` library row.")
        .def(py::init([](int32_t num, double offset, double factor, std::string hint) {
                 pulseq::SoftDelay delay;
                 delay.num = num;
                 delay.offset = offset;
                 delay.factor = factor;
                 delay.hint = std::move(hint);
                 return delay;
             }),
             py::arg("num"), py::arg("offset"), py::arg("factor"), py::arg("hint"))
        .def_readwrite("num", &pulseq::SoftDelay::num)
        .def_readwrite("offset", &pulseq::SoftDelay::offset)
        .def_readwrite("factor", &pulseq::SoftDelay::factor)
        .def_readwrite("hint", &pulseq::SoftDelay::hint);

    auto sequence_class =
        py::class_<pulseq::Sequence>(module, "Sequence", "Event libraries and a block table.")
            .def(py::init<>())

        /* -- header ---------------------------------------------------- */
        .def("set_version", &pulseq::Sequence::set_version, py::arg("major"), py::arg("minor"),
             py::arg("revision"))
        .def("set_rasters", &pulseq::Sequence::set_rasters, py::arg("rf"), py::arg("grad"),
             py::arg("adc"), py::arg("block"))
        .def("publish_rasters", &pulseq::Sequence::publish_rasters,
             "Record the raster times in `[DEFINITIONS]`.")

        /* -- definitions ----------------------------------------------- */
        .def(
            "set_definition",
            [](pulseq::Sequence& self, const std::string& key, const py::object& value) {
                self.set_definition(key, definition_from(value));
            },
            py::arg("key"), py::arg("value"))
        .def(
            "set_integer_definition",
            [](pulseq::Sequence& self, const std::string& key, std::vector<double> values) {
                self.set_definition(key, pulseq::Definition::integers(std::move(values)));
            },
            py::arg("key"), py::arg("values"),
            "Record whole numbers, which the writer formats without a decimal point.")
        .def(
            "definitions",
            [](const pulseq::Sequence& self) {
                py::dict out;
                for (const auto& entry : self.definitions())
                {
                    const pulseq::Definition& value = entry.second;
                    if (value.kind() == pulseq::Definition::Kind::Text)
                        out[py::str(entry.first)] = py::str(value.text());
                    else
                        out[py::str(entry.first)] = py::cast(value.numbers());
                }
                return out;
            },
            "Every `[DEFINITIONS]` entry, as a dict.")

        /* -- event libraries ------------------------------------------- */
        .def(
            "register_rf",
            [](pulseq::Sequence& self, const Row& values, const std::string& use) {
                return self.register_rf(row_of(values, pulseq::RF_WIDTH, "an RF event"),
                                        use.empty() ? 'u' : use[0]);
            },
            py::arg("values"), py::arg("use") = "u")
        .def(
            "register_trap",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_trap(row_of(values, pulseq::TRAP_WIDTH, "a trapezoid"));
            },
            py::arg("values"))
        .def(
            "register_arbitrary",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_arbitrary(
                    row_of(values, pulseq::ARB_WIDTH, "an arbitrary gradient"));
            },
            py::arg("values"))
        .def(
            "register_adc",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_adc(row_of(values, pulseq::ADC_WIDTH, "an ADC event"));
            },
            py::arg("values"))
        .def(
            "register_trigger",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_trigger(row_of(values, pulseq::TRIGGER_WIDTH, "a trigger"));
            },
            py::arg("values"))
        .def(
            "register_rotation",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_rotation(row_of(values, pulseq::ROTATION_WIDTH, "a rotation"));
            },
            py::arg("values"))
        .def("register_label_set", &pulseq::Sequence::register_label_set, py::arg("value"),
             py::arg("label_id"))
        .def("register_label_inc", &pulseq::Sequence::register_label_inc, py::arg("value"),
             py::arg("label_id"))
        .def(
            "register_rf_shim",
            [](pulseq::Sequence& self, const Row& values) {
                return self.register_rf_shim(values.data(), static_cast<int>(values.size()));
            },
            py::arg("values"))
        .def("register_soft_delay", &pulseq::Sequence::register_soft_delay, py::arg("delay"))

        /* -- shapes ---------------------------------------------------- */
        .def(
            "register_shape",
            [](pulseq::Sequence& self, int num_uncompressed, const Row& samples) {
                return self.register_shape(num_uncompressed, samples.data(),
                                           static_cast<int>(samples.size()));
            },
            py::arg("num_uncompressed"), py::arg("samples"),
            "Register an already-compressed shape.")
        .def(
            "register_raw_shape",
            [](pulseq::Sequence& self, const Row& samples) {
                return self.register_raw_shape(samples.data(), static_cast<int>(samples.size()));
            },
            py::arg("samples"), "Register uncompressed samples; compress_shapes() encodes them.")
        .def(
            "register_raw_shape_divided",
            [](pulseq::Sequence& self, const Row& samples, double divisor) {
                return self.register_raw_shape_divided(samples.data(),
                                                       static_cast<int>(samples.size()), divisor);
            },
            py::arg("samples"), py::arg("divisor"))
        .def("compress_shapes", &pulseq::Sequence::compress_shapes,
             py::call_guard<py::gil_scoped_release>(),
             "Run-length encode every shape registered raw.")

        /* -- extensions and labels ------------------------------------- */
        .def("extension_type_id", &pulseq::Sequence::extension_type_id, py::arg("name"),
             "The id for an extension name, minting one if it is new.")
        .def("set_extension_type_id", &pulseq::Sequence::set_extension_type_id, py::arg("name"),
             py::arg("id"), "Pin an extension name to a chosen id.")
        .def("label_id", &pulseq::Sequence::label_id, py::arg("name"),
             "The id for a label name, minting one if it is not built in.")
        .def("label_name", &pulseq::Sequence::label_name, py::arg("id"))
        .def("chain_extension", &pulseq::Sequence::chain_extension, py::arg("type_id"),
             py::arg("ref"), py::arg("next"))

        /* -- blocks ---------------------------------------------------- */
        // `add_block` is attached after the class rather than here, because it
        // is METH_FASTCALL. See add_block_fast above.
        .def(
            "set_block",
            [](pulseq::Sequence& self, int index, int32_t rf, int32_t gx, int32_t gy, int32_t gz,
               int32_t adc, int32_t ext, double duration) {
                pulseq::Block block;
                block.rf = rf;
                block.gx = gx;
                block.gy = gy;
                block.gz = gz;
                block.adc = adc;
                block.ext = ext;
                block.duration = duration;
                self.set_block(index, block);
            },
            py::arg("index"), py::arg("rf"), py::arg("gx"), py::arg("gy"), py::arg("gz"),
            py::arg("adc"), py::arg("ext"), py::arg("duration"))
        .def("get_block", &pulseq::Sequence::get_block, py::arg("index"))
        .def("num_blocks", &pulseq::Sequence::num_blocks)
        .def("duration", &pulseq::Sequence::duration, "Total duration in seconds.")
        .def("remove_duplicates", &pulseq::Sequence::remove_duplicates,
             py::call_guard<py::gil_scoped_release>(),
             "Collapse identical library rows and renumber the block table.")
        .def("__len__", &pulseq::Sequence::num_blocks);

    // METH_FASTCALL has no pybind11 spelling, so the descriptor is built by
    // hand and bound onto the type the class just created.
    {
        PyTypeObject* type = reinterpret_cast<PyTypeObject*>(sequence_class.ptr());
        sequence_class.attr("add_block") =
            py::reinterpret_steal<py::object>(PyDescr_NewMethod(type, &add_block_fast_def));
    }

    module.def(
        "write_text",
        [](pulseq::Sequence& sequence, bool create_signature) {
            std::string written;
            {
                py::gil_scoped_release unlocked;
                written = pulseq::write_text(sequence, create_signature);
            }
            return py::bytes(written);
        },
        py::arg("sequence"), py::arg("create_signature") = true,
        "Serialize as a Pulseq `.seq` text file.");

    module.def("required_revision", &pulseq::required_revision, py::arg("sequence"),
               "The Pulseq revision the sequence's contents actually need.");

    module.def(
        "compress_shape",
        [](const Row& samples) {
            const std::vector<double> out =
                pulseq::compress_shape(samples.data(), static_cast<int>(samples.size()));
            return py::array_t<double>(static_cast<py::ssize_t>(out.size()), out.data());
        },
        py::arg("samples"), "Run-length encode samples on their derivative.");

    module.def(
        "decompress_shape",
        [](const Row& samples, int num_uncompressed) {
            const std::vector<double> out = pulseq::decompress_shape(
                samples.data(), static_cast<int>(samples.size()), num_uncompressed);
            return py::array_t<double>(static_cast<py::ssize_t>(out.size()), out.data());
        },
        py::arg("samples"), py::arg("num_uncompressed"), "The inverse of compress_shape.");
}
