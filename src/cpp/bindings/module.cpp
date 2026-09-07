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

    using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using IntMatrix = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
    using IntVector = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

    /** An (N, width) double matrix, or an error naming what was wrong. */
    Matrix as_matrix(const py::object& source, int width, const char* what)
    {
        auto array = py::cast<Matrix>(source);
        if (array.ndim() != 2 || array.shape(1) != width)
            throw std::invalid_argument(
                std::string(what) + " must have shape (N, " + std::to_string(width) + ")");
        return array;
    }

    /** An (N, width) int32 matrix, or an error naming what was wrong. */
    IntMatrix as_int_matrix(const py::object& source, int width, const char* what)
    {
        auto array = py::cast<IntMatrix>(source);
        if (array.ndim() != 2 || array.shape(1) != width)
            throw std::invalid_argument(
                std::string(what) + " must have shape (N, " + std::to_string(width) + ")");
        return array;
    }

    /** Replace a fixed-width table wholesale, in one copy. */
    void fill_table(pulseq::Table& table, const py::object& source, const char* what)
    {
        auto array = as_matrix(source, table.width(), what);
        const auto rows = static_cast<int>(array.shape(0));
        table.resize(rows);
        if (rows)
            std::memcpy(table.data(), array.data(), sizeof(double) * rows * table.width());
    }

    /** The same, for a table of whole numbers. */
    void fill_int_table(pulseq::IntTable& table, const py::object& source, const char* what)
    {
        auto array = as_int_matrix(source, table.width(), what);
        const auto rows = static_cast<int>(array.shape(0));
        table.resize(rows);
        if (rows)
            std::memcpy(table.data(), array.data(), sizeof(int32_t) * rows * table.width());
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
        /* -- bulk loading ---------------------------------------------- */
        //
        // A sequence that already exists as dense arrays -- one read from a
        // file, or one composed elsewhere -- crosses in one copy per library
        // rather than one call per row. A protocol-scale scan is millions of
        // rows, and registering them one at a time costs more than everything
        // else put together. These replace rather than append, and they trust
        // the ids they are given, because whoever holds the arrays built both.
        .def(
            "set_rf",
            [](pulseq::Sequence& self, const py::object& rows, const std::string& uses) {
                fill_table(self.rf_library(), rows, "rf");
                if (static_cast<int>(uses.size()) != self.rf_library().size())
                    throw std::invalid_argument("one use character per RF row is required");
                self.rf_uses().assign(uses.begin(), uses.end());
            },
            py::arg("rows"), py::arg("uses"))
        .def(
            "set_gradients",
            [](pulseq::Sequence& self, const py::object& traps, const py::object& arbitrary,
               const py::object& slots) {
                // Two fixed-width tables plus the map from the shared id onto
                // them: a positive slot is a trapezoid row, a negative one an
                // arbitrary row, both 1-based.
                fill_table(self.trap_library(), traps, "traps");
                fill_table(self.arb_library(), arbitrary, "arbitrary gradients");
                auto array = py::cast<IntVector>(slots);
                self.set_grad_slots(array.data(), static_cast<int>(array.size()));
            },
            py::arg("traps"), py::arg("arbitrary"), py::arg("slots"))
        .def(
            "set_adc",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_table(self.adc_library(), rows, "adc");
            },
            py::arg("rows"))
        .def(
            "set_triggers",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_table(self.trigger_library(), rows, "triggers");
            },
            py::arg("rows"))
        .def(
            "set_rotations",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_table(self.rotation_library(), rows, "rotations");
            },
            py::arg("rows"))
        .def(
            "set_extensions",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_int_table(self.extensions_library(), rows, "extension chains");
            },
            py::arg("rows"))
        .def(
            "set_label_set",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_int_table(self.label_set_library(), rows, "label set");
            },
            py::arg("rows"))
        .def(
            "set_label_inc",
            [](pulseq::Sequence& self, const py::object& rows) {
                fill_int_table(self.label_inc_library(), rows, "label inc");
            },
            py::arg("rows"))
        .def(
            "set_shapes",
            [](pulseq::Sequence& self, const py::object& lengths, const py::object& starts,
               const py::object& samples) {
                auto counts = py::cast<IntVector>(lengths);
                auto offsets = py::cast<IntVector>(starts);
                auto values = py::cast<Matrix>(samples);
                if (offsets.size() != counts.size() + 1)
                    throw std::invalid_argument(
                        "starts must hold one more offset than there are shapes");
                self.set_shapes(counts.data(), static_cast<int>(counts.size()), offsets.data(),
                                values.data());
            },
            py::arg("lengths"), py::arg("starts"), py::arg("samples"))
        .def(
            "set_rf_shims",
            [](pulseq::Sequence& self, const py::object& starts, const py::object& values) {
                auto offsets = py::cast<IntVector>(starts);
                auto data = py::cast<Matrix>(values);
                if (offsets.size() < 1)
                    throw std::invalid_argument("starts must hold at least one offset");
                self.set_rf_shims(offsets.data(), static_cast<int>(offsets.size()) - 1,
                                  data.data());
            },
            py::arg("starts"), py::arg("values"))
        .def(
            "set_soft_delays",
            [](pulseq::Sequence& self, const std::vector<int32_t>& numbers,
               const std::vector<double>& offsets, const std::vector<double>& factors,
               const std::vector<std::string>& hints) {
                if (numbers.size() != offsets.size() || numbers.size() != factors.size() ||
                    numbers.size() != hints.size())
                    throw std::invalid_argument("every soft delay column must be the same length");
                self.soft_delay_library().clear();
                for (size_t row = 0; row < numbers.size(); ++row)
                {
                    pulseq::SoftDelay delay;
                    delay.num = numbers[row];
                    delay.offset = offsets[row];
                    delay.factor = factors[row];
                    delay.hint = hints[row];
                    self.register_soft_delay(delay);
                }
            },
            py::arg("numbers"), py::arg("offsets"), py::arg("factors"), py::arg("hints"))
        .def(
            "set_blocks",
            [](pulseq::Sequence& self, const py::array& events, const py::object& durations) {
                // Six columns, or Pulseq's seven with the legacy delay id
                // leading. The wide table is taken as it stands, because the
                // narrowing then happens in the copy this makes anyway rather
                // than in a NumPy slice that would pass over the whole table
                // to arrive at the same rows.
                if (events.ndim() != 2)
                    throw std::invalid_argument("the block table must be two-dimensional");
                const auto columns = static_cast<int>(events.shape(1));
                if (columns != pulseq::BLOCK_WIDTH && columns != pulseq::BLOCK_WIDTH + 1)
                    throw std::invalid_argument(
                        "the block table must have 6 or 7 columns, not " +
                        std::to_string(columns));

                auto spans = py::cast<Matrix>(durations);
                const auto rows = static_cast<int>(events.shape(0));
                if (spans.ndim() != 1 || static_cast<int>(spans.shape(0)) != rows)
                    throw std::invalid_argument("one duration per block is required");

                if (columns == pulseq::BLOCK_WIDTH)
                {
                    auto narrow = as_int_matrix(events, pulseq::BLOCK_WIDTH, "block events");
                    self.set_blocks(narrow.data(), spans.data(), rows);
                    return;
                }

                auto wide = py::cast<IntMatrix>(events);
                std::vector<int32_t> narrow(static_cast<size_t>(rows) * pulseq::BLOCK_WIDTH);
                const int32_t* source = wide.data();
                for (int row = 0; row < rows; ++row)
                    std::memcpy(&narrow[static_cast<size_t>(row) * pulseq::BLOCK_WIDTH],
                                source + static_cast<size_t>(row) * (pulseq::BLOCK_WIDTH + 1) + 1,
                                sizeof(int32_t) * pulseq::BLOCK_WIDTH);
                self.set_blocks(narrow.data(), spans.data(), rows);
            },
            py::arg("events"), py::arg("durations"))

        /* -- counts ------------------------------------------------------ */
        .def("num_rf", [](const pulseq::Sequence& self) { return self.rf_library().size(); })
        .def("num_gradients", &pulseq::Sequence::num_gradients)
        .def("num_adc", [](const pulseq::Sequence& self) { return self.adc_library().size(); })
        .def("num_triggers",
             [](const pulseq::Sequence& self) { return self.trigger_library().size(); })
        .def("num_rotations",
             [](const pulseq::Sequence& self) { return self.rotation_library().size(); })
        .def("num_extensions",
             [](const pulseq::Sequence& self) { return self.extensions_library().size(); })
        .def("num_label_set",
             [](const pulseq::Sequence& self) { return self.label_set_library().size(); })
        .def("num_label_inc",
             [](const pulseq::Sequence& self) { return self.label_inc_library().size(); })
        .def("num_shapes",
             [](const pulseq::Sequence& self) { return self.shape_library().size(); })
        .def("num_soft_delays", [](const pulseq::Sequence& self) {
            return static_cast<int>(self.soft_delay_library().size());
        })

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
