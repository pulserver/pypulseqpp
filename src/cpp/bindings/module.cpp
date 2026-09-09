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

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulseq/analysis.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/shape.hpp"
#include "pulseq/kspace.hpp"
#include "pulseq/timing.hpp"
#include "pulseq/waveforms.hpp"
#include "pulseq/types.hpp"
#include "pulseqpp_events.h"
#include "pulseqpp_decode.h"
#include "pulseqpp_eventtypes.h"

#include "pulseq/binary.hpp"
#include "pulseq/read.hpp"
#include "pulseq/safety.hpp"
#include "pulseq/write.hpp"

namespace py = pybind11;

namespace
{
    /**
     * The sequence the module exposes.
     *
     * `pulseq::Sequence` plus the memory an event needs: which shape ids this
     * sequence issued for a waveform it has already seen, so a pulse played a
     * thousand times registers its shape once.
     */
    using Sequence = pulseqpp_events::BoundSequence;


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
     * Timing findings as dicts, each carrying only what its kind reports.
     *
     * A finding is a fixed struct in C++ so that judging a million-block
     * sequence allocates nothing per block, but what a caller reads is the
     * report the toolbox writes: the fields that mean something for this kind
     * of problem, and no others, so a message template naming them formats
     * without a missing key and without a stray zero.
     */
    py::list findings_as_dicts(const std::vector<pulseq::TimingFinding>& findings)
    {
        py::list out;
        for (size_t i = 0; i < findings.size(); ++i)
        {
            const pulseq::TimingFinding& f = findings[i];
            py::dict d;
            d["block"] = f.block;
            d["event"] = f.event;
            d["field"] = f.field;
            d["error_type"] = f.error_type;

            if (f.error_type == "RASTER")
            {
                d["value"] = f.value;
                d["value_rounded"] = f.value_rounded;
                d["error"] = f.error;
                d["raster"] = f.raster;
            }
            else if (f.error_type == "NEGATIVE_DELAY")
            {
                d["value"] = f.value;
            }
            else if (f.error_type == "BLOCK_DURATION_MISMATCH")
            {
                d["value"] = f.value;
                d["duration"] = f.duration;
            }
            else if (f.error_type == "ADC_SAMPLES_DIVISOR")
            {
                d["value"] = static_cast<int64_t>(f.value);
                d["divisor"] = f.divisor;
            }
            else if (f.error_type == "RF_DEAD_TIME" || f.error_type == "ADC_DEAD_TIME")
            {
                d["value"] = f.value;
                d["dead_time"] = f.dead_time;
            }
            else if (f.error_type == "RF_RINGDOWN_TIME")
            {
                d["value"] = f.value;
                d["duration"] = f.duration;
                d["ringdown_time"] = f.ringdown_time;
            }
            else if (f.error_type == "POST_ADC_DEAD_TIME")
            {
                d["value"] = f.value;
                d["duration"] = f.duration;
                d["dead_time"] = f.dead_time;
            }
            else if (f.error_type == "GRADIENT_START_DELAY")
            {
                d["value"] = f.value;
                d["amplitude"] = f.amplitude;
            }
            else if (f.error_type == "GRADIENT_END_NONZERO")
            {
                d["value"] = f.value;
                d["duration"] = f.duration;
                d["amplitude"] = f.amplitude;
            }
            else if (f.error_type == "SOFT_DELAY_HINT_INCONSISTENCY")
            {
                d["value"] = f.hint;
                d["hint"] = f.hint;
                d["prev_hint"] = f.prev_hint;
                d["numID"] = static_cast<int64_t>(f.num_id);
            }
            else if (f.error_type == "SOFT_DELAY_INVALID_NUMID")
            {
                d["value"] = static_cast<int64_t>(f.value);
                d["hint"] = f.hint;
                d["numID"] = static_cast<int64_t>(f.num_id);
            }
            else
            {
                d["value"] = f.value;
                d["hint"] = f.hint;
                d["numID"] = static_cast<int64_t>(f.num_id);
            }
            out.append(d);
        }
        return out;
    }

    /** A capsule owning @p buffer, so an array over it keeps it alive. */
    template <typename T> py::capsule keep_alive_capsule(std::shared_ptr<const T> buffer)
    {
        auto* held = new std::shared_ptr<const T>(std::move(buffer));
        return py::capsule(held, [](void* owned) {
            delete static_cast<std::shared_ptr<const T>*>(owned);
        });
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

            Sequence& sequence = py::cast<Sequence&>(py::handle(self));
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

    /**
     * `add_block_events(*events)`, called without building a tuple.
     *
     * The compiled events go straight to `build_block`, which registers what
     * the sequence has not seen and hands back the block's rows -- so a block
     * costs one crossing rather than one per event.
     */
    PyObject* add_block_events_fast(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
    {
        try
        {
            Sequence& sequence = py::cast<Sequence&>(py::handle(self));
            return PyLong_FromLong(
                sequence.add_block(pulseqpp_events::build_block(sequence, args, nargs)));
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

    /**
     * `set_block_events(index, *events)`, the same crossing written over an
     * existing block.
     *
     * A sequence read out block by block and put back -- rotated, rescaled,
     * a label changed -- makes this call once per block, so it is bound the
     * same way `add_block_events` is.
     */
    PyObject* set_block_events_fast(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
    {
        try
        {
            if (nargs < 1)
                throw std::invalid_argument("set_block_events() needs a block index");
            Sequence& sequence = py::cast<Sequence&>(py::handle(self));
            const long index = PyLong_AsLong(args[0]);
            if (index == -1 && PyErr_Occurred())
                return nullptr;
            sequence.set_block(
                static_cast<int>(index),
                pulseqpp_events::build_block(sequence, args + 1, nargs - 1));
            Py_RETURN_NONE;
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

    PyMethodDef set_block_events_def = {
        "set_block_events",
        reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(set_block_events_fast)),
        METH_FASTCALL,
        PyDoc_STR("set_block_events(index, *events) -> None")};

    PyMethodDef add_block_events_def = {
        "add_block_events",
        reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(add_block_events_fast)),
        METH_FASTCALL,
        PyDoc_STR("add_block_events(*events) -> int")};

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

    // The events a block is made of, as compiled objects rather than
    // dictionaries. See pulseqpp_eventtypes.h.
    pulseqpp_types::bind(module);

    py::enum_<pulseq::ShapeRole>(module, "ShapeRole", py::arithmetic(),
                                 "What a shape is played as. Masks, so they combine.")
        .value("NONE", pulseq::SHAPE_ROLE_NONE)
        .value("RF_MAGNITUDE", pulseq::SHAPE_ROLE_RF_MAGNITUDE)
        .value("RF_PHASE", pulseq::SHAPE_ROLE_RF_PHASE)
        .value("RF_TIME", pulseq::SHAPE_ROLE_RF_TIME)
        .value("GRADIENT", pulseq::SHAPE_ROLE_GRADIENT)
        .value("GRADIENT_TIME", pulseq::SHAPE_ROLE_GRADIENT_TIME)
        .value("ADC_PHASE", pulseq::SHAPE_ROLE_ADC_PHASE)
        .value("TIME", pulseq::SHAPE_ROLE_TIME)
        .export_values();

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
        py::class_<Sequence>(module, "Sequence", "Event libraries and a block table.")
            .def(py::init<>())

        /* -- header ---------------------------------------------------- */
        .def("set_version", &Sequence::set_version, py::arg("major"), py::arg("minor"),
             py::arg("revision"))
        .def("set_rasters", &Sequence::set_rasters, py::arg("rf"), py::arg("grad"),
             py::arg("adc"), py::arg("block"))
        .def("publish_rasters", &Sequence::publish_rasters,
             "Record the raster times in `[DEFINITIONS]`.")

        /* -- definitions ----------------------------------------------- */
        .def(
            "set_definition",
            [](Sequence& self, const std::string& key, const py::object& value) {
                self.set_definition(key, definition_from(value));
            },
            py::arg("key"), py::arg("value"))
        .def(
            "set_integer_definition",
            [](Sequence& self, const std::string& key, std::vector<double> values) {
                self.set_definition(key, pulseq::Definition::integers(std::move(values)));
            },
            py::arg("key"), py::arg("values"),
            "Record whole numbers, which the writer formats without a decimal point.")
        .def(
            "definitions",
            [](const Sequence& self) {
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
            [](Sequence& self, const Row& values, const std::string& use) {
                return self.register_rf(row_of(values, pulseq::RF_WIDTH, "an RF event"),
                                        use.empty() ? 'u' : use[0]);
            },
            py::arg("values"), py::arg("use") = "u")
        .def(
            "register_trap",
            [](Sequence& self, const Row& values) {
                return self.register_trap(row_of(values, pulseq::TRAP_WIDTH, "a trapezoid"));
            },
            py::arg("values"))
        .def(
            "register_arbitrary",
            [](Sequence& self, const Row& values) {
                return self.register_arbitrary(
                    row_of(values, pulseq::ARB_WIDTH, "an arbitrary gradient"));
            },
            py::arg("values"))
        .def(
            "register_adc",
            [](Sequence& self, const Row& values) {
                return self.register_adc(row_of(values, pulseq::ADC_WIDTH, "an ADC event"));
            },
            py::arg("values"))
        .def(
            "register_trigger",
            [](Sequence& self, const Row& values) {
                return self.register_trigger(row_of(values, pulseq::TRIGGER_WIDTH, "a trigger"));
            },
            py::arg("values"))
        .def(
            "register_rotation",
            [](Sequence& self, const Row& values) {
                return self.register_rotation(row_of(values, pulseq::ROTATION_WIDTH, "a rotation"));
            },
            py::arg("values"))
        .def("register_label_set", &Sequence::register_label_set, py::arg("value"),
             py::arg("label_id"))
        .def("register_label_inc", &Sequence::register_label_inc, py::arg("value"),
             py::arg("label_id"))
        .def(
            "register_rf_shim",
            [](Sequence& self, const Row& values) {
                return self.register_rf_shim(values.data(), static_cast<int>(values.size()));
            },
            py::arg("values"))
        .def("register_soft_delay", &Sequence::register_soft_delay, py::arg("delay"))

        /* -- shapes ---------------------------------------------------- */
        .def(
            "register_shape",
            [](Sequence& self, int num_uncompressed, const Row& samples) {
                return self.register_shape(num_uncompressed, samples.data(),
                                           static_cast<int>(samples.size()));
            },
            py::arg("num_uncompressed"), py::arg("samples"),
            "Register an already-compressed shape.")
        .def(
            "register_raw_shape",
            [](Sequence& self, const Row& samples) {
                return self.register_raw_shape(samples.data(), static_cast<int>(samples.size()));
            },
            py::arg("samples"), "Register uncompressed samples; compress_shapes() encodes them.")
        .def(
            "register_raw_shape_divided",
            [](Sequence& self, const Row& samples, double divisor) {
                return self.register_raw_shape_divided(samples.data(),
                                                       static_cast<int>(samples.size()), divisor);
            },
            py::arg("samples"), py::arg("divisor"))
        .def("compress_shapes", &Sequence::compress_shapes,
             py::call_guard<py::gil_scoped_release>(),
             "Run-length encode every shape registered raw.")

        /* -- what shapes are played as ----------------------------------- */
        .def(
            "shape_roles",
            [](const Sequence& self) {
                const pulseq::ShapeLibrary& shapes = self.shape_library();
                py::array_t<uint32_t> out(shapes.size());
                uint32_t* values = out.mutable_data();
                for (int id = 1; id <= shapes.size(); ++id)
                    values[id - 1] = shapes.roles(id);
                return out;
            },
            "Per shape, a mask of what it is played as. See ShapeRole.")
        .def(
            "shapes_with_role",
            [](const Sequence& self, uint32_t role) {
                const pulseq::ShapeLibrary& shapes = self.shape_library();
                std::vector<int32_t> found;
                for (int id = 1; id <= shapes.size(); ++id)
                    if (shapes.roles(id) & role)
                        found.push_back(id);
                return py::array_t<int32_t>(static_cast<py::ssize_t>(found.size()), found.data());
            },
            py::arg("role"),
            "The ids of every shape played as any of `role`, in id order.")

        /* -- extensions and labels ------------------------------------- */
        .def("extension_type_id", &Sequence::extension_type_id, py::arg("name"),
             "The id for an extension name, minting one if it is new.")
        .def("set_extension_type_id", &Sequence::set_extension_type_id, py::arg("name"),
             py::arg("id"), "Pin an extension name to a chosen id.")
        .def("extension_type_name", &Sequence::extension_type_name, py::arg("id"),
             "The name an extension id stands for.")
        .def(
            "extension_chain",
            [](const Sequence& self, int32_t head) {
                /* Type and reference per link, as a 2-by-n array: the shape
                 * the toolboxes report a block's extensions in. */
                const pulseq::IntTable& links = self.extensions_library();
                std::vector<int32_t> types;
                std::vector<int32_t> refs;
                int32_t node = head;
                while (node > 0 && node <= links.size())
                {
                    const int32_t* link = links.row(node);
                    types.push_back(link[0]);
                    refs.push_back(link[1]);
                    node = link[2];
                }
                const py::ssize_t held = static_cast<py::ssize_t>(types.size());
                py::array_t<int32_t> out({static_cast<py::ssize_t>(2), held});
                auto view = out.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < held; ++i)
                {
                    view(0, i) = types[static_cast<size_t>(i)];
                    view(1, i) = refs[static_cast<size_t>(i)];
                }
                return out;
            },
            py::arg("head"),
            "The extension chain from `head`, as type and reference ids.")
        .def("label_id", &Sequence::label_id, py::arg("name"),
             "The id for a label name, minting one if it is not built in.")
        .def("label_name", &Sequence::label_name, py::arg("id"))
        .def("chain_extension", &Sequence::chain_extension, py::arg("type_id"),
             py::arg("ref"), py::arg("next"))

        /* -- blocks ---------------------------------------------------- */
        /* -- reading the block table back -------------------------------- */
        //
        // Views, not copies: a million-row table costs nothing to read a
        // column out of. The array owns a share of the buffer it points into,
        // and the sequence copies that buffer before writing to it while a
        // view is out, so a view is a snapshot. It never sees a later write
        // and it never outlives its memory -- it stays valid even if the
        // sequence itself is collected.
        .def(
            "block_events",
            [](const Sequence& self) {
                /* The six columns the file carries, over a table that holds
                 * more: a stride rather than a copy, so this stays a view
                 * into the table and the rotation column stays out of a
                 * caller's way. */
                auto buffer = self.block_events_buffer();
                const int32_t* first = buffer->data();
                const py::ssize_t item = static_cast<py::ssize_t>(sizeof(int32_t));
                return py::array_t<int32_t>(
                    {static_cast<py::ssize_t>(self.num_blocks()),
                     static_cast<py::ssize_t>(pulseq::BLOCK_FILE_COLUMNS)},
                    {item * pulseq::BLOCK_WIDTH, item},
                    first,
                    keep_alive_capsule(std::move(buffer)));
            },
            "The block table as an (N, 6) snapshot: rf, gx, gy, gz, adc, ext.")
        .def(
            "block_rotations",
            [](const Sequence& self) {
                /* The rotation each block turns its gradients by, as a column
                 * of the same table -- a rotation is an extension in the file
                 * and stays one, and this is how often it is asked for. */
                auto buffer = self.block_events_buffer();
                const int32_t* first = buffer->data() + pulseq::BLOCK_WIDTH - 1;
                const py::ssize_t item = static_cast<py::ssize_t>(sizeof(int32_t));
                return py::array_t<int32_t>(
                    {static_cast<py::ssize_t>(self.num_blocks())},
                    {item * pulseq::BLOCK_WIDTH},
                    first,
                    keep_alive_capsule(std::move(buffer)));
            },
            "Per block, the rotation row it turns its gradients by; 0 for none.")
        .def(
            "block_durations",
            [](const Sequence& self) {
                auto buffer = self.block_durations_buffer();
                const double* first = buffer->data();
                return py::array_t<double>({self.num_blocks()}, first,
                                           keep_alive_capsule(std::move(buffer)));
            },
            "Every block's duration in seconds, as a snapshot.")
        // A soft delay rewrites a block's duration and nothing else about it,
        // so it gets a scalar setter rather than a rebuild of the block.
        .def(
            "set_block_duration",
            [](Sequence& self, int index, double seconds) {
                if (index < 1 || index > self.num_blocks())
                    throw py::index_error("block index out of range");
                self.block_durations()[index - 1] = seconds;
            },
            py::arg("index"), py::arg("seconds"))

        .def(
            "repetition",
            [](Sequence& self) {
                const pulseq::Repetition found = self.repetition();
                return py::make_tuple(found.size, found.start);
            },
            "The repeating unit of the scan as (size, start), in blocks; a "
            "size of 0 when the sequence does not repeat.")
        .def(
            "locate_repetition",
            [](const Sequence& self, int size) {
                const pulseq::Repetition found = self.locate_repetition(size);
                return py::make_tuple(found.size, found.start);
            },
            py::arg("size"),
            "Where a repeating unit of the given size starts, as (size, start).")
        .def(
            "detect_rf_uses", &Sequence::detect_rf_uses, py::arg("b0"), py::arg("gamma"),
            "Label every pulse the file did not, from what the pulse does. "
            "Returns how many were labelled.")
        .def(
            "apply_soft_delays",
            [](Sequence& self, const std::map<std::string, double>& values) {
                pulseq::SoftDelayReport report;
                {
                    py::gil_scoped_release unlocked;
                    report = self.apply_soft_delays(values);
                }

                py::list rounded;
                for (size_t i = 0; i < report.rounded.size(); ++i)
                {
                    const pulseq::SoftDelayReport::Rounding& note = report.rounded[i];
                    py::dict entry;
                    entry["block"] = note.block;
                    entry["hint"] = note.hint;
                    entry["numID"] = note.num;
                    entry["error"] = note.error;
                    rounded.append(entry);
                }

                py::dict out;
                out["hints"] = report.hints;
                out["rounded"] = rounded;
                out["problem"] = py::none();
                if (report.problem != pulseq::SoftDelayReport::Problem::None)
                {
                    py::dict problem;
                    problem["kind"] =
                        report.problem == pulseq::SoftDelayReport::Problem::HintRenumbered
                        ? "hint_renumbered"
                        : (report.problem ==
                                   pulseq::SoftDelayReport::Problem::NumberRenamed
                               ? "number_renamed"
                               : "negative");
                    problem["block"] = report.block;
                    problem["hint"] = report.hint;
                    problem["numID"] = report.num;
                    problem["duration"] = report.duration;
                    problem["offset"] = report.offset;
                    problem["factor"] = report.factor;
                    out["problem"] = problem;
                }
                return out;
            },
            py::arg("values"),
            "Set each named soft delay, by hint, to the value given. Returns "
            "the hints found, what had to be rounded, and the first problem.")

        /* -- counts ------------------------------------------------------ */
        .def("num_rf", [](const Sequence& self) { return self.rf_library().size(); })
        .def("num_gradients", &Sequence::num_gradients)
        .def("num_adc", [](const Sequence& self) { return self.adc_library().size(); })
        .def("num_triggers",
             [](const Sequence& self) { return self.trigger_library().size(); })
        .def("num_rotations",
             [](const Sequence& self) { return self.rotation_library().size(); })
        .def("num_extensions",
             [](const Sequence& self) { return self.extensions_library().size(); })
        .def("num_label_set",
             [](const Sequence& self) { return self.label_set_library().size(); })
        .def("num_label_inc",
             [](const Sequence& self) { return self.label_inc_library().size(); })
        .def("num_shapes",
             [](const Sequence& self) { return self.shape_library().size(); })
        .def("num_rf_shims",
             [](const Sequence& self) { return self.rf_shim_library().size(); })
        .def("num_soft_delays", [](const Sequence& self) {
            return static_cast<int>(self.soft_delay_library().size());
        })

        // `add_block` is attached after the class rather than here, because it
        // is METH_FASTCALL. See add_block_fast above.
        .def(
            "set_block",
            [](Sequence& self, int index, int32_t rf, int32_t gx, int32_t gy, int32_t gz,
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
        .def("get_block", &Sequence::get_block, py::arg("index"))
        .def(
            "decode_block",
            [](const Sequence& self, int index) {
                return pulseqpp_decode::decode_block(self, index);
            },
            py::arg("index"),
            "Block `index` (1-based) as the events it plays, rather than as "
            "the ids they are stored under.")
        .def("num_blocks", &Sequence::num_blocks)
        .def("edits", &Sequence::edits,
             "How many times the sequence has been edited; it only rises.")

        /* -- definitions and instances --------------------------------- */
        .def("num_block_definitions", &Sequence::num_block_definitions,
             "How many distinct block structures the scan plays.")
        .def("num_rf_definitions", &Sequence::num_rf_definitions)
        .def("num_grad_definitions", &Sequence::num_grad_definitions)
        .def("num_adc_definitions", &Sequence::num_adc_definitions)
        .def(
            "instance_definitions",
            [](const Sequence& self) {
                const auto& v = self.instance_definitions();
                return py::array_t<int32_t>(static_cast<py::ssize_t>(v.size()), v.data());
            },
            "Per block, the id of the definition it plays.")
        .def(
            "instance_adc_definitions",
            [](const Sequence& self) {
                const auto& v = self.instance_adc_definitions();
                return py::array_t<int32_t>(static_cast<py::ssize_t>(v.size()), v.data());
            },
            "Per block, the ADC definition it digitises with; 0 if it does not.")
        .def(
            "instance_parameters",
            [](const Sequence& self) {
                const std::vector<double> v = self.instance_parameters();
                const py::ssize_t rows =
                    static_cast<py::ssize_t>(v.size() / pulseq::INSTANCE_WIDTH);
                return py::array_t<double>(
                    {rows, static_cast<py::ssize_t>(pulseq::INSTANCE_WIDTH)}, v.data());
            },
            "Per block, its per-playout parameters.  See INSTANCE_WIDTH.")
        .def("duration", &Sequence::duration, "Total duration in seconds.")
        .def(
            "event_counts",
            [](const Sequence& self) {
                const std::array<int64_t, pulseq::BLOCK_FILE_COLUMNS> counts = self.event_counts();
                return py::array_t<int64_t>(
                    static_cast<py::ssize_t>(pulseq::BLOCK_FILE_COLUMNS), counts.data());
            },
            "How many blocks carry an event in each column: rf, gx, gy, gz, "
            "adc, extension.")
        .def(
            "scale_gradient_axis", &Sequence::scale_gradient_axis, py::arg("axis"),
            py::arg("modifier"),
            "Scale every gradient played on an axis (0, 1 or 2) by a factor.")
        .def("remove_duplicates", &Sequence::remove_duplicates,
             py::call_guard<py::gil_scoped_release>(),
             "Collapse identical library rows and renumber the block table.")
        .def("__len__", &Sequence::num_blocks);

    // METH_FASTCALL has no pybind11 spelling, so the descriptor is built by
    // hand and bound onto the type the class just created.
    {
        PyTypeObject* type = reinterpret_cast<PyTypeObject*>(sequence_class.ptr());
        sequence_class.attr("add_block") =
            py::reinterpret_steal<py::object>(PyDescr_NewMethod(type, &add_block_fast_def));
        sequence_class.attr("add_block_events") =
            py::reinterpret_steal<py::object>(PyDescr_NewMethod(type, &add_block_events_def));
        sequence_class.attr("set_block_events") =
            py::reinterpret_steal<py::object>(PyDescr_NewMethod(type, &set_block_events_def));
    }

    module.def(
        "write_text",
        [](Sequence& sequence, bool create_signature) {
            std::string written;
            {
                py::gil_scoped_release unlocked;
                written = pulseq::write_text(sequence, create_signature);
            }
            return py::bytes(written);
        },
        py::arg("sequence"), py::arg("create_signature") = true,
        "Serialize as a Pulseq `.seq` text file.");

    module.def(
        "write_text_v141",
        [](Sequence& sequence, bool create_signature, double gamma, double field) {
            if (!sequence.soft_delay_library().empty())
            {
                // The reference toolbox warns rather than refusing, and so
                // does this: the file is valid apart from the delays, which
                // 1.4.1 has no way to carry.
                PyErr_WarnEx(
                    PyExc_UserWarning,
                    "write_text_v141(): this sequence uses soft delays, which the 1.4.1 "
                    "format cannot carry; they are left out of the file",
                    1);
            }
            std::string written;
            {
                py::gil_scoped_release unlocked;
                written = pulseq::write_text_v141(sequence, create_signature, gamma, field);
            }
            return py::bytes(written);
        },
        py::arg("sequence"), py::arg("create_signature") = true,
        py::arg("gamma") = 42576000.0, py::arg("field") = 1.5,
        "Serialize as a Pulseq 1.4.1 `.seq` text file.");

    module.def(
        "check_timing",
        [](const Sequence& sequence,
           double rf_raster_time,
           double grad_raster_time,
           double adc_raster_time,
           double block_duration_raster,
           double rf_dead_time,
           double rf_ringdown_time,
           double adc_dead_time,
           double adc_samples_divisor) {
            pulseq::TimingLimits limits;
            limits.rf_raster_time = rf_raster_time;
            limits.grad_raster_time = grad_raster_time;
            limits.adc_raster_time = adc_raster_time;
            limits.block_duration_raster = block_duration_raster;
            limits.rf_dead_time = rf_dead_time;
            limits.rf_ringdown_time = rf_ringdown_time;
            limits.adc_dead_time = adc_dead_time;
            limits.adc_samples_divisor = adc_samples_divisor;

            std::vector<pulseq::TimingFinding> findings;
            {
                py::gil_scoped_release unlocked;
                findings = pulseq::check_timing(sequence, limits);
            }
            return findings_as_dicts(findings);
        },
        py::arg("sequence"), py::arg("rf_raster_time"), py::arg("grad_raster_time"),
        py::arg("adc_raster_time"), py::arg("block_duration_raster"),
        py::arg("rf_dead_time") = 0.0, py::arg("rf_ringdown_time") = 0.0,
        py::arg("adc_dead_time") = 0.0, py::arg("adc_samples_divisor") = 1.0,
        "Every timing problem in the sequence, one dict per finding.");

    module.def(
        "register_event",
        [](Sequence& sequence, py::handle event) {
            /* One event is a block of one, minus the linking: the extension
             * chain is left uncommitted, so registering a trigger costs the
             * trigger row and not an extension row nobody points at. */
            PyObject* item = event.ptr();
            py::dict out;
            /* The shapes first, so the row that follows points at ids this
             * sequence has already been given rather than at copies. */
            out["shapes"] = pulseqpp_events::warm_event(sequence, event);

            int32_t chain[8][2];
            int chained = 0;
            const pulseq::Block block =
                pulseqpp_events::collect_block(sequence, &item, 1, chain, chained);

            if (block.rf > 0)
            {
                out["kind"] = "rf";
                out["id"] = block.rf;
            }
            else if (block.gx > 0 || block.gy > 0 || block.gz > 0)
            {
                out["kind"] = "grad";
                out["id"] = block.gx > 0 ? block.gx : (block.gy > 0 ? block.gy : block.gz);
            }
            else if (block.adc > 0)
            {
                out["kind"] = "adc";
                out["id"] = block.adc;
            }
            else if (chained > 0)
            {
                out["kind"] = sequence.extension_type_name(chain[0][0]);
                out["id"] = chain[0][1];
            }
            else
            {
                out["kind"] = "delay";
                out["id"] = 0;
            }
            return out;
        },
        py::arg("sequence"), py::arg("event"),
        "Register one event's row and shapes, and report what it was stored "
        "as: its kind, its library id, and the ids of its shapes.");

    const auto peak_as_dict = [](const pulseq::Peak& found) {
        py::dict out;
        out["value"] = found.value;
        out["block"] = found.block;
        out["axis"] = found.axis;
        return out;
    };

    const auto axes_as_list = [peak_as_dict](const std::array<pulseq::Peak, 3>& found) {
        py::list out;
        for (size_t axis = 0; axis < found.size(); ++axis)
            out.append(peak_as_dict(found[axis]));
        return out;
    };

    module.def(
        "max_gradient",
        [peak_as_dict, axes_as_list](const Sequence& sequence) {
            pulseq::GradientReport found;
            {
                py::gil_scoped_release unlocked;
                found = pulseq::max_gradient(sequence);
            }
            py::dict out;
            out["per_axis"] = peak_as_dict(found.per_axis);
            out["vector"] = peak_as_dict(found.vector);
            out["axes"] = axes_as_list(found.axes);
            return out;
        },
        py::arg("sequence"),
        "The strongest gradient the sequence plays: the worst axis, each "
        "axis on its own, and the vector magnitude.");

    module.def(
        "max_slew",
        [peak_as_dict, axes_as_list](
            const Sequence& sequence, double max_slew, double grad_raster_time) {
            pulseq::GradientLimits limits;
            limits.max_slew = max_slew;
            limits.grad_raster_time = grad_raster_time;

            pulseq::SlewReport found;
            {
                py::gil_scoped_release unlocked;
                found = pulseq::max_slew(sequence, limits);
            }

            py::dict out;
            out["per_axis"] = peak_as_dict(found.per_axis);
            out["vector"] = peak_as_dict(found.vector);
            out["axes"] = axes_as_list(found.axes);
            return out;
        },
        py::arg("sequence"), py::arg("max_slew") = 0.0,
        py::arg("grad_raster_time") = 10e-6,
        "What the sequence asks in the way of slewing, within its blocks: "
        "the worst axis, each axis on its own, and the vector magnitude.");

    module.def(
        "grad_continuity",
        [](const Sequence& sequence, double max_slew, double grad_raster_time) {
            pulseq::GradientLimits limits;
            limits.max_slew = max_slew;
            limits.grad_raster_time = grad_raster_time;

            pulseq::ContinuityReport found;
            {
                py::gil_scoped_release unlocked;
                found = pulseq::continuity(sequence, limits);
            }

            py::list jumps;
            for (size_t i = 0; i < found.discontinuities.size(); ++i)
            {
                const pulseq::Discontinuity& where = found.discontinuities[i];
                py::dict entry;
                entry["block"] = where.block;
                entry["axis"] = where.axis;
                entry["before"] = where.before;
                entry["after"] = where.after;
                entry["slew"] = where.slew;
                entry["limit"] = where.limit;
                jumps.append(entry);
            }

            py::dict out;
            out["discontinuities"] = jumps;
            out["ends_at_zero"] = found.ends_at_zero;
            return out;
        },
        py::arg("sequence"), py::arg("max_slew") = 0.0,
        py::arg("grad_raster_time") = 10e-6,
        "Where a gradient does not carry on from the block before it, and "
        "whether the sequence leaves its gradients at zero.");

    module.def(
        "flip_angles",
        [](const Sequence& sequence) {
            std::vector<double> angles;
            {
                py::gil_scoped_release unlocked;
                angles = pulseq::flip_angles(sequence);
            }
            return py::array_t<double>(
                static_cast<py::ssize_t>(angles.size()), angles.data());
        },
        py::arg("sequence"),
        "Every distinct flip angle the sequence uses, in degrees, ascending.");

    module.def(
        "kspace_coverage",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& samples,
           double threshold) {
            if (samples.ndim() != 2)
                throw std::invalid_argument(
                    "the sampled trajectory must be one row per axis");

            const int axes = static_cast<int>(samples.shape(0));
            const int count = static_cast<int>(samples.shape(1));

            pulseq::KspaceCoverage found;
            {
                py::gil_scoped_release unlocked;
                found = pulseq::kspace_coverage(samples.data(), axes, count, threshold);
            }

            py::dict out;
            out["unique_positions"] = py::array_t<double>(
                static_cast<py::ssize_t>(found.unique_positions.size()),
                found.unique_positions.data());
            out["repeats_min"] = found.repeats_min;
            out["repeats_max"] = found.repeats_max;
            out["repeats_median"] = found.repeats_median;
            out["is_cartesian"] = found.is_cartesian;
            return out;
        },
        py::arg("samples"), py::arg("threshold"),
        "What the sampled trajectory covers: the distinct positions along "
        "each axis, how often a position is revisited, and whether the "
        "positions fill a grid.");

    module.def(
        "calculate_kspace",
        [](const Sequence& sequence,
           std::array<double, 3> delay,
           std::array<double, 3> offset,
           int first_block,
           int last_block,
           double b0,
           double gamma,
           bool samples_only) {
            pulseq::KspaceOptions options;
            options.samples_only = samples_only;
            options.delay = delay;
            options.offset = offset;
            options.first_block = first_block;
            options.last_block = last_block;
            options.b0 = b0;
            options.gamma = gamma;

            pulseq::Kspace found;
            {
                py::gil_scoped_release unlocked;
                found = pulseq::calculate_kspace(sequence, options);
            }

            const auto stacked = [](const std::array<std::vector<double>, 3>& rows) {
                const py::ssize_t held =
                    static_cast<py::ssize_t>(rows[0].size());
                py::array_t<double> out({static_cast<py::ssize_t>(3), held});
                auto view = out.mutable_unchecked<2>();
                for (py::ssize_t axis = 0; axis < 3; ++axis)
                    for (py::ssize_t i = 0; i < held; ++i)
                        view(axis, i) = rows[static_cast<size_t>(axis)][static_cast<size_t>(i)];
                return out;
            };
            const auto row = [](const std::vector<double>& values) {
                return py::array_t<double>(
                    static_cast<py::ssize_t>(values.size()), values.data());
            };

            py::list gradients;
            for (int axis = 0; axis < 3; ++axis)
            {
                py::dict channel;
                channel["t"] = row(found.gradient_times[static_cast<size_t>(axis)]);
                channel["v"] = row(found.gradient_values[static_cast<size_t>(axis)]);
                gradients.append(channel);
            }

            py::dict out;
            out["k_traj"] = stacked(found.position);
            out["t_ktraj"] = row(found.times);
            out["k_traj_adc"] = stacked(found.sampled);
            out["t_adc"] = row(found.adc_times);
            out["pm_adc"] = row(found.adc_modulation);
            out["t_excitation"] = row(found.excitation_times);
            out["t_refocusing"] = row(found.refocusing_times);
            out["slicepos"] = stacked(found.slice_position);
            out["gradients"] = gradients;
            out["warnings"] = found.warnings;
            return out;
        },
        py::arg("sequence"), py::arg("delay") = std::array<double, 3>{{0.0, 0.0, 0.0}},
        py::arg("offset") = std::array<double, 3>{{0.0, 0.0, 0.0}},
        py::arg("first_block") = 1, py::arg("last_block") = 0, py::arg("b0") = 1.5,
        py::arg("gamma") = 42576000.0, py::arg("samples_only") = false,
        "Follow the sequence into k-space: the trajectory, where it is "
        "sampled, and the gradients it was integrated from.");

    module.def(
        "waveforms_and_times",
        [](const Sequence& sequence,
           bool append_rf,
           int first_block,
           int last_block,
           double b0,
           double gamma) {
            pulseq::WaveformOptions options;
            options.append_rf = append_rf;
            options.first_block = first_block;
            options.last_block = last_block;
            options.b0 = b0;
            options.gamma = gamma;

            pulseq::Waveforms made;
            {
                py::gil_scoped_release unlocked;
                made = pulseq::waveforms_and_times(sequence, options);
            }

            /* Each channel as the 2-by-n array the toolboxes report: a row of
             * times over a row of amplitudes. */
            const auto paired = [](const std::vector<double>& t,
                                   const std::vector<double>& v) {
                const py::ssize_t held = static_cast<py::ssize_t>(t.size());
                py::array_t<double> out({static_cast<py::ssize_t>(2), held});
                auto view = out.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < held; ++i)
                {
                    view(0, i) = t[static_cast<size_t>(i)];
                    view(1, i) = v[static_cast<size_t>(i)];
                }
                return out;
            };

            const auto moments = [](const std::vector<pulseq::PulseMoment>& held) {
                const py::ssize_t count = static_cast<py::ssize_t>(held.size());
                py::array_t<double> out({static_cast<py::ssize_t>(3), count});
                auto view = out.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < count; ++i)
                {
                    view(0, i) = held[static_cast<size_t>(i)].time;
                    view(1, i) = held[static_cast<size_t>(i)].frequency;
                    view(2, i) = held[static_cast<size_t>(i)].phase;
                }
                return out;
            };

            py::list waves;
            for (int axis = 0; axis < 3; ++axis)
                waves.append(paired(made.times[static_cast<size_t>(axis)],
                                    made.amplitudes[static_cast<size_t>(axis)]));
            if (append_rf)
            {
                const py::ssize_t held = static_cast<py::ssize_t>(made.rf_times.size());
                py::array_t<std::complex<double>> rf(
                    {static_cast<py::ssize_t>(2), held});
                auto view = rf.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < held; ++i)
                {
                    view(0, i) = made.rf_times[static_cast<size_t>(i)];
                    view(1, i) = made.rf_signal[static_cast<size_t>(i)];
                }
                waves.append(rf);
            }

            const py::ssize_t samples =
                static_cast<py::ssize_t>(made.adc_times.size());
            py::array_t<double> fp({static_cast<py::ssize_t>(2), samples});
            {
                auto view = fp.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < samples; ++i)
                {
                    view(0, i) = made.adc_frequency[static_cast<size_t>(i)];
                    view(1, i) = made.adc_phase[static_cast<size_t>(i)];
                }
            }

            const py::ssize_t windows =
                static_cast<py::ssize_t>(made.window_frequency.size());
            py::array_t<double> window_fp({windows, static_cast<py::ssize_t>(2)});
            {
                auto view = window_fp.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < windows; ++i)
                {
                    view(i, 0) = made.window_frequency[static_cast<size_t>(i)];
                    view(i, 1) = made.window_phase[static_cast<size_t>(i)];
                }
            }

            py::dict out;
            out["wave_data"] = waves;
            out["window_fp"] = window_fp;
            out["duration"] = made.duration;
            out["tfp_excitation"] = moments(made.excitation);
            out["tfp_refocusing"] = moments(made.refocusing);
            out["t_adc"] = py::array_t<double>(samples, made.adc_times.data());
            out["fp_adc"] = fp;
            out["pm_adc"] = py::array_t<double>(samples, made.adc_modulation.data());
            out["warnings"] = made.warnings;
            return out;
        },
        py::arg("sequence"), py::arg("append_rf") = false, py::arg("first_block") = 1,
        py::arg("last_block") = 0, py::arg("b0") = 1.5, py::arg("gamma") = 42576000.0,
        "Expand the sequence into the gradient waveforms it plays, the RF "
        "moments, and the ADC sampling.");

    module.def(
        "write_binary",
        [](Sequence& sequence) {
            std::string written;
            {
                py::gil_scoped_release unlocked;
                written = pulseq::write_binary(sequence);
            }
            return py::bytes(written);
        },
        py::arg("sequence"), "Serialize as a Pulseq binary sequence file.");

    module.def(
        "is_binary",
        [](const py::bytes& contents) { return pulseq::is_binary(std::string(contents)); },
        py::arg("contents"), "Whether the bytes open with the binary magic.");

    module.def(
        "required_revision",
        [](const Sequence& sequence) { return pulseq::required_revision(sequence); },
        py::arg("sequence"),
               "The Pulseq revision the sequence's contents actually need.");

    module.def(
        "read",
        [](const py::bytes& contents, bool verify) {
            // The base is what read() builds; the bound sequence is what
            // Python holds, so the one is moved into the other.
            Sequence sequence;
            {
                const std::string text = contents;
                py::gil_scoped_release unlocked;
                static_cast<pulseq::Sequence&>(sequence) = pulseq::read(text, verify);
            }
            return sequence;
        },
        py::arg("contents"), py::arg("verify") = false,
        "Parse a Pulseq `.seq` file back into a Sequence.");

    module.def(
        "read_file",
        [](const std::string& path, bool verify) {
            Sequence sequence;
            {
                py::gil_scoped_release unlocked;
                static_cast<pulseq::Sequence&>(sequence) = pulseq::read_file(path, verify);
            }
            return sequence;
        },
        py::arg("path"), py::arg("verify") = false, "As read(), for a file on disk.");

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
