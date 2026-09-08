/**
 * @file pulseqpp_decode.h
 * @brief One block, as the events it plays rather than as the ids they are stored under.
 *
 * The core holds a block the way the file does: a row of library ids and a
 * duration. That is what makes a million-block scan cheap to build and to
 * write, and it is all `add_block` and the writers ever need. Everything that
 * *looks at* a sequence needs the events themselves.
 *
 * What comes back is the compiled event types, not namespaces, and that is the
 * point twice over. In Python they read like the events a factory hands
 * back -- `rf.signal`, `gx.waveform`, `gx.area`, `adc.dwell` -- because those
 * are computed from the same fields either way. Handed to `add_block` or
 * `set_block` they take the fast path, and the shape registration comes with
 * them, so reading a block out and putting it back registers nothing new.
 *
 * Nothing is converted on the way. The libraries store a pulse as a magnitude
 * in 0..1, a phase in turns and one amplitude, and an arbitrary gradient as a
 * normalised waveform and one amplitude; the compiled events store them the
 * same way. Decoding is decompressing the shapes and copying the row.
 */

#ifndef PULSERVER_PULSEQPP_DECODE_H
#define PULSERVER_PULSEQPP_DECODE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

#include "pulseq/events.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/shape.hpp"
#include "pulseqpp_eventtypes.h"
#include "pulseqpp_events.h"

namespace pulseqpp_decode
{
    namespace py = pybind11;

    /** A nanosecond, the grid a Pulseq file is written on. */
    constexpr double kEps = 1e-9;

    /** What a `use` code in the RF library stands for. */
    inline const char* use_name(char code)
    {
        switch (code)
        {
        case 'e':
            return "excitation";
        case 'r':
            return "refocusing";
        case 'i':
            return "inversion";
        case 's':
            return "saturation";
        case 'p':
            return "preparation";
        case 'o':
            return "other";
        default:
            return "undefined";
        }
    }

    inline std::vector<double> decompressed(const pulseq::ShapeLibrary& shapes, int id)
    {
        if (id < 1 || id > shapes.size())
            return {};
        return pulseq::decompress_shape(
            shapes.samples(id), shapes.num_compressed(id), shapes.num_uncompressed(id));
    }

    /** The C++ event inside a Python object one of the factories made. */
    template <typename T> T& inside(const py::object& made)
    {
        return *static_cast<T*>(pulseqpp_types::event_of(made.ptr()));
    }

    /**
     * One RF row as a pulse.
     *
     * A pulse laid out on the RF raster carries no time shape, and its samples
     * sit at the centre of each raster interval; one that carries a time shape
     * says where every sample is, and lasts until the raster boundary at or
     * past its last one.
     */
    inline py::object decode_rf(
        const pulseq::Sequence& seq,
        int id,
        int32_t serial,
        double rf_raster)
    {
        const double* row = seq.rf_library().row(id);
        const pulseq::ShapeLibrary& shapes = seq.shape_library();
        const int magnitude_shape = static_cast<int>(row[1]);
        const int phase_shape = static_cast<int>(row[2]);
        const int time_shape = static_cast<int>(row[3]);

        py::object made = pulseqpp_types::new_rf();
        pulseq::RfEvent& rf = inside<pulseq::RfEvent>(made);

        rf.amplitude = row[0];
        rf.magnitude = decompressed(shapes, magnitude_shape);
        rf.phase = decompressed(shapes, phase_shape);
        rf.center = row[4];
        rf.delay = row[5];
        rf.freq_ppm = row[6];
        rf.phase_ppm = row[7];
        rf.freq_offset = row[8];
        rf.phase_offset = row[9];

        if (time_shape > 0)
        {
            rf.t = decompressed(shapes, time_shape);
            for (size_t i = 0; i < rf.t.size(); ++i)
                rf.t[i] *= rf_raster;
            const double last = rf.t.empty() ? 0.0 : rf.t.back();
            rf.shape_dur = std::ceil((last - kEps) / rf_raster) * rf_raster;
        }
        else
        {
            // A pulse laid out on the raster carries no time shape; its
            // samples sit at the centre of each interval, and a caller
            // reading `t` wants those times rather than an empty array.
            // Registering it again recognises them and adds no shape.
            const size_t count = rf.magnitude.size();
            rf.t.resize(count);
            for (size_t i = 0; i < count; ++i)
                rf.t[i] = (static_cast<double>(i) + 0.5) * rf_raster;
            rf.shape_dur = static_cast<double>(count) * rf_raster;
        }

        const std::vector<char>& uses = seq.rf_uses();
        rf.use = use_name(
            id >= 1 && id <= static_cast<int>(uses.size())
                ? uses[static_cast<size_t>(id) - 1]
                : 'u');

        // The shapes are this sequence's own, so putting the pulse back
        // registers nothing.
        rf.registered.note(serial, magnitude_shape, time_shape > 0 ? time_shape : 0);
        rf.phase_shape = phase_shape;
        return made;
    }

    /** One gradient id as a trapezoid or as a waveform, on axis @p axis. */
    inline py::object decode_grad(
        const pulseq::Sequence& seq,
        int id,
        int axis,
        int32_t serial,
        double grad_raster)
    {
        const pulseq::Axis on = static_cast<pulseq::Axis>(axis);

        if (seq.grad_kind(id) == pulseq::GradKind::Trap)
        {
            const double* row = seq.trap_library().row(seq.grad_row(id));
            py::object made = pulseqpp_types::new_trap();
            pulseq::TrapEvent& trap = inside<pulseq::TrapEvent>(made);
            trap.amplitude = row[0];
            trap.rise_time = row[1];
            trap.flat_time = row[2];
            trap.fall_time = row[3];
            trap.delay = row[4];
            trap.axis = on;
            return made;
        }

        const double* row = seq.arb_library().row(seq.grad_row(id));
        const int amp_shape = static_cast<int>(row[3]);
        const int time_shape = static_cast<int>(row[4]);

        py::object made = pulseqpp_types::new_grad();
        pulseq::GradEvent& grad = inside<pulseq::GradEvent>(made);
        grad.amplitude = row[0];
        grad.waveform = decompressed(seq.shape_library(), amp_shape);
        grad.first = row[1];
        grad.last = row[2];
        grad.delay = row[5];
        grad.axis = on;

        // A grid the registration can name again is a formula, not data: 1 is
        // the raster centres, 2 the half-raster variant, and anything else is
        // a time shape that has to be carried sample by sample.
        const double count = static_cast<double>(grad.waveform.size());
        if (time_shape == 0)
        {
            grad.tt_grid = 1;
            grad.tt_raster = grad_raster;
            grad.last_time = (count - 0.5) * grad_raster;
        }
        else if (time_shape == -1)
        {
            grad.tt_grid = 2;
            grad.tt_raster = grad_raster;
            grad.last_time = count * 0.5 * grad_raster;
        }
        else
        {
            grad.tt = decompressed(seq.shape_library(), time_shape);
            for (size_t i = 0; i < grad.tt.size(); ++i)
                grad.tt[i] *= grad_raster;
            grad.last_time = grad.tt.empty() ? 0.0 : grad.tt.back();
        }

        grad.registered.note(serial, amp_shape, time_shape);
        return made;
    }

    /** One ADC row as a window. */
    inline py::object decode_adc(const pulseq::Sequence& seq, int id, int32_t serial)
    {
        const double* row = seq.adc_library().row(id);
        const int phase_shape = static_cast<int>(row[7]);

        py::object made = pulseqpp_types::new_adc();
        pulseq::AdcEvent& adc = inside<pulseq::AdcEvent>(made);
        adc.num_samples = row[0];
        adc.dwell = row[1];
        adc.delay = row[2];
        adc.freq_ppm = row[3];
        adc.phase_ppm = row[4];
        adc.freq_offset = row[5];
        adc.phase_offset = row[6];
        adc.phase_modulation = decompressed(seq.shape_library(), phase_shape);
        adc.registered.note(serial, phase_shape, 0);
        return made;
    }

    /** The extension chain a block heads, decoded into the block's fields. */
    inline void decode_extensions(
        const pulseq::Sequence& seq,
        int32_t head,
        int32_t serial,
        py::dict& block)
    {
        const pulseq::IntTable& links = seq.extensions_library();
        py::list triggers;
        py::list labels;

        int32_t node = head;
        while (node > 0 && node <= links.size())
        {
            const int32_t* link = links.row(node);
            const std::string kind = seq.extension_type_name(link[0]);
            const int32_t ref = link[1];

            if (kind == "TRIGGERS" && ref >= 1 && ref <= seq.trigger_library().size())
            {
                const double* row = seq.trigger_library().row(ref);
                py::object made = pulseqpp_types::new_trigger();
                pulseq::TriggerEvent& trigger = inside<pulseq::TriggerEvent>(made);
                trigger.control = row[0];
                trigger.channel = row[1];
                trigger.delay = row[2];
                trigger.duration_s = row[3];
                triggers.append(made);
            }
            else if (kind == "LABELSET" || kind == "LABELINC")
            {
                const bool setting = kind == "LABELSET";
                const pulseq::IntTable& lib =
                    setting ? seq.label_set_library() : seq.label_inc_library();
                if (ref >= 1 && ref <= lib.size())
                {
                    const int32_t* row = lib.row(ref);
                    py::object made = pulseqpp_types::new_label();
                    pulseq::LabelEvent& label = inside<pulseq::LabelEvent>(made);
                    label.value = row[0];
                    label.label = seq.label_name(row[1]);
                    label.setting = setting;
                    label.owner = serial;
                    label.label_id = row[1];
                    labels.append(made);
                }
            }
            else if (
                kind == "DELAYS" && ref >= 1 &&
                ref <= static_cast<int32_t>(seq.soft_delay_library().size()))
            {
                const pulseq::SoftDelay& row =
                    seq.soft_delay_library()[static_cast<size_t>(ref) - 1];
                py::object made = pulseqpp_types::new_soft_delay();
                pulseq::SoftDelayEvent& soft = inside<pulseq::SoftDelayEvent>(made);
                soft.num = row.num;
                soft.offset = row.offset;
                soft.factor = row.factor;
                soft.hint = row.hint;
                block["soft_delay"] = made;
            }
            else if (kind == "ROTATIONS" && ref >= 1 && ref <= seq.rotation_library().size())
            {
                const double* row = seq.rotation_library().row(ref);
                py::object made = pulseqpp_types::new_rotation();
                pulseq::RotationEvent& rotation = inside<pulseq::RotationEvent>(made);
                rotation.quaternion = {{row[0], row[1], row[2], row[3]}};
                block["rotation"] = made;
            }
            else if (kind == "RF_SHIMS" && ref >= 1 && ref <= seq.rf_shim_library().size())
            {
                // No compiled type carries a shim, so it goes back as the
                // namespace `make_rf_shim` hands out -- which is what both
                // a caller reading it and `add_block` expect.
                const double* row = seq.rf_shim_library().row(ref);
                const int length = seq.rf_shim_library().length(ref);
                std::vector<std::complex<double>> shim(static_cast<size_t>(length) / 2);
                for (size_t i = 0; i < shim.size(); ++i)
                    shim[i] = std::polar(row[2 * i], row[2 * i + 1]);
                py::object namespace_type =
                    py::module_::import("types").attr("SimpleNamespace");
                block["rf_shim"] = namespace_type(
                    py::arg("type") = "rf_shim",
                    py::arg("shim_vector") = py::array_t<std::complex<double>>(
                        static_cast<py::ssize_t>(shim.size()), shim.data()));
            }
            node = link[2];
        }

        if (!triggers.empty())
            block["trig"] = triggers;
        if (!labels.empty())
            block["label"] = labels;
    }

    /** Block @p index (1-based) as the events it plays. */
    inline py::dict decode_block(const pulseqpp_events::BoundSequence& seq, int index)
    {
        const pulseq::Block row = seq.get_block(index);
        const int32_t serial = seq.serial;
        const double rf_raster = seq.rf_raster_time();
        const double grad_raster = seq.grad_raster_time();

        py::dict block;
        block["block_duration"] = row.duration;
        block["rf"] = py::none();
        block["gx"] = py::none();
        block["gy"] = py::none();
        block["gz"] = py::none();
        block["adc"] = py::none();
        block["delay"] = py::none();
        block["label"] = py::none();
        block["soft_delay"] = py::none();

        if (row.rf > 0)
            block["rf"] = decode_rf(seq, row.rf, serial, rf_raster);

        static const char* const kAxis[3] = {"gx", "gy", "gz"};
        const int32_t gradients[3] = {row.gx, row.gy, row.gz};
        for (int axis = 0; axis < 3; ++axis)
        {
            if (gradients[axis] > 0)
                block[kAxis[axis]] =
                    decode_grad(seq, gradients[axis], axis, serial, grad_raster);
        }

        if (row.adc > 0)
            block["adc"] = decode_adc(seq, row.adc, serial);

        if (row.ext > 0)
            decode_extensions(seq, row.ext, serial, block);

        return block;
    }

} // namespace pulseqpp_decode

#endif /* PULSERVER_PULSEQPP_DECODE_H */
