/**
 * @file pulseqpp_eventtypes.h
 * @brief CPython event types with direct scalar fields and computed waveform properties.
 *
 * Scalar fields use PyMemberDef offsets into Holder<T>; non-scalars use
 * accessors. Setters do not validate hardware limits. Assigning a waveform
 * renormalises its samples and invalidates its shape registration; changing
 * only its amplitude preserves the registration.
 */

#ifndef PULSERVER_PULSEQPP_EVENTTYPES_H
#define PULSERVER_PULSEQPP_EVENTTYPES_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulseq/events.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseqpp_types
{
    namespace py = pybind11;
    using namespace pulseq;

    /**
     * A full turn in radians.
     *
     * `M_PI` is not standard C++: MSVC defines it only behind
     * `_USE_MATH_DEFINES`, set before the first `<cmath>` anywhere in the
     * translation unit, which a header cannot promise.
     */
    inline constexpr double TWO_PI = 6.283185307179586476925286766559;

    /* ================================================================== */
    /*  Normalisation                                                     */
    /* ================================================================== */

    /** Split a complex RF waveform into an amplitude, a magnitude and a phase. */
    inline void normalise_rf(const std::complex<double>* values, int count, RfEvent& out)
    {
        out.magnitude.resize(static_cast<size_t>(count));
        out.phase.resize(static_cast<size_t>(count));
        double peak = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double m = std::abs(values[i]);
            if (m > peak)
                peak = m;
        }
        out.amplitude = peak;
        for (int i = 0; i < count; ++i)
        {
            out.magnitude[static_cast<size_t>(i)] = peak > 0.0 ? std::abs(values[i]) / peak : 0.0;
            double angle = std::arg(values[i]);
            if (angle < 0.0)
                angle += TWO_PI;
            out.phase[static_cast<size_t>(i)] = angle / TWO_PI;
        }
        out.registered = Registration{};
    }

    /**
     * Split a gradient waveform into an amplitude and a normalised shape.
     *
     * The amplitude carries the sign of the first non-zero sample, which is
     * Pulseq's convention and what keeps a waveform and its negation one
     * shape rather than two.
     */
    /** Split @p values into a signed amplitude -- the largest magnitude,
     *  carrying the sign of the first nonzero sample -- and the waveform
     *  divided by it. Pass @p peak_abs when the largest magnitude is already
     *  known (a negative value asks for it to be measured). */
    inline void normalise_grad(
        const double* values,
        int count,
        GradEvent& out,
        double peak_abs = -1.0)
    {
        double peak = peak_abs;
        if (peak < 0.0)
        {
            peak = 0.0;
            for (int i = 0; i < count; ++i)
            {
                const double m = std::fabs(values[i]);
                if (m > peak)
                    peak = m;
            }
        }
        if (peak > 0.0)
        {
            for (int i = 0; i < count; ++i)
            {
                if (values[i] != 0.0)
                {
                    if (values[i] < 0.0)
                        peak = -peak;
                    break;
                }
            }
        }
        out.amplitude = peak;
        out.view = nullptr;
        out.view_count = 0;
        out.view_divisor = 1.0;
        out.waveform.assign(values, values + count);
        if (peak != 0.0)
        {
            double* w = out.waveform.data();
            for (int i = 0; i < count; ++i)
                w[i] = w[i] / peak;
        }
        out.registered = Registration{};
    }

    /* ================================================================== */
    /*  Duck-typed views                                                  */
    /* ================================================================== */

    inline py::array_t<std::complex<double>> rf_signal(const RfEvent& rf)
    {
        const size_t n = rf.magnitude.size();
        py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(n));
        std::complex<double>* data = out.mutable_data();
        for (size_t i = 0; i < n; ++i)
        {
            const double turns = TWO_PI * rf.phase[i];
            data[i] = std::polar(rf.amplitude * rf.magnitude[i], turns);
        }
        return out;
    }

    /** The signed peak normalise_grad() divides by: the largest magnitude,
     *  carrying the sign of the first nonzero sample. */
    inline double grad_signed_peak(const double* values, int count, double peak_abs = -1.0)
    {
        double peak = peak_abs;
        if (peak < 0.0)
        {
            peak = 0.0;
            for (int i = 0; i < count; ++i)
            {
                const double m = std::fabs(values[i]);
                if (m > peak)
                    peak = m;
            }
        }
        if (peak > 0.0)
        {
            for (int i = 0; i < count; ++i)
            {
                if (values[i] != 0.0)
                {
                    if (values[i] < 0.0)
                        peak = -peak;
                    break;
                }
            }
        }
        return peak;
    }

    /** Give @p out a view of @p values in place of a copy: the same amplitude
     *  normalise_grad() would set, the shape left to be divided out where it
     *  is read. The caller keeps @p values alive while the view stands. */
    inline void view_grad(const double* values, int count, GradEvent& out, double peak_abs = -1.0)
    {
        const double peak = grad_signed_peak(values, count, peak_abs);
        out.amplitude = peak;
        out.waveform.clear();
        out.view = values;
        out.view_count = count;
        out.view_divisor = peak != 0.0 ? peak : 1.0;
        out.registered = Registration{};
    }

    inline int grad_count(const GradEvent& g)
    {
        return g.view ? g.view_count : static_cast<int>(g.waveform.size());
    }

    /** The normalised shape sample @p i, from the view or the copy. */
    inline double grad_shape_at(const GradEvent& g, int i)
    {
        return g.view ? g.view[i] / g.view_divisor : g.waveform[static_cast<size_t>(i)];
    }

    /** Own the samples: a view is copied in and divided out, once. */
    inline void materialise_grad(GradEvent& g)
    {
        if (!g.view)
            return;
        std::vector<double> owned(static_cast<size_t>(g.view_count));
        for (int i = 0; i < g.view_count; ++i)
            owned[static_cast<size_t>(i)] = g.view[i] / g.view_divisor;
        g.waveform = std::move(owned);
        g.view = nullptr;
        g.view_count = 0;
        g.view_divisor = 1.0;
    }

    inline py::array_t<double> grad_waveform(const GradEvent& g)
    {
        const int n = grad_count(g);
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        double* data = out.mutable_data();
        for (int i = 0; i < n; ++i)
            data[i] = g.amplitude * grad_shape_at(g, i);
        return out;
    }

    /** The sample times an event was built on, materialised on the raster. */
    inline py::array_t<double> raster_times(
        const std::vector<double>& stored,
        int count,
        double raster,
        double offset)
    {
        if (!stored.empty())
            return py::array_t<double>(static_cast<py::ssize_t>(stored.size()), stored.data());
        py::array_t<double> out(count);
        double* data = out.mutable_data();
        for (int i = 0; i < count; ++i)
            data[i] = (i + offset) * raster;
        return out;
    }

    inline std::vector<double> as_vector(const py::object& source)
    {
        if (source.is_none())
            return {};
        auto array =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(source);
        const double* data = array.data();
        return std::vector<double>(data, data + array.size());
    }

    /* ================================================================== */
    /*  The instance layout                                               */
    /* ================================================================== */

    /**
     * A Python object with the event *in* it.
     *
     * Inline, not behind a pointer, because that is what makes a field a
     * `PyMemberDef`: the offset a member descriptor holds is measured from
     * the head of the instance, so there can be no indirection between them.
     */
    template <typename T> struct Holder
    {
        PyObject_HEAD T event;

        /**
         * Compatibility attributes assigned by PyPulseq scripts, not registration keys.
         *
         * Unset attributes raise AttributeError so upstream hasattr-based checks do
         * not mistake them for existing registrations. Keep them after event to
         * preserve the scalar field offsets.
         */
        PyObject* compat_id;
        PyObject* compat_shape_ids;
        /** What a gradient's `view` points into, kept alive by the holder. */
        PyObject* view_source;
    };

/* Every event derives from Event, so none of them is standard-layout and
 * offsetof is formally off-limits.  It is nevertheless exactly right for
 * single, non-virtual inheritance, which is what these are; bind() proves it
 * at import against a live object rather than trusting the compiler. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

    /** Where the C++ object starts.  The same for all nine -- see bind(). */
    constexpr Py_ssize_t EVENT_OFFSET = static_cast<Py_ssize_t>(offsetof(Holder<RfEvent>, event));

    /** A field's offset from the head of the instance. */
#define PULSEQPP_FIELD(EventType, member) \
    (static_cast<Py_ssize_t>(offsetof(Holder<EventType>, event) + offsetof(EventType, member)))

    /** The same, for a field of the holder rather than of the event. */
#define PULSEQPP_HOLDER_FIELD(EventType, member) \
    (static_cast<Py_ssize_t>(offsetof(Holder<EventType>, member)))

/**
 * `id` and `shape_IDs`, the two attributes PyPulseq's `register_*_event`
 * protocol assigns to an event.  `Py_T_OBJECT_EX` is what gives them
 * ordinary attribute behaviour: unset reads raise AttributeError, so
 * `hasattr` answers False until something assigns, which is what upstream's
 * own registration branches on.
 */
#define PULSEQPP_COMPAT_MEMBERS(EventType) \
    {"id", \
     Py_T_OBJECT_EX, \
     PULSEQPP_HOLDER_FIELD(EventType, compat_id), \
     0, \
     PyDoc_STR("Set by PyPulseq's register_*_event protocol; not read here.")}, \
    { \
        "shape_IDs", Py_T_OBJECT_EX, PULSEQPP_HOLDER_FIELD(EventType, compat_shape_ids), 0, \
            PyDoc_STR("Set by PyPulseq's register_*_event protocol; not read here.") \
    }

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    /** The base the nine event types share; none of them is subclassable. */
    extern PyTypeObject EventBaseType;

    /** Is this one of ours?  One dereference, because the nine are leaves. */
    inline bool is_event(PyObject* object)
    {
        return Py_TYPE(object)->tp_base == &EventBaseType;
    }

    inline Event* event_of(PyObject* object)
    {
        return reinterpret_cast<Event*>(reinterpret_cast<char*>(object) + EVENT_OFFSET);
    }

    /**
     * An empty event of each kind, for a caller that is about to fill it in.
     *
     * What comes back is the same type a factory hands back, so an event
     * built this way goes into `add_block` on the fast path and reads back in
     * Python with its fields in slots.
     */
    py::object new_rf();
    py::object new_trap();
    py::object new_grad();
    py::object new_adc();
    py::object new_label();
    py::object new_trigger();
    py::object new_rotation();
    py::object new_soft_delay();
    py::object new_delay();

    void bind(py::module_& m);

} // namespace pulseqpp_types

#endif /* PULSERVER_PULSEQPP_EVENTTYPES_H */
