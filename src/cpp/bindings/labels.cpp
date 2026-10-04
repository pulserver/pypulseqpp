/**
 * @file labels.cpp
 * @brief The sticky label state behind `sequences.Labels`.
 *
 * A sequence asks for its labels once per block, so the call is a plain
 * CPython type rather than a pybind11 class: keyword arguments arrive as the
 * dict the interpreter builds, with no conversion layer in between. Events
 * are made by the `make_label` the writer was constructed with, once per
 * (label, kind, value), and handed back as the same object afterwards.
 */

#include "labels.h"

#include <Python.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace
{
    struct Entry
    {
        PyObject* name = nullptr;  // owned
        long long last = 0;
        long long step = 0;
        bool has_last = false;
        bool has_step = false;
        /** (value << 1 | is_inc) -> event, owned. */
        std::unordered_map<long long, PyObject*> events;
    };

    struct LabelWriter
    {
        PyObject_HEAD
        PyObject* make_label;  // owned
        std::vector<Entry>* entries;
    };

    Entry& entry_for(LabelWriter* self, PyObject* name)
    {
        std::vector<Entry>& entries = *self->entries;
        for (Entry& e : entries)
            if (e.name == name)
                return e;
        for (Entry& e : entries)
            if (PyUnicode_Compare(e.name, name) == 0)
                return e;
        entries.emplace_back();
        Py_INCREF(name);
        entries.back().name = name;
        return entries.back();
    }

    PyObject* event_for(LabelWriter* self, Entry& e, bool inc, long long value)
    {
        const long long key = (value << 1) | (inc ? 1 : 0);
        auto found = e.events.find(key);
        if (found != e.events.end())
            return found->second;
        PyObject* event = PyObject_CallFunction(
            self->make_label, "OsL", e.name, inc ? "INC" : "SET", value);
        if (event == nullptr)
            return nullptr;
        e.events.emplace(key, event);
        return event;
    }

    void restart(LabelWriter* self)
    {
        for (Entry& e : *self->entries)
            e.has_last = e.has_step = false;
    }

    /** Integer value of a label argument; false with an exception set. */
    bool label_value(PyObject* item, long long& value)
    {
        PyObject* number = PyNumber_Index(item);
        if (number == nullptr)
        {
            PyErr_Clear();
            number = PyNumber_Long(item);
        }
        if (number == nullptr)
            return false;
        value = PyLong_AsLongLong(number);
        Py_DECREF(number);
        return !(value == -1 && PyErr_Occurred());
    }

    /** A changed ONCE value forgets every label's last value first. */
    bool apply_once(LabelWriter* self, PyObject* kwargs)
    {
        PyObject* once = PyDict_GetItemString(kwargs, "ONCE");
        if (once == nullptr)
            return true;
        long long value = 0;
        if (!label_value(once, value))
            return false;
        PyObject* key = PyUnicode_InternFromString("ONCE");
        Entry& e = entry_for(self, key);
        Py_DECREF(key);
        if (!e.has_last || e.last != value)
            restart(self);
        return true;
    }

    /** Appends the event that moves the label to value, if it changes. */
    bool append_label(LabelWriter* self, Entry& e, long long value, PyObject* events)
    {
        if (e.has_last && e.last == value)
            return true;
        const bool inc = e.has_last && e.has_step && value - e.last == e.step;
        PyObject* event = event_for(self, e, inc, inc ? e.step : value);
        if (event == nullptr || PyList_Append(events, event) < 0)
            return false;
        e.has_step = e.has_last;
        e.step = value - e.last;
        e.last = value;
        e.has_last = true;
        return true;
    }

    PyObject* writer_call(PyObject* object, PyObject* args, PyObject* kwargs)
    {
        auto* self = reinterpret_cast<LabelWriter*>(object);
        if (args != nullptr && PyTuple_GET_SIZE(args) != 0)
        {
            PyErr_SetString(PyExc_TypeError, "labels are passed by name");
            return nullptr;
        }
        PyObject* events = PyList_New(0);
        if (events == nullptr || kwargs == nullptr)
            return events;
        if (!apply_once(self, kwargs))
        {
            Py_DECREF(events);
            return nullptr;
        }
        Py_ssize_t position = 0;
        PyObject* name;
        PyObject* item;
        while (PyDict_Next(kwargs, &position, &name, &item))
        {
            long long value = 0;
            if (!label_value(item, value) || !append_label(self, entry_for(self, name), value, events))
            {
                Py_DECREF(events);
                return nullptr;
            }
        }
        return events;
    }

    PyObject* writer_restart(PyObject* object, PyObject*)
    {
        restart(reinterpret_cast<LabelWriter*>(object));
        Py_RETURN_NONE;
    }

    int writer_init(PyObject* object, PyObject* args, PyObject* kwargs)
    {
        auto* self = reinterpret_cast<LabelWriter*>(object);
        PyObject* make_label = nullptr;
        static const char* keywords[] = {"make_label", nullptr};
        if (!PyArg_ParseTupleAndKeywords(
                args, kwargs, "O", const_cast<char**>(keywords), &make_label))
            return -1;
        Py_INCREF(make_label);
        Py_XSETREF(self->make_label, make_label);
        return 0;
    }

    PyObject* writer_new(PyTypeObject* type, PyObject*, PyObject*)
    {
        auto* self = reinterpret_cast<LabelWriter*>(type->tp_alloc(type, 0));
        if (self == nullptr)
            return nullptr;
        self->make_label = nullptr;
        self->entries = new std::vector<Entry>();
        return reinterpret_cast<PyObject*>(self);
    }

    void writer_dealloc(PyObject* object)
    {
        auto* self = reinterpret_cast<LabelWriter*>(object);
        PyTypeObject* type = Py_TYPE(object);
        if (self->entries != nullptr)
        {
            for (Entry& e : *self->entries)
            {
                Py_XDECREF(e.name);
                for (auto& event : e.events)
                    Py_DECREF(event.second);
            }
            delete self->entries;
        }
        Py_XDECREF(self->make_label);
        type->tp_free(object);
        Py_DECREF(type);
    }

    PyMethodDef writer_methods[] = {
        {"_restart", writer_restart, METH_NOARGS, "Forget every label's last value."},
        {nullptr, nullptr, 0, nullptr},
    };

    PyType_Slot writer_slots[] = {
        {Py_tp_new, reinterpret_cast<void*>(writer_new)},
        {Py_tp_init, reinterpret_cast<void*>(writer_init)},
        {Py_tp_dealloc, reinterpret_cast<void*>(writer_dealloc)},
        {Py_tp_call, reinterpret_cast<void*>(writer_call)},
        {Py_tp_methods, writer_methods},
        {0, nullptr},
    };

    PyType_Spec writer_spec = {
        "pypulseqpp._ext.LabelWriter",
        sizeof(LabelWriter),
        0,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        writer_slots,
    };
} // namespace

void pypulseqpp_bind_labels(pybind11::module_& m)
{
    PyObject* type = PyType_FromSpec(&writer_spec);
    if (type == nullptr)
        throw pybind11::error_already_set();
    m.add_object("LabelWriter", pybind11::reinterpret_steal<pybind11::object>(type));
}
