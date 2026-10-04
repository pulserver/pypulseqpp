/**
 * @file labels.h
 * @brief The sticky label state behind `sequences.Labels`, bound as
 *        `pypulseqpp._ext.LabelWriter`.
 */

#pragma once

#include <pybind11/pybind11.h>

void pypulseqpp_bind_labels(pybind11::module_& m);
