/**
 * @file ptx.h
 * @brief The small-tip parallel-transmit design kernels, bound as `pypulseqpp._ext.ptx`.
 */

#pragma once

#include <pybind11/pybind11.h>

void pypulseqpp_bind_ptx(pybind11::module_& m);
