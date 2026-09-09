/**
 * @file arbgrad.h
 * @brief The arbitrary-gradient solver, bound as `pypulseqpp._ext.arbgrad`.
 */

#pragma once

#include <pybind11/pybind11.h>

void pypulseqpp_bind_arbgrad(pybind11::module_& m);
