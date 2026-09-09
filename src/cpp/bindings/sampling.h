/**
 * @file sampling.h
 * @brief The sampling-pattern kernels, bound as `pypulseqpp._ext.sampling`.
 */

#pragma once

#include <pybind11/pybind11.h>

void pypulseqpp_bind_sampling(pybind11::module_& m);
