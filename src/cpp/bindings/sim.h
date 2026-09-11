/**
 * @file sim.h
 * @brief The Bloch simulation kernel, bound as `pypulseqpp._ext.sim`.
 */

#pragma once

#include <pybind11/pybind11.h>

void pypulseqpp_bind_sim(pybind11::module_& m);
