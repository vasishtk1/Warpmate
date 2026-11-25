// GPU (CUDA) version of the general endgame solver.
//
// Mirrors egtb::Solver: solving a material first solves everything it can
// turn into (by captures and promotions). Every solved table is kept in GPU
// memory, because larger endgames look up smaller ones from inside the
// kernels. A copy is also brought back to the CPU for saving and checking.
//
// No CUDA types appear here, so ordinary C++ files can include it.
#pragma once

#include <functional>
#include <map>
#include <string>

#include "egtb/solver.h"

namespace wm::egtb {

class GpuSolver {
public:
    GpuSolver();
    ~GpuSolver();
    GpuSolver(const GpuSolver&) = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    // Called when a table finishes (e.g. to print or save it).
    std::function<void(const TableResult&)> on_table;

    const TableResult& solve(const Material& m);

    // Tables already solved (CPU copies), keyed by name.
    const std::map<std::string, TableResult>& tables() const { return tables_; }

    // Time spent in GPU kernels for a table, measured with CUDA events.
    double kernel_ms(const std::string& name) const { return kernel_ms_.at(name); }

    // Name of the GPU in use, e.g. "Tesla T4".
    static std::string device_name();

private:
    TableResult solve_one(const Material& m);

    std::map<std::string, TableResult> tables_;
    std::map<std::string, Value*> device_tables_;  // stay on the GPU for lookups
    std::map<std::string, double> kernel_ms_;
};

}  // namespace wm::egtb
