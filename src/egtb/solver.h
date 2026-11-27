// CPU driver for the general solver: material names, dependencies, and
// solving a whole family of endgames in the right order.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "core/values.h"
#include "egtb/material.h"
#include "egtb/position.h"

namespace wm::egtb {

// "KQvKR" <-> Material. parse_material returns false on bad input, and puts
// the pieces in canonical order (but does not colour-flip).
std::string material_name(const Material& m);
bool parse_material(const std::string& name, Material* out, std::string* error = nullptr);

// Canonical form of a material: canonical order, stronger side as White.
Material canonical(Material m);

// Every material reachable from `m` by one capture or promotion, in
// canonical form, excluding king vs king.
std::vector<Material> direct_dependencies(const Material& m);

// How each round chooses which positions to check.
enum class Algorithm {
    kFullScan,  // check every unsolved position, every round (the original)
    kFrontier,  // check only positions that can change this round (see retro.h)
};

const char* algorithm_name(Algorithm a);

// What one round did: how many positions it ran the per-position step on,
// and how many of those it solved.
struct RoundStats {
    uint64_t checked = 0;
    uint64_t solved = 0;
};

struct TableResult {
    Material m;
    std::vector<Value> values;
    int rounds = 0;
    double seconds = 0.0;
    std::vector<RoundStats> round_stats;  // one entry per round, round 1 first
};

struct TableSummary {
    uint64_t wins[2] = {0, 0};    // by side to move
    uint64_t draws[2] = {0, 0};
    uint64_t losses[2] = {0, 0};
    uint64_t invalid = 0;
    Value longest_win = 0;
    uint64_t longest_win_index = 0;
};

TableSummary summarize(const Material& m, const std::vector<Value>& values);

// Solves endgames on the CPU, remembering finished tables so each is solved
// once. Solving a material first solves everything it depends on.
class Solver {
public:
    explicit Solver(int threads = 0, Algorithm algorithm = Algorithm::kFrontier)
        : threads_(threads), algorithm_(algorithm) {}

    // Called when a table finishes (e.g. to print or save it).
    std::function<void(const TableResult&)> on_table;

    const TableResult& solve(const Material& m);

    // Tables already solved, keyed by name.
    const std::map<std::string, TableResult>& tables() const { return tables_; }

    // Sub-tables needed to look up children of `m` (all must be solved).
    std::vector<SubTable> sub_tables(const Material& m) const;

private:
    TableResult solve_one(const Material& m);
    void solve_full_scan(const Context& ctx, int max_sub, int threads, TableResult* result);
    void solve_frontier(const Context& ctx, int max_sub, int threads, TableResult* result);
    int threads_;
    Algorithm algorithm_;
    std::map<std::string, TableResult> tables_;
};

// Every 3-piece endgame (KQvK, KRvK, KBvK, KNvK, KPvK).
std::vector<Material> all_three_piece();

// Every 4-piece endgame except KPvKP (29 of the 30): two pieces against a
// lone king (KQQvK ... KPPvK) and one piece against one (KQvKQ ... KNvKP).
// KPvKP is left out because pawns on both sides need en passant, which the
// move generator doesn't handle yet.
std::vector<Material> all_four_piece();

}  // namespace wm::egtb
