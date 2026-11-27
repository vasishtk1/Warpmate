#include "egtb/solver.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>

#include "egtb/retro.h"
#include "egtb/unmove.h"

namespace wm::egtb {

namespace {

constexpr char kLetters[] = "KQRBNP";  // indexed by PieceType

int type_from_letter(char c) {
    for (int t = 0; t < 6; ++t) {
        if (kLetters[t] == c) return t;
    }
    return -1;
}

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

// Runs fn(begin, end, thread_id) over [0, total) split across threads.
template <typename Fn>
void parallel_for(uint64_t total, int threads, Fn fn) {
    std::vector<std::thread> pool;
    const uint64_t chunk = (total + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        const uint64_t begin = std::min<uint64_t>(total, t * chunk);
        const uint64_t end = std::min<uint64_t>(total, begin + chunk);
        pool.emplace_back(fn, begin, end, t);
    }
    for (auto& th : pool) th.join();
}

}  // namespace

std::string material_name(const Material& m) {
    std::string s;
    for (int i = 0; i < m.n; ++i) {
        if (i > 0 && m.type[i] == kKing && m.color[i] == kBlack) s += 'v';
        s += kLetters[m.type[i]];
    }
    return s;
}

bool parse_material(const std::string& name, Material* out, std::string* error) {
    const size_t v = name.find('v');
    if (v == std::string::npos || name.find('v', v + 1) != std::string::npos) {
        set_error(error, "material must look like KQvKR (white pieces, 'v', black pieces)");
        return false;
    }
    Material m{};
    m.n = 0;
    const std::string sides[2] = {name.substr(0, v), name.substr(v + 1)};
    for (int c = 0; c < 2; ++c) {
        int kings = 0;
        for (char ch : sides[c]) {
            const int t = type_from_letter(ch);
            if (t < 0) {
                set_error(error, std::string("unknown piece letter '") + ch + "'");
                return false;
            }
            if (t == kKing) ++kings;
            if (m.n == kMaxPieces) {
                set_error(error, "at most " + std::to_string(kMaxPieces) + " pieces supported");
                return false;
            }
            m.type[m.n] = static_cast<uint8_t>(t);
            m.color[m.n] = static_cast<uint8_t>(c);
            ++m.n;
        }
        if (kings != 1) {
            set_error(error, "each side needs exactly one king");
            return false;
        }
    }
    sort_pieces(m.n, m.type, m.color, nullptr);
    *out = m;
    return true;
}

Material canonical(Material m) {
    sort_pieces(m.n, m.type, m.color, nullptr);
    if (black_is_stronger(m)) {
        for (int i = 0; i < m.n; ++i) m.color[i] ^= 1;
        sort_pieces(m.n, m.type, m.color, nullptr);
    }
    return m;
}

std::vector<Material> direct_dependencies(const Material& m) {
    std::vector<Material> out;
    auto add = [&](Material d) {
        if (d.n <= 2) return;  // king vs king needs no table
        d = canonical(d);
        for (const Material& e : out) {
            if (same_material(e, d)) return;
        }
        out.push_back(d);
    };
    auto without = [&](const Material& src, int skip) {
        Material d{};
        d.n = 0;
        for (int i = 0; i < src.n; ++i) {
            if (i == skip) continue;
            d.type[d.n] = src.type[i];
            d.color[d.n] = src.color[i];
            ++d.n;
        }
        return d;
    };

    for (int i = 0; i < m.n; ++i) {
        if (m.type[i] != kKing) add(without(m, i));  // piece i captured
    }
    for (int i = 0; i < m.n; ++i) {
        if (m.type[i] != kPawn) continue;
        for (uint8_t promo = kQueen; promo <= kKnight; ++promo) {
            Material promoted = m;
            promoted.type[i] = promo;
            add(promoted);  // pawn i promotes
            for (int j = 0; j < m.n; ++j) {
                // pawn i promotes while capturing enemy piece j
                if (m.color[j] != m.color[i] && m.type[j] != kKing) {
                    add(without(promoted, j));
                }
            }
        }
    }
    return out;
}

TableSummary summarize(const Material& m, const std::vector<Value>& values) {
    TableSummary s;
    for (uint64_t i = 0; i < values.size(); ++i) {
        const Value v = values[i];
        if (v == kInvalid) {
            ++s.invalid;
            continue;
        }
        const int stm = decode(m, i).stm;
        if (is_win(v)) {
            ++s.wins[stm];
            if (v > s.longest_win) {
                s.longest_win = v;
                s.longest_win_index = i;
            }
        } else if (is_loss(v)) {
            ++s.losses[stm];
        } else {
            ++s.draws[stm];
        }
    }
    return s;
}

std::vector<SubTable> Solver::sub_tables(const Material& m) const {
    std::vector<SubTable> subs;
    for (const Material& d : direct_dependencies(m)) {
        auto it = tables_.find(material_name(d));
        if (it == tables_.end()) {
            throw std::runtime_error("sub-table " + material_name(d) + " not solved");
        }
        subs.push_back(SubTable{d, it->second.values.data()});
    }
    return subs;
}

const TableResult& Solver::solve(const Material& requested) {
    const Material m = canonical(requested);
    const std::string name = material_name(m);
    auto it = tables_.find(name);
    if (it != tables_.end()) return it->second;

    // Solve everything this endgame can turn into first.
    for (const Material& d : direct_dependencies(m)) solve(d);

    TableResult result = solve_one(m);
    auto& stored = tables_[name] = std::move(result);
    if (on_table) on_table(stored);
    return stored;
}

const char* algorithm_name(Algorithm a) {
    return a == Algorithm::kFullScan ? "full scan" : "frontier";
}

TableResult Solver::solve_one(const Material& m) {
    const int threads =
        threads_ > 0 ? threads_ : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const auto start = std::chrono::steady_clock::now();

    const std::vector<SubTable> subs = sub_tables(m);
    const Context ctx{m, subs.data(), static_cast<int>(subs.size())};

    // The largest distance in any sub-table: rounds must continue at least
    // this far, because a child in a sub-table can decide a position late.
    int max_sub = 0;
    for (const SubTable& st : subs) {
        const uint64_t size = table_size(st.m);
        for (uint64_t i = 0; i < size; ++i) {
            if (is_decided(st.values[i])) max_sub = std::max<int>(max_sub, st.values[i]);
        }
    }

    TableResult result;
    result.m = m;
    const uint64_t size = table_size(m);
    result.values.resize(size);
    Value* table = result.values.data();
    parallel_for(size, threads, [&](uint64_t begin, uint64_t end, int) {
        for (uint64_t i = begin; i < end; ++i) table[i] = init_value(ctx, i);
    });

    if (algorithm_ == Algorithm::kFullScan) {
        solve_full_scan(ctx, max_sub, threads, &result);
    } else {
        solve_frontier(ctx, max_sub, threads, &result);
    }

    for (Value& v : result.values) {
        if (v == kUnknown) v = kDraw;
    }
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

// The original algorithm: every round checks every unsolved position.
void Solver::solve_full_scan(const Context& ctx, int max_sub, int threads, TableResult* result) {
    const uint64_t size = result->values.size();
    // Rounds read one buffer and write a copy. Updating in place is safe in
    // principle (see the parity argument in docs/solver.md), but threads
    // reading and writing the same bytes at once is undefined behaviour in
    // standard C++, so the CPU version double-buffers. The GPU solver updates
    // in place.
    std::vector<Value> next(size);
    std::vector<RoundStats> per_thread(threads);
    int ply = 1;
    for (; ply < kDraw; ++ply) {
        next = result->values;
        const Value* src = result->values.data();
        Value* dst = next.data();
        parallel_for(size, threads, [&](uint64_t begin, uint64_t end, int t) {
            RoundStats stats;
            for (uint64_t i = begin; i < end; ++i) {
                if (src[i] != kUnknown) continue;
                ++stats.checked;
                const Value v = retro_step(ctx, src, i, ply);
                if (v != kUnknown) {
                    dst[i] = v;
                    ++stats.solved;
                }
            }
            per_thread[t] = stats;
        });
        result->values.swap(next);
        RoundStats total;
        for (const RoundStats& s : per_thread) {
            total.checked += s.checked;
            total.solved += s.solved;
        }
        result->round_stats.push_back(total);
        if (total.solved == 0 && ply > max_sub + 1) break;
    }
    result->rounds = ply;
}

// The faster algorithm: a round only checks positions that can change in it
// (see "Frontier solving" in retro.h). Produces exactly the same table.
void Solver::solve_frontier(const Context& ctx, int max_sub, int threads, TableResult* result) {
    const uint64_t size = result->values.size();
    if (size > UINT32_MAX) throw std::runtime_error("table too large for 32-bit position lists");
    Value* table = result->values.data();

    // Rounds at which a capture or promotion can decide each position.
    std::vector<uint8_t> win_round(size, 0), loss_round(size, 0);
    // mark[i] == ply: a child of position i was solved in round ply-1.
    // Several threads may mark the same parent at once, so the marks are
    // atomic (relaxed: only the value matters, not the ordering).
    std::unique_ptr<std::atomic<uint8_t>[]> mark(new std::atomic<uint8_t>[size]);
    parallel_for(size, threads, [&](uint64_t begin, uint64_t end, int) {
        for (uint64_t i = begin; i < end; ++i) {
            mark[i].store(0, std::memory_order_relaxed);
            if (table[i] == kUnknown && ctx.num_subs > 0) {
                sub_triggers(ctx, table, i, &win_round[i], &loss_round[i]);
            }
        }
    });

    // Positions solved in the previous round. Round 0 solved the checkmates.
    std::vector<uint32_t> frontier;
    for (uint64_t i = 0; i < size; ++i) {
        if (table[i] == 0) frontier.push_back(static_cast<uint32_t>(i));
    }

    std::vector<std::vector<uint32_t>> solved(threads);
    std::vector<uint64_t> checked(threads);
    int ply = 1;
    for (; ply < kDraw; ++ply) {
        // 1. Mark the parents of everything solved last round.
        const auto round = static_cast<uint8_t>(ply);
        parallel_for(frontier.size(), threads, [&](uint64_t begin, uint64_t end, int) {
            struct MarkParents {
                const Material* m;
                const Value* table;
                std::atomic<uint8_t>* mark;
                uint8_t round;
                bool operator()(const Position& p) {
                    const uint64_t index = encode(*m, p);
                    if (table[index] == kUnknown) mark[index].store(round, std::memory_order_relaxed);
                    return true;
                }
            } visit{&ctx.m, table, mark.get(), round};
            for (uint64_t k = begin; k < end; ++k) {
                for_each_unmove(ctx.m, decode(ctx.m, frontier[k]), visit);
            }
        });

        // 2. Check only the positions that can change this round. Results
        // are collected first and written afterwards, so no thread reads a
        // value written in this round.
        parallel_for(size, threads, [&](uint64_t begin, uint64_t end, int t) {
            solved[t].clear();
            uint64_t count = 0;
            for (uint64_t i = begin; i < end; ++i) {
                if (table[i] != kUnknown) continue;
                if (mark[i].load(std::memory_order_relaxed) != round && win_round[i] != round &&
                    loss_round[i] != round) {
                    continue;
                }
                ++count;
                if (retro_step(ctx, table, i, ply) != kUnknown) {
                    solved[t].push_back(static_cast<uint32_t>(i));
                }
            }
            checked[t] = count;
        });

        // 3. Write this round's results; they become the next frontier.
        RoundStats total;
        frontier.clear();
        for (int t = 0; t < threads; ++t) {
            total.checked += checked[t];
            for (uint32_t i : solved[t]) {
                table[i] = round;
                frontier.push_back(i);
            }
        }
        total.solved = frontier.size();
        result->round_stats.push_back(total);
        if (total.solved == 0 && ply > max_sub + 1) break;
    }
    result->rounds = ply;
}

std::vector<Material> all_three_piece() {
    std::vector<Material> out;
    for (const char* name : {"KQvK", "KRvK", "KBvK", "KNvK", "KPvK"}) {
        Material m;
        parse_material(name, &m);
        out.push_back(m);
    }
    return out;
}

std::vector<Material> all_four_piece() {
    const std::string pieces = "QRBNP";  // strongest first
    std::vector<Material> out;
    const auto add = [&](const std::string& name) {
        Material m;
        parse_material(name, &m);
        out.push_back(canonical(m));
    };
    for (size_t i = 0; i < pieces.size(); ++i) {
        for (size_t j = i; j < pieces.size(); ++j) {
            add(std::string("K") + pieces[i] + pieces[j] + "vK");  // e.g. KQRvK
            if (pieces[i] == 'P' && pieces[j] == 'P') continue;   // KPvKP: needs en passant
            add(std::string("K") + pieces[i] + "vK" + pieces[j]);  // e.g. KQvKR
        }
    }
    return out;
}

}  // namespace wm::egtb
