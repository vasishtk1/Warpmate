#include "egtb/notation.h"

#include <cctype>
#include <sstream>

#include "core/notation.h"

namespace wm::egtb {

namespace {

constexpr char kLetters[] = "KQRBNP";

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

}  // namespace

std::optional<Board> parse_fen(const std::string& fen, std::string* error) {
    std::istringstream in(fen);
    std::string placement, side;
    if (!(in >> placement >> side)) {
        set_error(error, "FEN needs at least a board and a side to move");
        return std::nullopt;
    }
    Board b;
    int rank = 7, file = 0;
    int kings[2] = {0, 0};
    for (char c : placement) {
        if (c == '/') {
            if (file != 8) {
                set_error(error, "each FEN rank must cover 8 squares");
                return std::nullopt;
            }
            --rank;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file += c - '0';
        } else {
            const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            int type = -1;
            for (int t = 0; t < 6; ++t) {
                if (kLetters[t] == upper) type = t;
            }
            if (type < 0) {
                set_error(error, std::string("unknown piece '") + c + "'");
                return std::nullopt;
            }
            if (rank < 0 || file > 7) {
                set_error(error, "FEN board is off the edge");
                return std::nullopt;
            }
            if (b.n == kMaxPieces) {
                set_error(error, "at most " + std::to_string(kMaxPieces) + " pieces supported");
                return std::nullopt;
            }
            const uint8_t color = std::isupper(static_cast<unsigned char>(c)) ? kWhite : kBlack;
            if (type == kKing) ++kings[color];
            b.type[b.n] = static_cast<uint8_t>(type);
            b.color[b.n] = color;
            b.sq[b.n] = static_cast<uint8_t>(make_square(file, rank));
            ++b.n;
            ++file;
        }
        if (file > 8) {
            set_error(error, "FEN rank has more than 8 squares");
            return std::nullopt;
        }
    }
    if (rank != 0 || file != 8) {
        set_error(error, "FEN board must have 8 ranks of 8 squares");
        return std::nullopt;
    }
    if (kings[0] != 1 || kings[1] != 1) {
        set_error(error, "each side needs exactly one king");
        return std::nullopt;
    }
    if (side != "w" && side != "b") {
        set_error(error, "side to move must be 'w' or 'b'");
        return std::nullopt;
    }
    b.stm = side == "w" ? kWhite : kBlack;
    return b;
}

std::string to_fen(const Board& b) {
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const int sq = make_square(file, rank);
            char c = 0;
            for (int i = 0; i < b.n; ++i) {
                if (b.sq[i] == sq) {
                    c = kLetters[b.type[i]];
                    if (b.color[i] == kBlack) c = static_cast<char>(std::tolower(c));
                }
            }
            if (c) {
                if (empty) out += static_cast<char>('0' + empty);
                empty = 0;
                out += c;
            } else {
                ++empty;
            }
        }
        if (empty) out += static_cast<char>('0' + empty);
        if (rank) out += '/';
    }
    out += b.stm == kWhite ? " w - - 0 1" : " b - - 0 1";
    return out;
}

void to_position(const Board& b, Material* m, Position* p) {
    Board s = b;
    sort_pieces(s.n, s.type, s.color, s.sq);
    m->n = s.n;
    for (int i = 0; i < s.n; ++i) {
        m->type[i] = s.type[i];
        m->color[i] = s.color[i];
        p->sq[i] = s.sq[i];
    }
    p->stm = s.stm;
}

Board to_board(const Material& m, const Position& p) {
    Board b;
    b.n = m.n;
    for (int i = 0; i < m.n; ++i) {
        b.type[i] = m.type[i];
        b.color[i] = m.color[i];
        b.sq[i] = p.sq[i];
    }
    b.stm = p.stm;
    return b;
}

Board apply_move(const Material& m, const Position& p, const Move& mv) {
    Board b;
    for (int i = 0; i < m.n; ++i) {
        if (i == mv.captured) continue;
        b.type[b.n] = (i == mv.piece && mv.promo != kNone) ? mv.promo : m.type[i];
        b.color[b.n] = m.color[i];
        b.sq[b.n] = i == mv.piece ? mv.to : p.sq[i];
        ++b.n;
    }
    b.stm = p.stm ^ 1;
    return b;
}

Value lookup(const SubTable* subs, int num_subs, Board b) {
    return lookup_pieces(subs, num_subs, b.n, b.type, b.color, b.sq, b.stm);
}

std::string move_to_string(const Material& m, const Position& p, const Move& mv) {
    std::string s;
    if (m.type[mv.piece] == kPawn) {
        if (mv.captured != kNone) {
            s += static_cast<char>('a' + file_of(mv.from));
            s += 'x';
        }
    } else {
        s += kLetters[m.type[mv.piece]];
        if (mv.captured != kNone) s += 'x';
    }
    s += square_name(mv.to);
    if (mv.promo != kNone) {
        s += '=';
        s += kLetters[mv.promo];
    }
    // Check / checkmate marker, worked out on the resulting board.
    const Board after = apply_move(m, p, mv);
    Material am;
    Position ap;
    to_position(after, &am, &ap);
    if (in_check(am, ap, ap.stm)) s += count_moves(am, ap) == 0 ? "#" : "+";
    return s;
}

std::string describe(uint8_t stm, Value v) {
    if (v == kInvalid) return "Invalid position";
    if (v == kDraw || v == kUnknown) return "Draw";
    const char* mover = stm == kWhite ? "White" : "Black";
    const char* other = stm == kWhite ? "Black" : "White";
    if (v == 0) return std::string(mover) + " is checkmated";
    std::ostringstream out;
    out << (is_win(v) ? mover : other) << " mates in " << moves_to_mate(v);
    return out.str();
}

}  // namespace wm::egtb
