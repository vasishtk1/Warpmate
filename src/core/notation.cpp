#include "core/notation.h"

#include <cctype>
#include <sstream>

namespace wm {

std::string square_name(int sq) {
    std::string s;
    s += static_cast<char>('a' + file_of(sq));
    s += static_cast<char>('1' + rank_of(sq));
    return s;
}

namespace krk {

namespace {

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

}  // namespace

std::optional<Position> parse_fen(const std::string& fen, std::string* error) {
    std::istringstream in(fen);
    std::string board, side;
    if (!(in >> board >> side)) {
        set_error(error, "FEN needs at least a board and a side to move");
        return std::nullopt;
    }

    int wk = -1, wr = -1, bk = -1;
    int rank = 7, file = 0;
    for (char c : board) {
        if (c == '/') {
            if (file != 8) {
                set_error(error, "each FEN rank must cover 8 squares");
                return std::nullopt;
            }
            --rank;
            file = 0;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            file += c - '0';
        } else {
            if (rank < 0 || file > 7) {
                set_error(error, "FEN board is off the edge");
                return std::nullopt;
            }
            const int sq = make_square(file, rank);
            int* slot = nullptr;
            if (c == 'K') slot = &wk;
            else if (c == 'R') slot = &wr;
            else if (c == 'k') slot = &bk;
            else {
                set_error(error, std::string("unsupported piece '") + c +
                                     "' (only K, R and k are allowed)");
                return std::nullopt;
            }
            if (*slot != -1) {
                set_error(error, std::string("more than one '") + c + "'");
                return std::nullopt;
            }
            *slot = sq;
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
    if (wk < 0 || wr < 0 || bk < 0) {
        set_error(error, "need exactly one white king (K), white rook (R) and black king (k)");
        return std::nullopt;
    }
    if (side != "w" && side != "b") {
        set_error(error, "side to move must be 'w' or 'b'");
        return std::nullopt;
    }
    return Position{wk, wr, bk, side == "w" ? kWhite : kBlack};
}

std::string to_fen(const Position& p) {
    std::string out;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const int sq = make_square(file, rank);
            char c = 0;
            if (sq == p.wk) c = 'K';
            else if (sq == p.wr) c = 'R';
            else if (sq == p.bk) c = 'k';
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
    out += p.stm == kWhite ? " w - - 0 1" : " b - - 0 1";
    return out;
}

std::string move_to_string(const Position& p, const Move& m) {
    std::string s = m.piece == kWhiteRook ? "R" : "K";
    if (m.capture) s += 'x';
    s += square_name(m.to);
    if (!m.capture) {
        const Position next = make_move(p, m);
        if (side_to_move_in_check(next)) {
            Move replies[kMaxMoves];
            s += generate_moves(next, replies) == 0 ? "#" : "+";
        }
    }
    return s;
}

std::string describe(const Position& p, Value v) {
    if (v == kInvalid) return "Invalid position";
    if (v == kDraw || v == kUnknown) return "Draw";
    const char* mover = p.stm == kWhite ? "White" : "Black";
    const char* other = p.stm == kWhite ? "Black" : "White";
    if (v == 0) return std::string(mover) + " is checkmated";
    std::ostringstream out;
    const int n = moves_to_mate(v);
    if (is_win(v)) out << mover << " mates in " << n;
    else out << other << " mates in " << n;
    return out.str();
}

}  // namespace krk
}  // namespace wm
