#ifndef STRING_TRANSFORMS_H
#define STRING_TRANSFORMS_H

#include <string>
#include "bitboard.h"
#include "move.h"
#include "state.h"
#include "types.h"

/* UCI and string operations*/
const std::string piece_strings[6] = { "n", "b", "r", "q", "k", "p" };
std::string PieceStringLower(Piece piece);
std::string SquareName(Square sqr);
Bitboard BitboardFromString(std::string str);

// Move as uci string
std::string AsUci(Move move);

/* Debugging, printing out */
void PrettyPrint(const State& state);

State StateFromFen(std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

#endif