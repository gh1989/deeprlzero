#ifndef SQUARE_H
#define SQUARE_H

#include "bitboard.h"
#include "geometry.h"
#include "types.h"
#include <map>

/* Square operations */
Bitboard GetSquare(Square sqr);
template<typename Square, typename... SqArgs>
static Bitboard GetSquare(Square sqr, SqArgs... others) { return GetSquare(sqr) | GetSquare(others...); }
// Move a square
Bitboard BitMove(Bitboard bb, Square from, Square to);
// Check if a square is on 
bool IsOn(Bitboard bb, Square square);
// Turn a square off
Bitboard OffBit(Bitboard bb, Square off);
// Turn a square on
Bitboard OnBit(Bitboard bb, Square on);
// Convert bitboard to square
Square BbSqr(Bitboard bb);
// Convert square to bitboard
Bitboard SqrBb(Square sqr);

static const std::map<unsigned long long, Square> square_lookup = {
	{ 0x0000000000000001ULL, a1 },
	{ 0x0000000000000002ULL, b1 },
	{ 0x0000000000000004ULL, c1 },
	{ 0x0000000000000008ULL, d1 },
	{ 0x0000000000000010ULL, e1 },
	{ 0x0000000000000020ULL, f1 },
	{ 0x0000000000000040ULL, g1 },
	{ 0x0000000000000080ULL, h1 },
	{ 0x0000000000000100ULL, a2 },
	{ 0x0000000000000200ULL, b2 },
	{ 0x0000000000000400ULL, c2 },
	{ 0x0000000000000800ULL, d2 },
	{ 0x0000000000001000ULL, e2 },
	{ 0x0000000000002000ULL, f2 },
	{ 0x0000000000004000ULL, g2 },
	{ 0x0000000000008000ULL, h2 },
	{ 0x0000000000010000ULL, a3 },
	{ 0x0000000000020000ULL, b3 },
	{ 0x0000000000040000ULL, c3 },
	{ 0x0000000000080000ULL, d3 },
	{ 0x0000000000100000ULL, e3 },
	{ 0x0000000000200000ULL, f3 },
	{ 0x0000000000400000ULL, g3 },
	{ 0x0000000000800000ULL, h3 },
	{ 0x0000000001000000ULL, a4 },
	{ 0x0000000002000000ULL, b4 },
	{ 0x0000000004000000ULL, c4 },
	{ 0x0000000008000000ULL, d4 },
	{ 0x0000000010000000ULL, e4 },
	{ 0x0000000020000000ULL, f4 },
	{ 0x0000000040000000ULL, g4 },
	{ 0x0000000080000000ULL, h4 },
	{ 0x0000000100000000ULL, a5 },
	{ 0x0000000200000000ULL, b5 },
	{ 0x0000000400000000ULL, c5 },
	{ 0x0000000800000000ULL, d5 },
	{ 0x0000001000000000ULL, e5 },
	{ 0x0000002000000000ULL, f5 },
	{ 0x0000004000000000ULL, g5 },
	{ 0x0000008000000000ULL, h5 },
	{ 0x0000010000000000ULL, a6 },
	{ 0x0000020000000000ULL, b6 },
	{ 0x0000040000000000ULL, c6 },
	{ 0x0000080000000000ULL, d6 },
	{ 0x0000100000000000ULL, e6 },
	{ 0x0000200000000000ULL, f6 },
	{ 0x0000400000000000ULL, g6 },
	{ 0x0000800000000000ULL, h6 },
	{ 0x0001000000000000ULL, a7 },
	{ 0x0002000000000000ULL, b7 },
	{ 0x0004000000000000ULL, c7 },
	{ 0x0008000000000000ULL, d7 },
	{ 0x0010000000000000ULL, e7 },
	{ 0x0020000000000000ULL, f7 },
	{ 0x0040000000000000ULL, g7 },
	{ 0x0080000000000000ULL, h7 },
	{ 0x0100000000000000ULL, a8 },
	{ 0x0200000000000000ULL, b8 },
	{ 0x0400000000000000ULL, c8 },
	{ 0x0800000000000000ULL, d8 },
	{ 0x1000000000000000ULL, e8 },
	{ 0x2000000000000000ULL, f8 },
	{ 0x4000000000000000ULL, g8 },
	{ 0x8000000000000000ULL, h8 } };

#endif