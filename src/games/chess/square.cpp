#include "square.h"

Bitboard GetSquare(Square sqr) { return squares[sqr]; }

// Check if a square is on 
bool IsOn(Bitboard bb, Square square) {
	Bitboard s = SqrBb(square);
	return bool(bb & s);
}
// Turn a square off
Bitboard OffBit(Bitboard bb, Square off) {
	Bitboard s = SqrBb(off);
	return bb & (~s);
}
// Turn a square on
Bitboard OnBit(Bitboard bb, Square on) {
	Bitboard s = SqrBb(on);
	return bb | s;
}
// Convert bitboard to square
Square BbSqr(Bitboard bb) {
	return static_cast<Square>(bb.nMSB());
	return square_lookup.at(bb);
}
// Convert square to bitboard
Bitboard SqrBb(Square sqr) {
	return squares[sqr];
}

Bitboard BitMove(Bitboard bb, Square from, Square to) {
	return OnBit(OffBit(bb, from), to);
}

Square Reflect(Square sqr)
{
	int square_int = static_cast<int>(sqr);
	square_int = (square_int % 8) + (7 - (square_int / 8)) * 8;
	Square output = static_cast<Square>(square_int);
	return output;
}