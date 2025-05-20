#include "move.h"

Square GetFrom(Move move) {
	const __int16_t sqr = from_mask & move;
	return Square(sqr);
}
Square GetTo(Move move) {
	return Square((to_mask & move) >> to_bits);
}
SpecialMove SpecialMoveType(Move move) {
	return SpecialMove((flag_mask & move) >> flag_bits);
}
Piece PromotionPiece(Move move) {
	return Piece((prom_mask & move) >> prom_bits);
}

Move CreateMove(Square from, Square to) {
	return from + (to << 6);
}
Move CreatePromotion(Square from, Square to, Piece promo) {
	return CreateMove(from, to) + (PROMOTE << flag_bits) + (promo << prom_bits);
}
Move CreateEnPassant(Square from, Square to) {
	return CreateMove(from, to) + (ENPASSANT << flag_bits);
}

// From and to will be the king, this will give information
// about kingside/queenside and which king is castling.
Move CreateCastle(Square from, Square to) {
	return CreateMove(from, to) + (CASTLE << flag_bits);
}


Move ReflectMove(Move move) {
	auto s = GetFrom(move);
	s = Reflect(s);

	auto f = GetTo(move);
	f = Reflect(f);

	return s + (f << to_bits) + ((flag_mask + prom_mask) & move);
}
