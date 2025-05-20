#include "state.h"
#include <functional>

template<class Iter, class T>
Iter binary_find(Iter begin, Iter end, T val) {
	// Finds the lower bound in at most log(last - first) + 1 comparisons
	Iter i = std::lower_bound(begin, end, val);

	if (i != end && !(val < *i))
		return i; // found
	else
		return end; // not found
}

void State::Apply(Move move) {
	auto specl = SpecialMoveType(move);
	bool new50 = false;
	bool other = !blackMove;
	auto start = GetFrom(move);
	auto finish = GetTo(move);

	const uint8_t ip = NUMBER_PIECES * blackMove + PAWN;
	const uint8_t ip2 = NUMBER_PIECES * other + PAWN;
	const uint8_t ik = NUMBER_PIECES * blackMove + KING;
	const uint8_t rk = NUMBER_PIECES * blackMove + ROOK;

	/* Removing captured pieces of opposition */
	uint8_t i_other;
	int p_other;
	Bitboard pbb_other;
	for (p_other = 0; p_other < NUMBER_PIECES; p_other++) {
		i_other = NUMBER_PIECES * other + p_other;
		pbb_other = bbs[i_other];
		if (IsOn(pbb_other, finish))
		{
			bbs[i_other] = OffBit(pbb_other, finish);
			new50 = true;
			break;
		}
	}

	/* Update en passant state, disable castling */
	int i, p;
	Bitboard pbb;
	for (p = 0; p < NUMBER_PIECES; p++) {
		i = NUMBER_PIECES * blackMove + p;
		pbb = bbs[i];
		if (IsOn(pbb, start)) {
			new50 = (p == PAWN);
			if (p == PAWN && abs(finish - start) > 15) {
				bbs[12] = Bitboard(1ULL << (blackMove ? (finish - 8) : (start - 8)));
			}
			else { bbs[12] = 0; }

			if (p == KING) {
				castle = castle & ~(blackMove ? (BQ + BK) : (WQ + WK));
			}
			else if (p == ROOK) {
				if (start == a1 && !blackMove) castle &= ~WQ;
				else if (start == h1 && !blackMove) castle &= ~WK;
				else if (start == a8 && blackMove) castle &= ~BQ;
				else if (start == h8 && blackMove) castle &= ~BK;
			}
			bbs[i] = OffBit(bbs[i], start);
			break;
		}
	}

	/* Not special */
	if (specl == NONE) {
		bbs[i] = OnBit(bbs[i], finish);
	}
	/* En passant capture */
	if (specl == ENPASSANT) {
		auto enPassant = bbs[12];
		new50 = true;
		auto passantp = blackMove ? (enPassant>> 8) : (enPassant<< 8);
		bbs[ip] = bbs[ip] | bbs[12];
		bbs[ip2] = bbs[ip2] & ~passantp;
	}
	/* Promotion */
	if (specl == PROMOTE) {
		new50 = true;
		auto prom = PromotionPiece(move);
		const uint8_t i_prom = NUMBER_PIECES * blackMove + prom;
		auto our_promote_pieces =
			bbs[i_prom] = OnBit(bbs[i_prom], finish);
	}
	/* Castling */
	if (specl == CASTLE) {
		bbs[ik] = OnBit(bbs[ik], finish);
		bool qs = (finish % 8) < 4;
		auto s1 = Square(qs ? finish - 2 : finish + 1);
		auto s2 = Square(qs ? finish + 1 : finish - 1);
		bbs[rk] = BitMove(bbs[rk], s1, s2);
		castle = castle & ~(blackMove ? (BQ + BK) : (WQ + WK));
	}
	c50 = new50 ? 0 : c50 + 1;
	blackMove = other;
	plies++;
}
