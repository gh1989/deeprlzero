#include <numeric>

#include "geometry.h"
#include "move.h"
#include "move_generation.h"

/* (Pseudo-)Move generation */
TMoveContainer GenerateMoves(const State& state) {
	TMoveContainer moves;
	PawnMoves(state, moves);
	JumperMoves<KNIGHT>(state, moves);
	SliderMoves<BISHOP>(state, moves);
	SliderMoves<ROOK>(state, moves);
	JumperMoves<KING>(state, moves);
	KingCastling(state, moves);
}

void PawnMoves(
	const State& state, 
	TMoveContainer &moves)
{
	/* Pawn moves: without rotation */
	Bitboard square, push_once, push_twice, capt_diag;
	auto pawnBB = state.movePiece<PAWN>();
	const size_t id_ep = 12;
			
	const Piece promote_pieces[4] = { KNIGHT, BISHOP, ROOK, QUEEN };

	bool whitePieces 	 = !state.isBlackMove();
	int pawn_off_1    	 = whitePieces ?  8 :  -8;
	int pawn_off_2    	 = whitePieces ? 16 : -16;
	const int p_att_l 	 = whitePieces ?  7 :  -7;
	const int p_att_r  	 = whitePieces ?  9 :  -9;
	const auto cpattacks = whitePieces ? pawn_attacks_b : pawn_attacks;
	const int promRank   = whitePieces ? 6 : 1;
	const int dblPushRnk = whitePieces ? 1 : 6;

	auto fullOccupancy = state.blackOccupation() | state.whiteOccupation();

	for (auto sqbb : BitboardRange(pawnBB)) {

		auto sqr = Square(sqbb);
		int rank = sqr / 8;
		const bool promote_cond = rank == promRank;
		square = squares[sqr];
		// Pushes
		push_once = squares[sqr + pawn_off_1];
		if (!(push_once & fullOccupancy))
		{
			// Double push.
			if (rank == dblPushRnk)
			{
				push_twice = squares[sqr + pawn_off_2];
				if (!(push_twice & fullOccupancy))
					moves.emplace_back(CreateMove(sqr, Square(sqr + pawn_off_2)));
			}
			if (promote_cond)
				for (auto& prom : promote_pieces) 
					moves.emplace_back(CreatePromotion(sqr, Square(sqr + pawn_off_1), prom));
			else
				moves.emplace_back(CreateMove(sqr, Square(sqr + pawn_off_1)));
		}

		auto enemyOccupancy = state.enemyOccupation();
		auto enPassant = state.getEnPassant();
		if (cpattacks[sqr] & (enemyOccupancy | enPassant))
			for (int shft : {p_att_l, p_att_r})
			{
				// You cannot capture off the side of the board.
				uint8_t file = sqr % 8;
				if (file == 0 && shft == p_att_l)
					continue;
				if (file == 7 && shft == p_att_r)
					continue;
				if ((shft < -sqr) || (shft > 63 - sqr))
					throw;

				capt_diag = shft < 0 ? square >> -shft : square << shft;
				auto to = Square(sqr + shft);
				if (capt_diag & (enemyOccupancy | enPassant))
				{
					// Promotion
					if (promote_cond)
						for (auto& prom : promote_pieces)
							moves.emplace_back(CreatePromotion(sqr, to, prom));
					else
					{
						// Taking enpassant
						if (cpattacks[sqr] & enPassant)
							moves.emplace_back(CreateEnPassant(sqr, to));
						else
							moves.emplace_back(CreateMove(sqr, to));
					}
				}
			}
	}
}

void KingCastling(
	const State& state,
	TMoveContainer &moves)
{
	bool wPieces 	= !state.isBlackMove();
	auto rookBB     = state.movePiece<ROOK>();
	auto kingBB     = state.movePiece<KING>();
	auto kingStart  = wPieces ? e1 : e8;
	auto castle     = state.getCastleRights();
	
	auto kCastleEnd = wPieces ? g1 : g8;
	auto ksCastling = wPieces ? Castling::WK : Castling::BK;
	auto kRookStart = wPieces ? h1 : h8;
	if ((rookBB & squares[kRookStart]) && (castle & ksCastling ))
		moves.emplace_back(CreateMove(kingStart, kCastleEnd));

	auto qsCastling = wPieces ? Castling::WQ : Castling::BQ;
	auto qCastleEnd = wPieces ? c1 : c8;
	auto qRookStart = wPieces ? a1 : a8;
	if ((rookBB & squares[qRookStart]) && (castle & qsCastling ))
	{
		moves.emplace_back(CreateMove(kingStart, qCastleEnd));
	}
}

template <Piece _Piece>
void JumperMoves(
    const State& s,
    TMoveContainer &moves )
{
	auto attacks = _Piece == KNIGHT ? knight_attacks : neighbours;
	auto pieceBB = s.movePiece<_Piece>();
	auto moveOccupation = s.moveOccupation();

    for (const auto& sqbb : BitboardRange(pieceBB))
    {
        Square sqr = Square(sqbb);
        Bitboard square = squares[sqr];
        const auto& attack = attacks[sqr] & (~moveOccupation);
		for(auto knightJumpSqIdx : BitboardRange(attack))
			moves.emplace_back(CreateMove(sqr, Square(knightJumpSqIdx)));
    }
}

template <Piece _Piece>
inline void SliderMoves(
    const State& state,
    TMoveContainer &moves)
{
	auto directions = _Piece == BISHOP ? bishop_directions : rook_directions;
	auto enemyOccupancy = state.enemyOccupation();
	auto sliderPieces = state.movePiece<_Piece>();
	auto moveOccupancy = state.moveOccupation();
	
    for (auto sliderSquare : BitboardRange(sliderPieces))
    {
        Bitboard squareBB = squares[sliderSquare];
        for (const auto& dir : directions)
        {
            int dx = dir.first;
            int dy = dir.second;
            int sidx = sliderSquare;
            int rank = sidx / 8;
			while( true )
			{
				sidx += 8 * dx + dy;
				rank += dx;

				if ((dx > 0 && rank > 7) || (dx < 0 && rank < 0) || (dy > 0 && sidx % 8 > 7) || (dy < 0 && sidx % 8 < 0))
					break;
                if (sidx < 0 || sidx > 63)
                    break;

                Bitboard newSquareBB = squares[sidx];

				// Hit obstacle - break
                if (newSquareBB & moveOccupancy)
                    break;

                moves.emplace_back(CreateMove(Square(sliderSquare), Square(sidx)));

				// Hit enemy last - break
                if (newSquareBB & enemyOccupancy)
                    break;
            }
        }
    }
}

bool isCheck(const State& state, bool blackIsAttacking)
{
	Bitboard colourKing = blackIsAttacking ? state.whitePiece<KING>() : state.blackPiece<KING>();
	return isBitboardAttacked(colourKing, state, blackIsAttacking);
}

bool isBitboardAttacked(const Bitboard& bitboard, const State& state, bool blackIsAttacking)
{
	auto oppKnights = blackIsAttacking ? state.blackPiece<KNIGHT>() : state.whitePiece<KNIGHT>();
	auto oppQueens = blackIsAttacking ? state.blackPiece<QUEEN>() : state.whitePiece<QUEEN>();
	auto oppBishops = blackIsAttacking ? state.blackPiece<BISHOP>() : state.whitePiece<BISHOP>();
	auto occupancy = state.blackOccupation() | state.whiteOccupation();
	auto diagonals = oppQueens | oppBishops;
	auto oppRooks = blackIsAttacking ? state.blackPiece<ROOK>() : state.whitePiece<ROOK>();
	auto straights = oppQueens | oppRooks;
	auto movePawnAttackMask = blackIsAttacking ? pawn_attacks : pawn_attacks_b;
	auto enemyPawns = blackIsAttacking ? state.blackPiece<PAWN>() : state.whitePiece<PAWN>();

	for(auto itSqr : BitboardRange(bitboard))
	{
		Square cSqr = static_cast<Square>(itSqr);
		if ((knight_attacks[cSqr] & oppKnights).bit_number)
			return true;

		// If our king "attacks" the enemy pawn like a pawn it's in check.
		if ((movePawnAttackMask[cSqr] & enemyPawns).bit_number)
			return true;

		if (SquareConnectedToBitboard(cSqr, diagonals, occupancy&~diagonals, bishop_directions))
			return true;

		if (SquareConnectedToBitboard(cSqr, straights, occupancy&~straights, rook_directions))
			return true;
	}
	
	return false;
}

bool checkLegal(const State& state, Move move)
{
	auto moveKing = state.movePiece<KING>();
	auto oppBishops = state.enemyPiece<QUEEN>() | state.enemyPiece<BISHOP>();
	auto oppRooks = state.enemyPiece<QUEEN>() | state.enemyPiece<ROOK>();
	auto fromSq = GetFrom(move);
	auto toSq = GetTo(move);
	bool checkCheck = false;

	// Don't move the king into check
	if(fromSq == square_lookup.at(moveKing))
	{
		if( neighbours[moveKing] & state.enemyPiece<KING>() )
			return false;
		checkCheck = true;
	}

    // Don't castle through check
	if( SpecialMoveType(move) == SpecialMove::CASTLE )
	{
		// Check if we are already in check. So send current isBlackMove
		if (isCheck(state, !state.isBlackMove()))
			return false;

		Bitboard kingsPath(0);
		if( toSq == c8 )
			kingsPath = GetSquare(d8, c8);
		if( toSq == c1 )
			kingsPath = GetSquare(d1, c1);
		if( toSq == g8 )
			kingsPath = GetSquare(f8, g8);
		if( toSq == g1 )
			kingsPath = GetSquare(f1, g1);

		// Check if path is attacked. 
		return !isBitboardAttacked(kingsPath, state, !state.isBlackMove());
	}

	// Bad algorithm... prevent pinned pieces from moving
    if (!checkCheck)
    {
		auto diag = diagonals[fromSq];
        if ((moveKing & diag) && (oppBishops & diag))
            checkCheck = bool(diagonals[fromSq] & ~diagonals[toSq]);
    }
    if (!checkCheck) 
    {
        auto adiag = antidiagonals[fromSq];
        if ((moveKing & adiag) && (oppBishops & adiag))
            checkCheck = bool(antidiagonals[fromSq] & ~antidiagonals[toSq]);
    }
    if (!checkCheck)
    {
        auto rank = ranks[fromSq / 8];
        if ((moveKing & rank) && (oppRooks & rank))
            checkCheck = bool(ranks[fromSq/8] & ~ranks[toSq/8]);
    }
    if (!checkCheck)
    {
        auto file = files[fromSq % 8];
        if ((moveKing & file) && (oppRooks & file))
            checkCheck = bool(files[fromSq%8] & ~files[toSq%8]);
    }

	// Here's the bad part.
	if (checkCheck)
	{
		State state_copy(state);
		state_copy.Apply(move);

		// Need to flip the isBlackMove here because we are testing if we 
		// were to make the move would it be check with the previous player
		// still attacking.
		return isCheck(state_copy, !state_copy.isBlackMove());
	}

	return true;
}