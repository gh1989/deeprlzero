#include "chess.h"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

namespace deeprlzero {

Chess::Chess() {
    Reset();
}

void Chess::Reset() {
    // Initialize to standard chess starting position
    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    SetFromString(starting_fen, 1);
    game_over_ = false;
    result_ = 0;
    move_history_.clear();
}

std::vector<int> Chess::GetValidMoves() const {
    if (game_over_) {
        return {};
    }
    
    std::vector<int> valid_action_indices;
    TMoveContainer legal_moves = GenerateMoves(state_);
    
    for (const auto& move : legal_moves) {
        valid_action_indices.push_back(MoveToIndex(move));
    }
    
    return valid_action_indices;
}

void Chess::MakeMove(int action_index) {
    if (game_over_) {
        throw std::runtime_error("Game is already over");
    }
    
    Move move = IndexToMove(action_index);
    TMoveContainer legal_moves = GenerateMoves(state_);
    
    // Verify move is legal
    bool move_is_legal = false;
    for (const auto& legal_move : legal_moves) {
        if (move == legal_move) {
            move_is_legal = true;
            break;
        }
    }
    
    if (!move_is_legal) {
        throw std::invalid_argument("Illegal move");
    }
    
    // Store move for potential undo
    move_history_.push_back(move);
    
    // Apply move to state
    state_.Apply(move);
    
    // Update game status
    UpdateGameStatus();
}

void Chess::UndoMove(int) {
    if (move_history_.empty()) {
        throw std::runtime_error("No moves to undo");
    }
    
    // Chess doesn't easily support undoing moves with the current engine
    // So we'll reset and replay all moves except the last one
    Move last_move = move_history_.back();
    move_history_.pop_back();
    
    // Reset the board
    Reset();
    
    // Replay all moves except the last one
    for (const auto& move : move_history_) {
        state_.Apply(move);
    }
    
    // Update game status
    UpdateGameStatus();
}

bool Chess::IsTerminal() const {
    return game_over_;
}

float Chess::GetGameResult() const {
    if (!game_over_) {
        return 0.0f; // Game not finished
    }
    
    return static_cast<float>(result_);
}

bool Chess::CheckWin(int player) const {
    if (!game_over_) {
        return false;
    }
    
    return (player == 1 && result_ == 1) || (player == -1 && result_ == -1);
}

torch::Tensor Chess::GetCanonicalBoard() const {
    // Create a tensor with 12 planes for pieces (6 piece types × 2 colors) + 1 plane for player to move
    auto tensor = torch::zeros({Traits::kNumChannels, Traits::kBoardSize, Traits::kBoardSize});
    auto accessor = tensor.accessor<float, 3>();
    
    // Fill piece planes
    for (int i = 0; i < Traits::kBoardSize; ++i) {
        for (int j = 0; j < Traits::kBoardSize; ++j) {
            auto sq = i * 8 + j;
            
            // Current player's pieces (first 6 planes)
            if (!state_.isBlackMove()) {
                // White pieces when white to move  
                if (state_.whitePiece<PAWN>() & squares[sq]) accessor[0][i][j] = 1.0f;
                if (state_.whitePiece<KNIGHT>() & squares[sq]) accessor[1][i][j] = 1.0f;
                if (state_.whitePiece<BISHOP>() & squares[sq]) accessor[2][i][j] = 1.0f;
                if (state_.whitePiece<ROOK>() & squares[sq]) accessor[3][i][j] = 1.0f;
                if (state_.whitePiece<QUEEN>() & squares[sq]) accessor[4][i][j] = 1.0f;
                if (state_.whitePiece<KING>() & squares[sq]) accessor[5][i][j] = 1.0f;
            } else {
                // Black pieces when black to move
                if (state_.blackPiece<PAWN>() & squares[sq]) accessor[0][i][j] = 1.0f;
                if (state_.blackPiece<KNIGHT>() & squares[sq]) accessor[1][i][j] = 1.0f;
                if (state_.blackPiece<BISHOP>() & squares[sq]) accessor[2][i][j] = 1.0f;
                if (state_.blackPiece<ROOK>() & squares[sq]) accessor[3][i][j] = 1.0f;
                if (state_.blackPiece<QUEEN>() & squares[sq]) accessor[4][i][j] = 1.0f;
                if (state_.blackPiece<KING>() & squares[sq]) accessor[5][i][j] = 1.0f;
            }
            
            // Opponent's pieces (next 6 planes)
            if (!state_.isBlackMove()) {
                // Black pieces when white to move
                if (state_.blackPiece<PAWN>() & squares[sq]) accessor[6][i][j] = 1.0f;
                if (state_.blackPiece<KNIGHT>() & squares[sq]) accessor[7][i][j] = 1.0f;
                if (state_.blackPiece<BISHOP>() & squares[sq]) accessor[8][i][j] = 1.0f;
                if (state_.blackPiece<ROOK>() & squares[sq]) accessor[9][i][j] = 1.0f;
                if (state_.blackPiece<QUEEN>() & squares[sq]) accessor[10][i][j] = 1.0f;
                if (state_.blackPiece<KING>() & squares[sq]) accessor[11][i][j] = 1.0f;
            } else {
                // White pieces when black to move
                if (state_.whitePiece<PAWN>() & squares[sq]) accessor[6][i][j] = 1.0f;
                if (state_.whitePiece<KNIGHT>() & squares[sq]) accessor[7][i][j] = 1.0f;
                if (state_.whitePiece<BISHOP>() & squares[sq]) accessor[8][i][j] = 1.0f;
                if (state_.whitePiece<ROOK>() & squares[sq]) accessor[9][i][j] = 1.0f;
                if (state_.whitePiece<QUEEN>() & squares[sq]) accessor[10][i][j] = 1.0f;
                if (state_.whitePiece<KING>() & squares[sq]) accessor[11][i][j] = 1.0f;
            }
        }
    }
    
    // Additional features: Turn indicator
    if (!state_.isBlackMove()) {
        // White to move
        for (int i = 0; i < Traits::kBoardSize; ++i) {
            for (int j = 0; j < Traits::kBoardSize; ++j) {
                accessor[12][i][j] = 1.0f;
            }
        }
    }
    
    return tensor;
}

Chess Chess::Clone() const {
    Chess clone;
    clone.state_ = state_;
    clone.game_over_ = game_over_;
    clone.result_ = result_;
    clone.move_history_ = move_history_;
    return clone;
}

std::string Chess::ToString() const {
    return GetBoardString() + (state_.isBlackMove() ? " (Black to move)" : " (White to move)");
}

std::string StateToFen(const State& state) {
    return "";
}

std::string Chess::GetBoardString() const {
    return StateToFen(state_);
}

void Chess::SetFromString(const std::string& fen_str, int player) {
    state_ = StateFromFen(fen_str);
    
    // Ensure the player to move matches the requested player
    if ((player == 1 && state_.isBlackMove()) || (player == -1 && !state_.isBlackMove())) {
        state_.setBlackMove(!state_.isBlackMove());
    }
    
    game_over_ = false;
    result_ = 0;
    move_history_.clear();
    
    // Update game status
    UpdateGameStatus();
}

// Move template functions and helper functions before GenerateMoves
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
    auto moveOccupation = state.moveOccupation();
    
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

                if (newSquareBB & moveOccupation)
                    break;

                moves.emplace_back(CreateMove(Square(sliderSquare), Square(sidx)));

                if (newSquareBB & enemyOccupancy)
                    break;
            }
        }
    }
}

bool isBitboardAttacked(const Bitboard& bitboard, const State& state, bool blackIsAttacking);

/* (Pseudo-)Move generation */
TMoveContainer GenerateMoves(const State& state) {
    TMoveContainer moves;
    PawnMoves(state, moves);
    JumperMoves<KNIGHT>(state, moves);
    SliderMoves<BISHOP>(state, moves);
    SliderMoves<ROOK>(state, moves);
    JumperMoves<KING>(state, moves);
    KingCastling(state, moves);
    return moves;
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

int Chess::MoveToIndex(const Move& move) const {
    // Convert a chess Move to an action index
    // Format: from_square * 64 + to_square, with promotion pieces encoded as additions
    int from_square = GetFrom(move);
    int to_square = GetTo(move);
    int promotion_piece = PromotionPiece(move);
    
    int base_index = from_square * 64 + to_square;
    
    // Add promotion encoding if needed
    if (promotion_piece != Piece::NO_PIECE) {
        base_index += (promotion_piece - 1) * (64 * 64);
    }
    
    return base_index;
}

Move Chess::IndexToMove(int action_index) const {
    // Convert an action index back to a chess Move
    int promotion_encoding = action_index / (64 * 64);
    int base_move = action_index % (64 * 64);
    
    int from_square = base_move / 64;
    int to_square = base_move % 64;
    
    Piece promotion_piece = Piece::NO_PIECE;
    if (promotion_encoding > 0) {
        promotion_piece = static_cast<Piece>(promotion_encoding + 1);
        return CreatePromotion(Square(from_square), Square(to_square), promotion_piece);
    }
    
    return CreateMove(Square(from_square), Square(to_square));
}

bool Chess::IsCheckmate() const {
    if (!isCheck(state_, state_.isBlackMove())) {
        return false;
    }
    
    return GenerateMoves(state_).empty();
}

bool Chess::IsStalemate() const {
    if (isCheck(state_, state_.isBlackMove())) {
        return false;
    }
    
    return GenerateMoves(state_).empty();
}

bool Chess::IsDraw() const {
    // 50-move rule
    if (state_.get50MoveCount() >= 100) {
        return true;
    }
    
    // Insufficient material
    if (isInsufficientMaterial(state_)) {
        return true;
    }
    
    // Stalemate
    return IsStalemate();
}

void Chess::UpdateGameStatus() {
    if (IsCheckmate()) {
        game_over_ = true;
        result_ = state_.isBlackMove() ? 1 : -1; // If black is in checkmate, white wins
    } else if (IsDraw()) {
        game_over_ = true;
        result_ = 0;
    } else if (GenerateMoves(state_).empty()) {
        game_over_ = true;
        result_ = 0; // Stalemate
    }
}


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

void PrettyPrint(const State& state)
{
	// collect information
	std::string pos[8][8];
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			pos[i][j] = ' ';

	const std::string PIECE_STRINGS = "NBRQKP";
	const std::string piece_strings = "nbrqkp";

	for (int i = 0; i < NUMBER_PIECES; ++i)
	{
		// requires knowledge of implementation in state. Bad
		Bitboard wocc = state.getBitboard(i);
		Bitboard bocc = state.getBitboard(i+NUMBER_PIECES); 
		Piece piece = static_cast<Piece>(i);
		for (int i = 0; i < 64; i++)
		{
			int j = (7 - int(i / 8)) % 8;
			int k = i % 8;
			if (wocc & squares[i])
				pos[j][k] = PIECE_STRINGS[piece];
			if (bocc & squares[i])
				pos[j][k] = piece_strings[piece];
		}
	}

	// print out the board
	std::string baseline = "+---";
	for (auto j = 0; j < 7; j++)
		baseline += "+---";
	baseline += "+\n";

	std::string output = baseline;
	for (auto i = 0; i < 8; i++)
	{
		for (auto j = 0; j < 8; j++)
			output += "| " + pos[i][j] + " ";
		output += "|\n";
		output += baseline;
	}

	std::cout << output;
	Bitboard ep = state.getBitboard(12);

	if (ep)
	{
		std::cout << "en-passant: ";
		for (int i = 0; i < 63; i++)
		{
			if (squares[i] & ep)
				std::cout << SquareName(Square(i));
		}
		std::cout << std::endl;
	}
	std::cout << "fiftycounter: " << state.getMoveCount() << std::endl;
	int castlerights = state.getCastleRights();
	const std::string crights = "QKqk";
	std::cout << "castlerights: " << castlerights << " ";
	for (char c : crights)
	{
		if (castlerights % 2)
			std::cout << c;
		castlerights /= 2;
	}

	std::cout << std::endl;
	std::cout << "plies: " << state.getPlies() << std::endl;
	std::cout << "colour to move: " << (!state.isBlackMove() ? "white" : "black") << std::endl;
}

// Get square name as string
std::string SquareName(Square sqr)
{
	const std::string file_strings[8] = { "a", "b", "c", "d", "e", "f", "g", "h" };
	const std::string rank_strings[8] = { "1", "2", "3", "4", "5", "6", "7", "8" };
	int square_int = static_cast<int>(sqr);
	return file_strings[square_int % 8] + rank_strings[int(square_int / 8)];
}
std::string PieceStringLower(Piece piece) { return piece_strings[piece]; }

Bitboard BitboardFromString(std::string str)
{
	if (str[0] < 'a' || str[1] < '1' || str[0] > 'h' || str[1] > '8')
		throw std::runtime_error("Square string is formatted improperly.");
	uint64_t boardnum = str[0] - 'a' + 8 * (str[1] - '1');
	return Bitboard(1ULL << boardnum);
}

/* UCI and string operations */
std::string AsUci(Move move) {
	std::stringstream ss;
	ss << SquareName(GetFrom(move));
	ss << SquareName(GetTo(move));
	if (SpecialMoveType(move) == PROMOTE)
		ss << PieceStringLower(PromotionPiece(move));
	return  ss.str();
}

Bitboard get_en_passant(const std::string& fen) {
	std::size_t pos = fen.find(" ");
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
    if (pos == std::string::npos) {
        return 0;
    }
    std::string ep_square = fen.substr(pos+1, 2);
    if (ep_square == "- ") {
        return 0;
    }
    uint64_t enPassantIdx = ep_square[0] - 'a' + 8 * (ep_square[1] - '1');
	return Bitboard(1ULL << enPassantIdx);
}

int get_fifty_move_count(const std::string& fen) {
    std::size_t pos = fen.find(" ");
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
    pos = fen.find(" ", pos+1);
	std::size_t end = fen.find(" ", pos+1);
    if (pos == std::string::npos) {
        return 0;
    }
    return std::stoi(fen.substr(pos+1, end));
}

bool get_turn(const std::string& fen) {
    std::size_t pos = fen.find(" ");
    return (fen.substr(pos+1, 1) != "w");
}

State StateFromFen(std::string fen) 
{
	State state;
	// Find the position of the board part of the FEN string
	std::size_t pos_end = fen.find(" ");
	std::size_t rank = 7;
	std::size_t file = 0;
	// Go through the board part of the FEN string
	for (int i = 0; i < pos_end; i++) {
		char c = fen[i];
		if (c >= '1' && c <= '8') {
			// Empty squares
			file += (c - '0');
		} else if (c == '/') {
			// Skip to the next rank
			rank--;
			file = 0;
			continue;
		} else {
			// Piece squares
			Bitboard square = (1ULL << (8*rank+file));
			file++;

			bool _whitePiece = !std::islower(c);
			auto C = std::toupper(c);
			std::unordered_map<char, Piece> pieceLookup = {
    			{'P', PAWN},
    			{'N', KNIGHT},
    			{'B', BISHOP},
    			{'R', ROOK},
    			{'Q', QUEEN},
    			{'K', KING}
			};

			Piece _piece = pieceLookup[C]; 
			state.addToBitboard(!_whitePiece*NUMBER_PIECES+_piece, square );
		}
	}

	// Find the position of the castling part of the FEN string
	std::size_t castling_pos = fen.find_first_of(' ', pos_end + 1);

	// Check if there are any castling rights
	if (fen[castling_pos - 1] == '-') {
		// No castling rights
		state.setCastleRights(0);
	} else {
		auto castle = 0;
		// Parse the castling rights
		if (fen.find('K', castling_pos) != std::string::npos) {
			castle |= WK;
		}
		if (fen.find('Q', castling_pos) != std::string::npos) {
			castle |= WQ;
		}
		if (fen.find('k', castling_pos) != std::string::npos) {
			castle |= BK;
		}
		if (fen.find('q', castling_pos) != std::string::npos) {
			castle |= BQ;
		}
		state.setCastleRights(castle);
	}

	// Find the position of the ply part of the FEN string
	std::size_t move_pos = fen.find_last_of(' ');
	std::string move_str = fen.substr(move_pos + 1);
	
    auto plies = 2*(std::stoi(move_str)-1) + state.isBlackMove();
    auto enPassant = get_en_passant(fen);
	auto fiftyMove = get_fifty_move_count(fen);
	auto turn = get_turn(fen);

    state.setPlies(plies);
    state.setEnPassant(enPassant);
    state.setFiftyMove(fiftyMove);
    state.setTurn(turn);

	return state;
}

bool SquareConnectedToBitboard(
	Square source,
	Bitboard target,
	Bitboard obstacle,
	const std::array<std::pair<int, int>, 4>& directions)
{
	for (int i = 0; i < 4; i++)
	{
		std::pair<int, int> dir = directions[i];
		auto dx = dir.first;
		auto dy = dir.second;
		int sidx = source;
		int rank = sidx / 8;
		while (true)
		{
			if (dx > 0 && rank == 7)
				break;
			if (dx < 0 && rank == 0)
				break;
			int file = sidx % 8;
			if (dy > 0 && file == 7)
				break;
			if (dy < 0 && file == 0)
				break;

			sidx += 8 * dx + dy;
			rank += dx;

			if ((sidx > 63) | (sidx < 0))
				break;

			auto tmpbb = squares[sidx];
			if (tmpbb & obstacle)
				break;
			if (tmpbb & target)
				return true;
		}
	}
	return false;
}

}  // namespace deeprlzero