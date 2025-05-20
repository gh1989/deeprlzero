#include "chess.h"
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

#include "chess/move_generation.h"
#include "chess/string_transforms.h"

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
    std::vector<Move> legal_moves = GenerateLegalMoves(state_);
    
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
    std::vector<Move> legal_moves = GenerateLegalMoves(state_);
    
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
            Square sq = i * 8 + j;
            
            // Current player's pieces (first 6 planes)
            if (!state_.isBlackMove()) {
                // White pieces when white to move
                if (state_.whitePiece<Piece::Pawn>() & squares[sq]) accessor[0][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Knight>() & squares[sq]) accessor[1][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Bishop>() & squares[sq]) accessor[2][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Rook>() & squares[sq]) accessor[3][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Queen>() & squares[sq]) accessor[4][i][j] = 1.0f;
                if (state_.whitePiece<Piece::King>() & squares[sq]) accessor[5][i][j] = 1.0f;
            } else {
                // Black pieces when black to move
                if (state_.blackPiece<Piece::Pawn>() & squares[sq]) accessor[0][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Knight>() & squares[sq]) accessor[1][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Bishop>() & squares[sq]) accessor[2][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Rook>() & squares[sq]) accessor[3][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Queen>() & squares[sq]) accessor[4][i][j] = 1.0f;
                if (state_.blackPiece<Piece::King>() & squares[sq]) accessor[5][i][j] = 1.0f;
            }
            
            // Opponent's pieces (next 6 planes)
            if (!state_.isBlackMove()) {
                // Black pieces when white to move
                if (state_.blackPiece<Piece::Pawn>() & squares[sq]) accessor[6][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Knight>() & squares[sq]) accessor[7][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Bishop>() & squares[sq]) accessor[8][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Rook>() & squares[sq]) accessor[9][i][j] = 1.0f;
                if (state_.blackPiece<Piece::Queen>() & squares[sq]) accessor[10][i][j] = 1.0f;
                if (state_.blackPiece<Piece::King>() & squares[sq]) accessor[11][i][j] = 1.0f;
            } else {
                // White pieces when black to move
                if (state_.whitePiece<Piece::Pawn>() & squares[sq]) accessor[6][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Knight>() & squares[sq]) accessor[7][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Bishop>() & squares[sq]) accessor[8][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Rook>() & squares[sq]) accessor[9][i][j] = 1.0f;
                if (state_.whitePiece<Piece::Queen>() & squares[sq]) accessor[10][i][j] = 1.0f;
                if (state_.whitePiece<Piece::King>() & squares[sq]) accessor[11][i][j] = 1.0f;
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

std::string Chess::GetBoardString() const {
    return StateToFen(state_);
}

void Chess::SetFromString(const std::string& fen_str, int player) {
    state_ = FenToState(fen_str);
    
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

// Private helper methods

int Chess::MoveToIndex(const Move& move) const {
    // Convert a chess Move to an action index
    // Format: from_square * 64 + to_square, with promotion pieces encoded as additions
    int from_square = move.getFrom();
    int to_square = move.getTo();
    int promotion_piece = move.getPromoPiece();
    
    int base_index = from_square * 64 + to_square;
    
    // Add promotion encoding if needed
    if (promotion_piece != Piece::Empty) {
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
    
    Piece promotion_piece = Piece::Empty;
    if (promotion_encoding > 0) {
        promotion_piece = static_cast<Piece>(promotion_encoding + 1);
    }
    
    return Move(from_square, to_square, promotion_piece);
}

bool Chess::IsCheckmate() const {
    if (!IsInCheck(state_, state_.isBlackMove() ? Color::Black : Color::White)) {
        return false;
    }
    
    return GenerateLegalMoves(state_).empty();
}

bool Chess::IsStalemate() const {
    if (IsInCheck(state_, state_.isBlackMove() ? Color::Black : Color::White)) {
        return false;
    }
    
    return GenerateLegalMoves(state_).empty();
}

bool Chess::IsDraw() const {
    // 50-move rule
    if (state_.get50MoveCount() >= 100) {
        return true;
    }
    
    // Insufficient material
    if (IsInsufficientMaterial(state_)) {
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
    } else if (GenerateLegalMoves(state_).empty()) {
        game_over_ = true;
        result_ = 0; // Stalemate
    }
}

}  // namespace deeprlzero