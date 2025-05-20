#ifndef GAMES_CHESS_H
#define GAMES_CHESS_H

#include <vector>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "concepts.h"
#include "traits.h"
#include "chess/state.h"
#include "chess/move.h"
#include "chess/geometry.h"
#include "chess/square.h"


namespace deeprlzero {

class Chess {
public:
    using Traits = GameTraits<Chess>;
    
    Chess();
    std::vector<int> GetValidMoves() const;
    void MakeMove(int action_index);
    float GetGameResult() const;
    bool IsTerminal() const;
    bool CheckWin(int player) const;
    torch::Tensor GetCanonicalBoard() const;
    Chess Clone() const;
    void UndoMove(int action_index);
    void Reset();
    int GetCurrentPlayer() const { return state_.isBlackMove() ? -1 : 1; }
    
    // Methods from traits
    int GetActionSize() const { return Traits::kNumActions; }
    int GetInputChannels() const { return Traits::kNumChannels; }
    int GetNumActions() const { return Traits::kNumActions; }
    
    // Display methods
    std::string ToString() const;
    std::string GetBoardString() const;
    void SetFromString(const std::string& fen_str, int player);

private:
    State state_;
    std::vector<Move> move_history_;
    bool game_over_ = false;
    int result_ = 0; // 0 for draw, 1 for white win, -1 for black win
    
    // Helper methods
    int MoveToIndex(const Move& move) const;
    Move IndexToMove(int action_index) const;
    bool IsCheckmate() const;
    bool IsStalemate() const;
    bool IsDraw() const;
    void UpdateGameStatus();
};

static_assert(GameConcept<Chess>);

}  // namespace deeprlzero

#endif