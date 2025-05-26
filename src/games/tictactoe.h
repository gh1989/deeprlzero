#ifndef GAMES_TICTACTOE_H
#define GAMES_TICTACTOE_H

#include <vector>
#include <memory>
#include <torch/torch.h>

#include "concepts.h"
#include "traits.h"

namespace deeprlzero {

class TicTacToe {
public:
    using Traits = GameTraits<TicTacToe>;
    TicTacToe();
    std::vector<int> GetValidMoves() const;
    void MakeMove(int move);
    float GetGameResult() const;
    bool IsTerminal() const;
    bool CheckWin(int player) const;
    torch::Tensor GetCanonicalBoard() const;
    TicTacToe Clone() const;
    void UndoMove(int move);
    void Reset();
    int GetCurrentPlayer() const { return current_player_; }
    
    // static Methods from traits
    static int GetActionSize() { return Traits::kNumActions; }
    static int GetInputChannels() { return Traits::kNumChannels; }
    static int GetNumActions() { return Traits::kNumActions; }
    static int GetBoardSize() { return Traits::kBoardSize; }
    
    // Display methods
    std::string ToString() const;
    std::string GetBoardString() const;
    void SetFromString(const std::string& str, int player);

private:
    Traits::BoardType board_;
    int current_player_ = 1;
    bool IsBoardFull() const;
};

static_assert(GameConcept<TicTacToe>);

}
#endif