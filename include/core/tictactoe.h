#ifndef ALPHAZERO_TICTACTOE_H_
#define ALPHAZERO_TICTACTOE_H_

#include "core/game.h"
#include <array>

namespace alphazero {

class TicTacToe : public Game {
 public:
  static constexpr int kBoardSize = 3;
  static constexpr int kNumActions = 9;
  static constexpr int kNumChannels = 1;
  
  TicTacToe();
  
  virtual std::vector<int> GetValidMoves() const override;
  void MakeMove(int move) override;
  float GetGameResult() const override;
  bool IsTerminal() const override;
  int GetCurrentPlayer() const override { return current_player_; }
  torch::Tensor GetCanonicalBoard() const override;
  std::unique_ptr<Game> Clone() const override;
  int GetActionSize() const override { return kNumActions; }
  int GetInputChannels() const override { return kNumChannels; }

 private:
  std::array<std::array<int, kBoardSize>, kBoardSize> board_;
  int current_player_;
  
  bool CheckWin(int player) const;
  bool IsBoardFull() const;
  
};

}  // namespace alphazero

#endif  // ALPHAZERO_TICTACTOE_H_ 