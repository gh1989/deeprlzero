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
  void UndoMove(int move) override;

  // Add visualization method
  std::string ToString() const {
    std::string result;
    for (int i = 0; i < kBoardSize; ++i) {
      for (int j = 0; j < kBoardSize; ++j) {
        switch (board_[i][j]) {
          case 1: result += " X "; break;
          case -1: result += " O "; break;
          case 0: result += " . "; break;
        }
      }
      result += "\n";
    }
    return result;
  }

 private:
  std::array<std::array<int, kBoardSize>, kBoardSize> board_;
  int current_player_;
  
  bool CheckWin(int player) const;
  bool IsBoardFull() const;
  
};

}  // namespace alphazero

#endif  // ALPHAZERO_TICTACTOE_H_ 