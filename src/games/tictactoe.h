#ifndef GAMES_TICTACTOE_H
#define GAMES_TICTACTOE_H

#include "game.h"

namespace deeprlzero {

class TicTacToe : public Game {
 public:
  static constexpr int kBoardSize = 3;
  static constexpr int kNumActions = 9;
  static constexpr int kNumChannels = 3;
  
  TicTacToe();
  
  virtual std::vector<int> GetValidMoves() const override;
  void MakeMove(int move) override;
  void Reset() override;
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

  std::string GetBoardString() const;
  void SetFromString(const std::string& boardStr, int player);

 private:
  std::array<std::array<int, kBoardSize>, kBoardSize> board_;
  int current_player_;
  
  bool CheckWin(int player) const;
  bool IsBoardFull() const;
  
};

}

#endif