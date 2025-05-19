#include "tictactoe.h"

namespace deeprlzero {

TicTacToe::TicTacToe() {
  for (auto& row : board_) {
    row.fill(0);
  }
}

std::vector<int> TicTacToe::GetValidMoves() const {
  std::vector<int> valid_moves;
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    for (int j = 0; j < Traits::kBoardSize; ++j) {
      if (board_[i][j] == 0) {
        valid_moves.push_back(i * Traits::kBoardSize + j);
      }
    }
  }
  return valid_moves;
}

void TicTacToe::Reset() {
  for (auto& row : board_) {
    row.fill(0);
  }
  current_player_ = 1;
}

void TicTacToe::MakeMove(int move) {
  int row = move / Traits::kBoardSize;
  int col = move % Traits::kBoardSize;

  if (row < 0 || row >= Traits::kBoardSize || col < 0 || col >= Traits::kBoardSize ||
      board_[row][col] != 0) {
    throw std::invalid_argument("Invalid move");
  }

  board_[row][col] = current_player_;
  current_player_ = -current_player_;
}

bool TicTacToe::IsBoardFull() const {
  for (const auto& row : board_) {
    for (int cell : row) {
      if (cell == 0) return false;
    }
  }
  return true;
}

float TicTacToe::GetGameResult() const {
  if (CheckWin(1)) return 1.0f;
  if (CheckWin(-1)) return -1.0f;
  if (IsBoardFull()) return 0.0f;
  return 0.0f;  // Ongoing
}

bool TicTacToe::IsTerminal() const {
  return CheckWin(1) || CheckWin(-1) || IsBoardFull();
}

torch::Tensor TicTacToe::GetCanonicalBoard() const {
  auto tensor = torch::zeros({3, Traits::kBoardSize, Traits::kBoardSize});  // 3 channels!
  auto accessor = tensor.accessor<float, 3>();

  for (int i = 0; i < Traits::kBoardSize; ++i) {
    for (int j = 0; j < Traits::kBoardSize; ++j) {
      accessor[0][i][j] = (board_[i][j] == current_player_) ? 1.0f : 0.0f;
      accessor[1][i][j] = (board_[i][j] == -current_player_) ? 1.0f : 0.0f;
      accessor[2][i][j] = (current_player_ == 1) ? 1.0f : 0.0f;
    }
  }
  return tensor;
}

TicTacToe TicTacToe::Clone() const {
  TicTacToe clone;
  clone.board_ = board_;
  clone.current_player_ = current_player_;
  return clone;
}

bool TicTacToe::CheckWin(int player) const {
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    bool row_win = true;
    bool col_win = true;
    for (int j = 0; j < Traits::kBoardSize; ++j) {
      if (board_[i][j] != player) row_win = false;
      if (board_[j][i] != player) col_win = false;
    }
    if (row_win || col_win) return true;
  }

  bool diag1_win = true;
  bool diag2_win = true;
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    if (board_[i][i] != player) diag1_win = false;
    if (board_[i][Traits::kBoardSize - 1 - i] != player) diag2_win = false;
  }

  return diag1_win || diag2_win;
}

void TicTacToe::UndoMove(int move) {
  int row = move / Traits::kBoardSize;
  int col = move % Traits::kBoardSize;

  if (row < 0 || row >= Traits::kBoardSize || col < 0 || col >= Traits::kBoardSize) {
    throw std::invalid_argument("Invalid move to undo");
  }

  board_[row][col] = 0;                
  current_player_ = -current_player_;  
}

std::string TicTacToe::GetBoardString() const {
  std::string result;
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    for (int j = 0; j < Traits::kBoardSize; ++j) {
      if (board_[i][j] == 1) result += "1";
      else if (board_[i][j] == -1) result += "2"; 
      else result += "0";
    }
    //result += "/";
  }
  return result;
}

void TicTacToe::SetFromString(const std::string& boardStr, int player) {
  if (boardStr.size() != Traits::kBoardSize * Traits::  kBoardSize) {
    throw std::invalid_argument("Invalid board string length");
  }
  
  current_player_ = player;
  
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    for (int j = 0; j < Traits::kBoardSize; ++j) {
        char c = boardStr[i * Traits::kBoardSize + j];
      if (c == '1') board_[i][j] = 1;
      else if (c == '2') board_[i][j] = -1;  // 2 represents player -1
      else board_[i][j] = 0;
    }
  }
}

std::string TicTacToe::ToString() const {
  std::string result;
  for (int i = 0; i < Traits::kBoardSize; ++i) {
    for (int j = 0; j < Traits::kBoardSize; ++j) {
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

}
