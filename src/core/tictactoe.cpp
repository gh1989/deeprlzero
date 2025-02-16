#include "core/tictactoe.h"
#include <stdexcept>

namespace alphazero {

TicTacToe::TicTacToe() : current_player_(1) {
    for (auto& row : board_) {
        row.fill(0);
    }
}

std::vector<int> TicTacToe::GetValidMoves() const {
    std::vector<int> valid_moves;
    for (int i = 0; i < kBoardSize; ++i) {
        for (int j = 0; j < kBoardSize; ++j) {
            if (board_[i][j] == 0) {
                valid_moves.push_back(i * kBoardSize + j);
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
    int row = move / kBoardSize;
    int col = move % kBoardSize;
    
    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize || 
        board_[row][col] != 0) {
        throw std::invalid_argument("Invalid move");
    }
    
    board_[row][col] = current_player_;
    current_player_ = -current_player_;
}

float TicTacToe::GetGameResult() const {
    if (CheckWin(1)) return 1.0f;
    if (CheckWin(-1)) return -1.0f;
    if (IsBoardFull()) return 0.0f;  // Draw is 0 in Python version
    return 0.0f;  // Ongoing
}

bool TicTacToe::IsTerminal() const {
    return CheckWin(1) || CheckWin(-1) || IsBoardFull();
}

torch::Tensor TicTacToe::GetCanonicalBoard() const {
    auto tensor = torch::zeros({1, kBoardSize, kBoardSize});
    auto accessor = tensor.accessor<float, 3>();
    
    // Single channel with -1/0/1 values from current player's perspective
    for (int i = 0; i < kBoardSize; ++i) {
        for (int j = 0; j < kBoardSize; ++j) {
            accessor[0][i][j] = board_[i][j] * current_player_;
        }
    }
    return tensor;
}

std::unique_ptr<Game> TicTacToe::Clone() const {
    auto clone = std::make_unique<TicTacToe>();
    clone->board_ = board_;
    clone->current_player_ = current_player_;
    return clone;
}

bool TicTacToe::CheckWin(int player) const {
    // Check rows and columns
    for (int i = 0; i < kBoardSize; ++i) {
        bool row_win = true;
        bool col_win = true;
        for (int j = 0; j < kBoardSize; ++j) {
            if (board_[i][j] != player) row_win = false;
            if (board_[j][i] != player) col_win = false;
        }
        if (row_win || col_win) return true;
    }
    
    // Check diagonals
    bool diag1_win = true;
    bool diag2_win = true;
    for (int i = 0; i < kBoardSize; ++i) {
        if (board_[i][i] != player) diag1_win = false;
        if (board_[i][kBoardSize - 1 - i] != player) diag2_win = false;
    }
    
    return diag1_win || diag2_win;
}

bool TicTacToe::IsBoardFull() const {
    for (const auto& row : board_) {
        for (int cell : row) {
            if (cell == 0) return false;
        }
    }
    return true;
}

void TicTacToe::UndoMove(int move) {
    int row = move / kBoardSize;
    int col = move % kBoardSize;
    
    if (row < 0 || row >= kBoardSize || col < 0 || col >= kBoardSize) {
        throw std::invalid_argument("Invalid move to undo");
    }
    
    board_[row][col] = 0;  // Clear the position
    current_player_ = -current_player_;  // Switch back to previous player
}

}  // namespace alphazero 