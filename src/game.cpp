#include <future>
#include <stdexcept>
#include <thread>

#include "mcts.h"
#include "game.h"

namespace deeprlzero {

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
  if (IsBoardFull()) return 0.0f;
  return 0.0f;  // Ongoing
}

bool TicTacToe::IsTerminal() const {
  return CheckWin(1) || CheckWin(-1) || IsBoardFull();
}

torch::Tensor TicTacToe::GetCanonicalBoard() const {
  auto tensor = torch::zeros({3, kBoardSize, kBoardSize});  // 3 channels!
  auto accessor = tensor.accessor<float, 3>();

  for (int i = 0; i < kBoardSize; ++i) {
    for (int j = 0; j < kBoardSize; ++j) {
      // Channel 0: Current player pieces (binary mask)
      accessor[0][i][j] = (board_[i][j] == current_player_) ? 1.0f : 0.0f;

      // Channel 1: Opponent pieces (binary mask)
      accessor[1][i][j] = (board_[i][j] == -current_player_) ? 1.0f : 0.0f;

      // Channel 2: Turn indicator (all 1s if player 1, all 0s if player 2)
      accessor[2][i][j] = (current_player_ == 1) ? 1.0f : 0.0f;
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

  board_[row][col] = 0;                // Clear the position
  current_player_ = -current_player_;  // Switch back to previous player
}

template <typename GameType>
GameEpisode SelfPlay<GameType>::ExecuteEpisode() {
  network_->eval();

  GameEpisode episode;
  MCTS mcts(network_, config_);

  auto game = std::make_unique<GameType>();

  while (!game->IsTerminal()) {
    torch::Tensor board = game->GetCanonicalBoard();
    
    mcts.AddDirichletNoiseToRoot(game.get());

    for (int i = 0; i < config_.num_simulations; ++i) {
      mcts.Search(game.get(), mcts.GetRoot());
    }

    std::vector<float> policy = mcts.GetActionProbabilities(game.get(), current_temperature_);
    episode.boards.push_back(board);
    episode.policies.push_back(policy);

    int move = mcts.SelectMove(game.get(), current_temperature_);
    game->MakeMove(move);
    mcts.ResetRoot();  // Reset the tree for the next move
  }

  episode.outcome = game->GetGameResult();
  return episode;
}

template <typename GameType>
std::vector<GameEpisode> SelfPlay<GameType>::ExecuteEpisodesParallel() {
  std::vector<GameEpisode> episodes;
  episodes.reserve(config_.episodes_per_iteration);

  // Use configured thread count, with sensible limits
  const int num_threads = std::min(config_.num_threads, 
                                  std::max(1, config_.episodes_per_iteration));
  
  // Calculate base episodes per thread and remainder
  const int episodes_per_thread = config_.episodes_per_iteration / num_threads;
  const int remaining_episodes = config_.episodes_per_iteration % num_threads;

  std::vector<std::future<std::vector<GameEpisode>>> futures;

  // Start threads with the work distribution
  for (int i = 0; i < num_threads; ++i) {
    // Give one extra episode to some threads to handle the remainder
    int thread_episodes = episodes_per_thread + (i < remaining_episodes ? 1 : 0);
    
    futures.push_back(
        std::async(std::launch::async, [this, thread_episodes]() {
          std::vector<GameEpisode> thread_episodes_result;
          thread_episodes_result.reserve(thread_episodes);
          for (int j = 0; j < thread_episodes; ++j) {
            thread_episodes_result.push_back(ExecuteEpisode());
          }
          return thread_episodes_result;
        }));
  }

  // Collect results
  for (auto& future : futures) {
    auto thread_episodes = future.get();
    episodes.insert(episodes.end(), thread_episodes.begin(),
                    thread_episodes.end());
  }

  return episodes;
}

template GameEpisode SelfPlay<TicTacToe>::ExecuteEpisode();
template std::vector<GameEpisode>
SelfPlay<TicTacToe>::ExecuteEpisodesParallel();

}  