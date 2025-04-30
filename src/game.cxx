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

std::string TicTacToe::GetBoardString() const {
  std::string result;
  for (int i = 0; i < kBoardSize; ++i) {
    for (int j = 0; j < kBoardSize; ++j) {
      if (board_[i][j] == 1) result += "1";
      else if (board_[i][j] == -1) result += "2";  // Using 2 for player -1
      else result += "0";
    }
  }
  return result;
}

void TicTacToe::SetFromString(const std::string& boardStr, int player) {
  if (boardStr.size() != kBoardSize * kBoardSize) {
    throw std::invalid_argument("Invalid board string length");
  }
  
  current_player_ = player;
  
  for (int i = 0; i < kBoardSize; ++i) {
    for (int j = 0; j < kBoardSize; ++j) {
      char c = boardStr[i * kBoardSize + j];
      if (c == '1') board_[i][j] = 1;
      else if (c == '2') board_[i][j] = -1;  // 2 represents player -1
      else board_[i][j] = 0;
    }
  }
}

template <typename GameType>
GamePositions SelfPlay<GameType>::ExecuteEpisode() {
  network_->eval();

  GamePositions positions;
  MCTS mcts(network_, config_);

  auto game = std::make_unique<GameType>();

  while (!game->IsTerminal()) {
    torch::Tensor board = game->GetCanonicalBoard();
    ///actually let's not.
    //mcts.AddDirichletNoiseToRoot(game.get());
    for (int i = 0; i < config_.num_simulations; ++i) {
      mcts.Search(game.get(), mcts.GetRoot());
    }

    // Use a smarter temperature approach that decreases as the game progresses
    float move_temperature = config_.self_play_temperature;
    int move_number = positions.boards.size();

    std::vector<float> policy = mcts.GetActionProbabilities(game.get(), move_temperature);
    positions.boards.push_back(board);
    positions.policies.push_back(policy);

    int move = mcts.SelectMove(game.get(), move_temperature);
    game->MakeMove(move);
    mcts.ResetRoot();  
  }

  // Set the values based on game outcome. Flip value for player 2's perspective
  float outcome = game->GetGameResult();
  for (size_t i = 0; i < positions.boards.size(); i++) {
    positions.values.push_back((i % 2 == 0) ? outcome : -outcome);
  }

  return positions;
}

// Should add nodiscard and expected error or GamePositions
template <typename GameType>
GamePositions SelfPlay<GameType>::ExecuteEpisodesParallel() {
  // First, run a sample episode to estimate position count
  auto sample = ExecuteEpisode();
  int estimated_positions_per_episode = sample.boards.size();
  
  // Calculate total threads and distribution
  const int num_threads = std::min(config_.num_threads, 
                                  std::max(1, config_.episodes_per_iteration));
  const int episodes_per_thread = config_.episodes_per_iteration / num_threads;
  const int remaining_episodes = config_.episodes_per_iteration % num_threads;
  
  // Pre-compute episode assignments and positions
  std::vector<int> thread_episode_counts(num_threads);
  std::vector<int> position_offsets(num_threads);
  int total_estimated_positions = 0;
  
  for (int i = 0; i < num_threads; ++i) {
    thread_episode_counts[i] = episodes_per_thread + (i < remaining_episodes ? 1 : 0);
    position_offsets[i] = total_estimated_positions;
    total_estimated_positions += thread_episode_counts[i] * estimated_positions_per_episode;
  }
  
  // Pre-allocate the result vectors with generous padding
  GamePositions all_positions;
  const float padding_factor = 1.25;
  all_positions.boards.reserve(total_estimated_positions * padding_factor);
  all_positions.policies.reserve(total_estimated_positions * padding_factor);
  all_positions.values.reserve(total_estimated_positions * padding_factor);
  
  // Store thread results separately to avoid contention
  std::vector<GamePositions> thread_results(num_threads);
  
  // Launch threads
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread([this, i, thread_episodes = thread_episode_counts[i], 
                                  &thread_results]() {
      for (int j = 0; j < thread_episodes; ++j) {
        thread_results[i].Append(ExecuteEpisode());
      }
    }));
  }
  
  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }
  
  // Merge results sequentially to avoid allocation thrashing
  for (int i = 0; i < num_threads; ++i) {
    all_positions.Append(thread_results[i]);
  }
  
  return all_positions;
}

template GamePositions SelfPlay<TicTacToe>::ExecuteEpisode();
template GamePositions SelfPlay<TicTacToe>::ExecuteEpisodesParallel();

}  