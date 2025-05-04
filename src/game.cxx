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

  // Set the values based on game outcome without flipping perspective
  float outcome = game->GetGameResult();
  for (size_t i = 0; i < positions.boards.size(); i++) {
    positions.values.push_back(outcome);
  }

  return positions;
}

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
  
  std::vector<GamePositions> thread_results(num_threads);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.push_back(std::thread([this, i, thread_episodes = thread_episode_counts[i], 
                                  &thread_results]() {
      for (int j = 0; j < thread_episodes; ++j) {
        thread_results[i].Append(ExecuteEpisode());
      }
    }));
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  for (int i = 0; i < num_threads; ++i) {
    all_positions.Append(thread_results[i]);
  }
  
  return all_positions;
}

template <typename GameType> 
GamePositions SelfPlay<GameType>::AllEpisodes() {  
  GamePositions all_positions;
  std::map<std::string, float> minimax_values;
  std::map<std::string, std::vector<float>> minimax_policies;
  
  std::function<float(const GameType&, std::vector<float>&)> exploreMinimax;
  exploreMinimax = [&all_positions, &minimax_values, &minimax_policies, &exploreMinimax](
      const GameType& state, std::vector<float>& best_policy) -> float {
    std::string key = state.GetBoardString() + std::to_string(state.GetCurrentPlayer());

    if (minimax_values.count(key)) {
      best_policy = minimax_policies[key];
      return minimax_values[key];
    }

    if (state.IsTerminal()) {
      float result = state.GetGameResult();
      minimax_values[key] = result;
      return result;
    }

    auto valid_moves = state.GetValidMoves();
    float best_value = -2.0f; 
    std::vector<float> policy(GameType::kNumActions, 0.0f);
    std::vector<int> best_moves; 
    
    for (int move : valid_moves) {
      GameType next_state(state);
      next_state.MakeMove(move);
      
      if (next_state.IsTerminal() && next_state.GetGameResult() == 1.0f) {
        policy[move] = 1.0f; 
        minimax_values[key] = 1.0f;
        minimax_policies[key] = policy;
        best_policy = policy;
        
        all_positions.boards.push_back(state.GetCanonicalBoard());
        all_positions.policies.push_back(policy);
        all_positions.values.push_back(1.0f);
        
        return 1.0f;
      }
    }

    for (int move : valid_moves) {
      GameType next_state(state);
      next_state.MakeMove(move);
      
      std::vector<float> child_policy;
      float child_value = -exploreMinimax(next_state, child_policy); 
      
      if (child_value > best_value + 1e-6f) {  
        best_value = child_value;
        best_moves.clear();
        best_moves.push_back(move);
      }
      else if (std::abs(child_value - best_value) <= 1e-6) {  
        best_moves.push_back(move);
      }
    }
    
    for (int move : best_moves) {
      policy[move] = 1.0f / best_moves.size();
    }
    
    minimax_values[key] = best_value;
    minimax_policies[key] = policy;
    best_policy = policy;
    
    if (!state.IsTerminal()) {
      all_positions.boards.push_back(state.GetCanonicalBoard());
      all_positions.policies.push_back(policy);
      float adjusted_value = (state.GetCurrentPlayer() == 1) ? best_value : -best_value;
      all_positions.values.push_back(adjusted_value);
    }
        
    return best_value;
  };

  GameType initial_game;
  std::vector<float> initial_policy;
  exploreMinimax(initial_game, initial_policy);
  
  const int expected_total_positions = 5478; 
  const int expected_non_terminal_positions = 4520; 
  assert(minimax_values.size() == expected_total_positions );
  assert(all_positions.boards.size() == expected_non_terminal_positions);
  return all_positions;
}

template GamePositions SelfPlay<TicTacToe>::ExecuteEpisode();
template GamePositions SelfPlay<TicTacToe>::ExecuteEpisodesParallel();
template GamePositions SelfPlay<TicTacToe>::AllEpisodes();

}  