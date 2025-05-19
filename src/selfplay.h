#ifndef SELFPLAY_H
#define SELFPLAY_H

#include "mcts.h"
#include "logger.h"

#include "games/variant.h"
#include "games/positions.h"
#include "games/concepts.h"

namespace deeprlzero {

inline float CalculateAverageExplorationMetric(const std::vector<GamePositions>& episodes) {
  if (episodes.empty()) {
    return 0.0f;
  }
  
  float total_entropy = 0.0f;
  int total_moves = 0;
  
  for (const auto& episode : episodes) {
    for (const auto& policy : episode.policies) {
      total_entropy += CalculatePolicyEntropy(policy);
      total_moves++;
    }
  }
  Logger& logger = Logger::GetInstance();
  float exploration_metric = total_entropy / total_moves; 
  logger.LogFormat("Exploration metric: {:.4f}", exploration_metric);
  return exploration_metric;
}

template <typename GameVariant>
requires GameConcept<GameVariant> 
GamePositions ExecuteEpisode(std::shared_ptr<NeuralNetwork> network, const Config& config) {
  network->eval();

  GamePositions positions;
  MCTS mcts(network, config);

  GameVariant game;

  while (!IsTerminal(game)) {
    torch::Tensor board = GetCanonicalBoard(game);
    ///actually let's not.
    //mcts.AddDirichletNoiseToRoot(game.get());
    for (int i = 0; i < config.num_simulations; ++i) {
      mcts.Search(game, mcts.GetRoot());
    }

    // Use a smarter temperature approach that decreases as the game progresses
    float move_temperature = config.self_play_temperature;
    int move_number = positions.boards.size();

    std::vector<float> policy = mcts.GetActionProbabilities(game, move_temperature);
    positions.boards.push_back(board);
    positions.policies.push_back(policy);

    int move = mcts.SelectMove(game, move_temperature);
    MakeMove(game, move);
    mcts.ResetRoot();  
  }

  // Set the values based on game outcome without flipping perspective
  float outcome = GetGameResult(game);
  for (size_t i = 0; i < positions.boards.size(); i++) {
    positions.values.push_back(outcome);
  }
  
  // TODO: For games like chess, this simple approach won't work well.
  // We'll need to implement:
  // 1. Temporal discounting (γ^(T-t)) where future rewards are weighted less
  // 2. Intermediate value functions based on material/position
  // 3. TD(λ) or other bootstrapping approaches for long-term rewards
  // 4. Value targets that decay based on distance from terminal position

  return positions;
}

template <typename GameVariant>
requires GameConcept<GameVariant>
GamePositions ExecuteEpisodesParallel(std::shared_ptr<NeuralNetwork> network, const Config& config) {
  // First, run a sample episode to estimate position count
  auto sample = ExecuteEpisode<GameVariant>(network, config);
  int estimated_positions_per_episode = sample.boards.size();
  
  // Calculate total threads and distribution
  const int num_threads = std::min(config.num_threads, 
                                  std::max(1, config.episodes_per_iteration));
  const int episodes_per_thread = config.episodes_per_iteration / num_threads;
  const int remaining_episodes = config.episodes_per_iteration % num_threads;
  
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
    threads.push_back(std::thread([network, config, i, thread_episodes = thread_episode_counts[i], 
                                  &thread_results]() {
      for (int j = 0; j < thread_episodes; ++j) {
        thread_results[i].Append(ExecuteEpisode<GameVariant>(network, config));
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

template <typename GameVariant>
requires GameConcept<GameVariant>
GamePositions AllEpisodes() {  
  GamePositions all_positions;
  std::map<std::string, float> minimax_values;
  std::map<std::string, std::vector<float>> minimax_policies;
  
  std::function<float(const GameVariant&, std::vector<float>&)> exploreMinimax;
  exploreMinimax = [&all_positions, &minimax_values, &minimax_policies, &exploreMinimax](
      const GameVariant& state, std::vector<float>& best_policy) -> float {
    std::string key = GetBoardString(state) + std::to_string(GetCurrentPlayer(state));

    if (minimax_values.count(key)) {
      best_policy = minimax_policies[key];
      return minimax_values[key];
    }

    if (IsTerminal(state)) {
      float result = GetGameResult(state);
      minimax_values[key] = result;
      return result;
    }

    auto valid_moves = GetValidMoves(state);
    float best_value = -2.0f; 
    std::vector<float> policy(GetNumActions(state), 0.0f);
    std::vector<int> best_moves; 
    
    for (int move : valid_moves) {
      auto next_state = Clone(state);
      MakeMove(next_state, move);
      
      if (IsTerminal(next_state) && GetGameResult(next_state) == 1.0f) {
        policy[move] = 1.0f; 
        minimax_values[key] = 1.0f;
        minimax_policies[key] = policy;
        best_policy = policy;
        
        all_positions.boards.push_back(GetCanonicalBoard(state));
        all_positions.policies.push_back(policy);
        all_positions.values.push_back(1.0f);
        
        return 1.0f;
      }
    }

    for (int move : valid_moves) {
      // Think of this.
      auto next_state = Clone(state);
      MakeMove(next_state, move);
      
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

      all_positions.policies.push_back(policy);
      float adjusted_value = (GetCurrentPlayer(state) == 1) ? best_value : -best_value;
      all_positions.values.push_back(adjusted_value);
    }
        
    return best_value;
  };

  GameVariant initial_game;
  std::vector<float> initial_policy;
  exploreMinimax(initial_game, initial_policy);
  
  const int expected_total_positions = 5478; 
  const int expected_non_terminal_positions = 4520; 
  assert(minimax_values.size() == expected_total_positions );
  assert(all_positions.boards.size() == expected_non_terminal_positions);
  return all_positions;
}

}

#endif