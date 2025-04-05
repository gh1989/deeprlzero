#ifndef ALPHAZERO_SELF_PLAY_H_
#define ALPHAZERO_SELF_PLAY_H_

#include "core/game.h"
#include "core/mcts.h"
#include "core/config.h"
#include "core/mcts_stats.h"
#include "core/thread.h"
#include <vector>
#include <tuple>
#include <random>
#include <mutex>
#include <atomic>

namespace alphazero {

struct GameEpisode {
  std::vector<torch::Tensor> boards;           
  std::vector<std::vector<float>> policies;      
  std::vector<float> values;                     
  float outcome;                                 
};

template <typename GameType>
class SelfPlay {
  static_assert(std::is_base_of<Game, GameType>::value, "GameType must derive from Game");
 public:
   SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config, float temperature)
    : config_(config), current_temperature_(temperature) {
    // Make a clean copy of the network to avoid sharing issues
    network_ = std::dynamic_pointer_cast<NeuralNetwork>(network->clone(torch::kCPU));
    
    // Explicitly move to CPU and set to eval mode
    network_->to(torch::kCPU);
    network_->eval();
    
    mcts_stats_ = MCTSStats();
}

  std::vector<GameEpisode> ExecuteEpisodesParallel() {   
    // Set OpenMP parameters for optimal performance
    omp_set_dynamic(0);  // Disable dynamic adjustment of threads
    omp_set_nested(0);   // Disable nested parallelism    
    omp_set_num_threads(config_.num_threads);
    
    // Pre-allocate the episodes vector to avoid thread contention
    std::vector<GameEpisode> episodes;
    episodes.reserve(config_.episodes_per_iteration);
    
    std::cout << "\nGenerating " << config_.episodes_per_iteration << " episodes using " 
              << config_.num_threads << " threads..." << std::endl;
    
    // Use vector of vectors with mutex for thread safety
    std::vector<GameEpisode> all_episodes;
    std::mutex episodes_mutex;
    
    // Counter for completed episodes
    std::atomic<int> completed_episodes(0);
    
    // Parallel section - each thread handles multiple episodes
    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int num_threads = omp_get_num_threads();
      
      // Create a single self-play instance per thread to reuse across multiple episodes
      auto &thread_self_play = GetThreadLocalInstance<SelfPlay<GameType>>([&]() {
        return new SelfPlay<GameType>(network_, config_, current_temperature_);
      });
      
      // Calculate episode range for this thread - ensure positive size
      int episodes_per_thread = std::max(1, (config_.episodes_per_iteration + num_threads - 1) / num_threads);
      int start_idx = thread_id * episodes_per_thread;
      int end_idx = std::min(start_idx + episodes_per_thread, config_.episodes_per_iteration);
      
      // Local vector to store this thread's episodes
      std::vector<GameEpisode> thread_episodes;
      if (start_idx < config_.episodes_per_iteration) {
        // Only reserve if we have a valid range
        thread_episodes.reserve(end_idx - start_idx);
      
        // Process all episodes assigned to this thread
        for (int i = start_idx; i < end_idx; i++) {
          auto episode = thread_self_play.ExecuteEpisode();
          thread_episodes.push_back(std::move(episode));
          
          // Increment the counter and update progress display
          int current_completed = ++completed_episodes;
          
          // Update progress less frequently to reduce contention
          if (current_completed % 5 == 0 || current_completed == config_.episodes_per_iteration) {
            #pragma omp critical
            {
              std::cout << "Episodes completed: " << current_completed << "/" 
                        << config_.episodes_per_iteration << "\r" << std::flush;
            }
          }
        }
        
        // Add this thread's episodes to the main collection with thread safety
        std::lock_guard<std::mutex> lock(episodes_mutex);
        all_episodes.insert(all_episodes.end(), 
                           std::make_move_iterator(thread_episodes.begin()),
                           std::make_move_iterator(thread_episodes.end()));
      }
    }
    
    std::cout << "\nCompleted generating " << all_episodes.size() << " episodes." << std::endl;
    return all_episodes;
  }

  GameEpisode ExecuteEpisode() {
      network_->eval();
      
      GameEpisode episode;
      std::vector<int> players;
      MCTS mcts(network_, config_);
      
      auto game = std::make_unique<GameType>();
      
      assert(game.get() != nullptr);
      assert(game->GetActionSize() == config_.action_size);
      
      while (!game->IsTerminal()) {
          torch::Tensor board = game->GetCanonicalBoard();
          
          // Perform MCTS simulations before getting probabilities
          for (int i = 0; i < config_.num_simulations; ++i) {
              mcts.Search(game.get(), mcts.GetRoot());
          }
          
          std::vector<float> policy = mcts.GetActionProbabilities(game.get(), current_temperature_);
          int current_player = game->GetCurrentPlayer();
          players.push_back(current_player);
          
          episode.boards.push_back(board);
          episode.policies.push_back(policy);
          episode.values.push_back(0.0f);
          
          int move = mcts.SelectMove(game.get(), current_temperature_);
          game->MakeMove(move);
          mcts.ResetRoot();  // Reset the tree for the next move
      }
      
      // Add final board state
      episode.boards.push_back(game->GetCanonicalBoard());
      // Add zero policy for terminal state - no moves should be made
      std::vector<float> terminal_policy(config_.action_size, 0.0f);
      episode.policies.push_back(terminal_policy);
      episode.values.push_back(0.0f);
      players.push_back(game->GetCurrentPlayer());
      
      float final_result = game->GetGameResult();
      episode.outcome = final_result;
      
      // Backpropagate values from the end of the game with temporal discounting
      float gamma = 0.99; // Discount factor for future rewards
      float cumulative_value = final_result;
      for (int i = episode.values.size() - 1; i >= 0; --i) {
          episode.values[i] = cumulative_value;
          cumulative_value *= gamma; // Apply discount for earlier states
      }
      
      game->Reset();
      return episode;
  }
  
  const MCTSStats& GetStats() const { return mcts_stats_; }
  void ClearStats() { mcts_stats_ = MCTSStats(); }
  
 private:
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  float current_temperature_;
  std::mt19937 rng_{std::random_device{}()};
  MCTSStats mcts_stats_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_SELF_PLAY_H_ 