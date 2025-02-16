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
   SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
    network_->to(torch::kCPU);
    network_->eval();
    mcts_stats_ = MCTSStats();
}

  std::vector<GameEpisode> ExecuteEpisodesParallel() {       
    omp_set_num_threads(config_.num_threads);
    std::vector<GameEpisode> episodes;
    ParallelFor(config_.episodes_per_iteration, [&](int episode_idx) {
      // Get a thread-local SelfPlay instance using the factory function.
      auto &local_self_play = GetThreadLocalInstance<SelfPlay<GameType>>(
          [&]() {
        return new SelfPlay<GameType>(network_, config_);
      });
   
      GameEpisode episode = local_self_play.ExecuteEpisode();

      // Enter critical section to safely update shared data.
      #ifdef _OPENMP
      #pragma omp critical
      #endif
          {
            episodes.push_back(std::move(episode));
          }
      });

      return episodes;
  }

  GameEpisode ExecuteEpisode() {
      network_->to(torch::kCPU);
      GameEpisode episode;
      std::vector<int> players;
      MCTS mcts(network_, config_);
      
      auto game = std::make_unique<GameType>();
      
      assert(game.get() != nullptr);
      assert(game->GetActionSize() == config_.action_size);
      
      while (!game->IsTerminal()) {
          torch::Tensor board = game->GetCanonicalBoard();
          std::vector<float> policy = mcts.GetActionProbabilities(game.get(), config_.temperature);
          int current_player = game->GetCurrentPlayer();
          players.push_back(current_player);
          
          episode.boards.push_back(board);
          episode.policies.push_back(policy);
          episode.values.push_back(0.0f);
          
          int move = mcts.SelectMove(game.get(), config_.temperature);
          game->MakeMove(move);
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
      
      // Backpropagate values from the end of the game
      for (int i = episode.values.size() - 1; i >= 0; --i) {
          episode.values[i] = (players[i] == 1) ? -final_result : final_result;
      }
      
      game->Reset();
      return episode;
  }
  
  const MCTSStats& GetStats() const { return mcts_stats_; }
  void ClearStats() { mcts_stats_ = MCTSStats(); }
  
 private:
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  std::mt19937 rng_{std::random_device{}()};
  MCTSStats mcts_stats_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_SELF_PLAY_H_ 