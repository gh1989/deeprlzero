#ifndef ALPHAZERO_SELF_PLAY_H_
#define ALPHAZERO_SELF_PLAY_H_

#include "core/game.h"
#include "core/mcts.h"
#include "core/config.h"
#include "core/mcts_stats.h"
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
      
      float final_result = game->GetGameResult();
      episode.outcome = final_result;
      
      for (size_t i = 0; i < episode.values.size(); ++i) {
          episode.values[i] = (players[i] == 1) ? final_result : -final_result;
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