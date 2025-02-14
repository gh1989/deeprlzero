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

struct GameExample {
  torch::Tensor board;
  std::vector<float> policy;
  float value;
};

class SelfPlay {
 public:
  SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config);
  
  std::vector<GameExample> ExecuteEpisode();
  
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