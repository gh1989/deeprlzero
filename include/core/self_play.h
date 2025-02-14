#ifndef ALPHAZERO_SELF_PLAY_H_
#define ALPHAZERO_SELF_PLAY_H_

#include "core/game.h"
#include "core/mcts.h"
#include "core/config.h"
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
  
 private:
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace alphazero

#endif  // ALPHAZERO_SELF_PLAY_H_ 