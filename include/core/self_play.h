#ifndef ALPHAZERO_SELF_PLAY_H_
#define ALPHAZERO_SELF_PLAY_H_

#include "core/game.h"
#include "core/mcts.h"
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
  SelfPlay(std::shared_ptr<NeuralNetwork> network, int num_simulations, float c_puct, float temperature = 1.0f);
  
  std::vector<GameExample> ExecuteEpisode();
  
 private:
  std::shared_ptr<NeuralNetwork> network_;
  int num_simulations_;
  float c_puct_;
  float temperature_;
  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace alphazero

#endif  // ALPHAZERO_SELF_PLAY_H_ 