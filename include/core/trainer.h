#ifndef ALPHAZERO_TRAINER_H_
#define ALPHAZERO_TRAINER_H_

#include "core/neural_network.h"
#include "core/self_play.h"
#include <vector>
#include <deque>
#include <torch/torch.h>
#include <iostream>
#include <numeric>

namespace alphazero {

class Trainer {
 public:
  Trainer(const Config& config) : config_(config) {}
  
  void Train(std::shared_ptr<NeuralNetwork> network,
            const std::vector<GameExample>& examples);
  
 private:
  const Config& config_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_TRAINER_H_ 