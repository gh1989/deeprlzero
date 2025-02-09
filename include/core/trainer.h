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
  Trainer(int batch_size, int num_epochs, float learning_rate);
  
  void Train(std::shared_ptr<NeuralNetwork> network,
            const std::vector<GameExample>& examples);
  
 private:
  int batch_size_;
  int num_epochs_;
  float learning_rate_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_TRAINER_H_ 