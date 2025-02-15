#ifndef CORE_NEURAL_NETWORK_H_
#define CORE_NEURAL_NETWORK_H_

#include <torch/torch.h>

namespace alphazero {

// Container for your class declarations, e.g.:
class ResidualBlock : public torch::nn::Module {
 public:
  explicit ResidualBlock(int64_t num_filters);
  torch::Tensor forward(torch::Tensor x);
  // Other members...
};

class NeuralNetwork : public torch::nn::Module {
 public:
  NeuralNetwork(int64_t input_channels, int64_t num_filters,
                int64_t num_actions, int64_t num_residual_blocks);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  // Other members...
};

}  // namespace alphazero

#endif  // CORE_NEURAL_NETWORK_H_