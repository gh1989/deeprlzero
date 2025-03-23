#ifndef ALPHAZERO_CORE_NEURAL_NETWORK_H_
#define ALPHAZERO_CORE_NEURAL_NETWORK_H_

#include <torch/torch.h>
#include <memory>
#include <mutex>
#include "core/config.h"

namespace alphazero {

class NeuralNetwork : public torch::nn::Cloneable<NeuralNetwork>,
                     public std::enable_shared_from_this<NeuralNetwork> {
 public:
  // Simplified constructor without residual blocks
  NeuralNetwork(int64_t input_channels, 
                int64_t num_filters,
                int64_t num_actions,
                int64_t num_residual_blocks=2);
  
  // Constructor from config
  explicit NeuralNetwork(const Config& config);
  
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  
  // Fix the clone method to match the base class return type
  std::shared_ptr<torch::nn::Module> clone(const torch::optional<torch::Device>& device = torch::nullopt) const override {
    return torch::nn::Cloneable<NeuralNetwork>::clone(device);
  }
  
  // Implement the required reset() method from Cloneable
  void reset() override;
  
  void InitializeWeights();
  void MoveToDevice(const torch::Device& device);
  
  //void save(const std::string& path);
  //void load(const std::string& path);
  
 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  
  // Policy head
  torch::nn::Conv2d policy_conv{nullptr};
  torch::nn::BatchNorm2d policy_bn{nullptr};
  torch::nn::Linear policy_fc{nullptr};
  
  // Value head
  torch::nn::Conv2d value_conv{nullptr};
  torch::nn::BatchNorm2d value_bn{nullptr};
  torch::nn::Linear value_fc1{nullptr};
  torch::nn::Linear value_fc2{nullptr};

  torch::Tensor cached_policy_;   
  torch::Tensor cached_value_;
  //std::mutex forward_mutex_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_CORE_NEURAL_NETWORK_H_