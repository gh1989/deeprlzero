#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <memory>
#include <mutex>
#include <torch/torch.h>
#include <vector>

#include "config.h"

namespace deeprlzero {

// Residual block as a separate module
class ResidualBlock : public torch::nn::Module,  public std::enable_shared_from_this<ResidualBlock>  {
 public:
  ResidualBlock(int channels);
  torch::Tensor forward(torch::Tensor x);

 private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
};

class NeuralNetwork : public torch::nn::Module, public std::enable_shared_from_this<NeuralNetwork> {
 public:
  NeuralNetwork(const Config& config);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  std::pair<torch::Tensor, torch::Tensor> forward_impl(torch::Tensor x);

  void MoveToDevice(const torch::Device& device);
  void InitializeWeights();
  const torch::Tensor& GetCachedPolicy() const { return cached_policy_; }
  const torch::Tensor& GetCachedValue() const { return cached_value_; }
std::shared_ptr<NeuralNetwork> NetworkClone(const torch::Device& device) const;
void ValidateGradientFlow(const torch::Tensor& input,
                          const torch::Tensor& target_policy,
                          const torch::Tensor& target_value); 
 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d batch_norm{nullptr};
  std::vector<std::shared_ptr<ResidualBlock>> res_blocks;
  torch::nn::Linear policy_fc{nullptr};
  torch::nn::Linear value_fc{nullptr};
  torch::Tensor cached_policy_;
  torch::Tensor cached_value_;
  std::shared_ptr<std::mutex> forward_mutex_ = std::make_shared<std::mutex>();
  Config config_;
  int board_size_ = 9;  // tic-tac-toe
};


float CalculatePolicyEntropy(const std::vector<float>& policy);

std::shared_ptr<NeuralNetwork> CreateInitialNetwork(const Config& config);
std::shared_ptr<NeuralNetwork> LoadBestNetwork(const Config& config);
void SaveBestNetwork(std::shared_ptr<NeuralNetwork> network, const Config& config);

}  

#endif