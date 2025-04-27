#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <memory>
#include <mutex>
#include <torch/torch.h>

#include "core/config.h"

namespace deeprlzero {

class NeuralNetwork : public torch::nn::Cloneable<NeuralNetwork>,
                      public std::enable_shared_from_this<NeuralNetwork> {
 public:
  NeuralNetwork(const Config& config);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  std::pair<torch::Tensor, torch::Tensor> forward_impl(torch::Tensor x);

  void MoveToDevice(const torch::Device& device);
  void InitializeWeights();
  const torch::Tensor& GetCachedPolicy() const { return cached_policy_; }
  const torch::Tensor& GetCachedValue() const { return cached_value_; }
  std::shared_ptr<torch::nn::Module> clone(
      const torch::optional<torch::Device>& device =
          torch::nullopt) const override;

  void reset() override;
  void ValidateGradientFlow(const torch::Tensor& input,
                            const torch::Tensor& target_policy,
                            const torch::Tensor& target_value); 

  static std::shared_ptr<NeuralNetwork> CreateInitialNetwork(const Config& config);
  static std::shared_ptr<NeuralNetwork> LoadBestNetwork(const Config& config);
  static void SaveBestNetwork(std::shared_ptr<NeuralNetwork> network, const Config& config);
  static float CalculatePolicyEntropy(const std::vector<float>& policy);

 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d batch_norm{nullptr};
  torch::nn::Linear policy_fc{nullptr};
  torch::nn::Linear value_fc{nullptr};
  torch::Tensor cached_policy_;
  torch::Tensor cached_value_;
  std::shared_ptr<std::mutex> forward_mutex_ = std::make_shared<std::mutex>();
  Config config_;
  int board_size_ = 9;  // tic-tac-toe
};

}  

#endif