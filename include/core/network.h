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
  NeuralNetwork(int64_t input_channels, int64_t num_filters,
                int64_t num_actions, int64_t num_residual_blocks = 0);

  NeuralNetwork(const Config& config);

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
  std::pair<torch::Tensor, torch::Tensor> forward_impl(torch::Tensor x);

  void MoveToDevice(const torch::Device& device);
  void InitializeWeights();

  // Getters for cached forward results (useful for analysis)
  const torch::Tensor& GetCachedPolicy() const { return cached_policy_; }
  const torch::Tensor& GetCachedValue() const { return cached_value_; }

  // Fix the clone method to match the base class return type
  std::shared_ptr<torch::nn::Module> clone(
      const torch::optional<torch::Device>& device =
          torch::nullopt) const override;

  void reset() override;

  // Validates gradient flow by checking gradients after a backward pass
  void ValidateGradientFlow(const torch::Tensor& input,
                            const torch::Tensor& target_policy,
                            const torch::Tensor& target_value) {
    // Clear any existing gradients
    zero_grad();

    // Forward pass
    auto [policy_pred, value_pred] = forward(input);

    // Compute loss
    auto policy_loss =
        torch::nn::functional::cross_entropy(policy_pred, target_policy);
    auto value_loss = torch::nn::functional::mse_loss(value_pred, target_value);
    auto total_loss = policy_loss + value_loss;

    std::cout << "\n===== LOSS DETAILS =====\n";
    std::cout << "Policy Loss: " << policy_loss.item<float>() << std::endl;
    std::cout << "Value Loss: " << value_loss.item<float>() << std::endl;
    std::cout << "Total Loss: " << total_loss.item<float>() << std::endl;

    // Backward pass
    total_loss.backward();

    // Check gradients for each parameter
    std::cout << "\n===== GRADIENT FLOW DETAILS =====\n";
    bool all_grads_ok = true;

    // Get all named parameters
    auto named_parameters = this->named_parameters();
    for (const auto& pair : named_parameters) {
      const std::string& name = pair.key();
      const torch::Tensor& param = pair.value();

      if (!param.grad().defined()) {
        std::cout << "Warning: No gradient defined for: " << name << std::endl;
        all_grads_ok = false;
      } else {
        float grad_norm = param.grad().abs().sum().item<float>();
        if (grad_norm == 0) {
          std::cout << "Warning: Zero gradient for: " << name << std::endl;
          all_grads_ok = false;
        } else {
          std::cout << "Parameter: " << name << ", Gradient norm: " << grad_norm
                    << std::endl;
        }
      }
    }

    if (all_grads_ok) {
      std::cout << "\nGradient flow validated successfully." << std::endl;
    } else {
      std::cout << "\nGradient flow issues detected. This may impact training "
                   "effectiveness."
                << std::endl;
      std::cout << "Common causes:\n";
      std::cout << "1. Loss function not connected to all parameters\n";
      std::cout << "2. Some layers not contributing to the network output\n";
      std::cout << "3. Incorrect input or target tensor shapes\n";
    }
  }

 private:
  // Network layers
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d batch_norm{nullptr};
  torch::nn::Linear policy_fc{nullptr};
  torch::nn::Linear value_fc{nullptr};

  // Cached results from forward pass
  torch::Tensor cached_policy_;
  torch::Tensor cached_value_;

  // Thread safety
  std::shared_ptr<std::mutex> forward_mutex_ = std::make_shared<std::mutex>();

  // Board dimensions
  int board_size_ = 9;  // 3x3 board
};

}  

#endif