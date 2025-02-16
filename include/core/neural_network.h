#ifndef ALPHAZERO_CORE_NEURAL_NETWORK_H_
#define ALPHAZERO_CORE_NEURAL_NETWORK_H_

#include <memory>
#include <vector>
#include <torch/torch.h>
#include <mutex>
#include "core/logger.h"

namespace alphazero {

class ResidualBlock : public torch::nn::Module {
 public:
  ResidualBlock(int64_t num_filters);
  torch::Tensor forward(torch::Tensor x);
  
  // Add non-const getter methods
  torch::nn::Conv2d& GetConv1() { return conv1; }
  torch::nn::Conv2d& GetConv2() { return conv2; }
  torch::nn::BatchNorm2d& GetBn1() { return bn1; }
  torch::nn::BatchNorm2d& GetBn2() { return bn2; }
  
  // Keep const getters for read-only access
  const torch::nn::Conv2d& GetConv1() const { return conv1; }
  const torch::nn::Conv2d& GetConv2() const { return conv2; }
  const torch::nn::BatchNorm2d& GetBn1() const { return bn1; }
  const torch::nn::BatchNorm2d& GetBn2() const { return bn2; }

 private:
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

};

class NeuralNetwork : public torch::nn::Module {
 public:
  NeuralNetwork(int64_t input_channels = 1, 
                int64_t num_filters = 32,
                int64_t num_actions = 9,
                int64_t num_residual_blocks = 3);

  virtual std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

  // Fix return type to match parent class
  std::shared_ptr<torch::nn::Module> clone(const torch::optional<torch::Device>& device = torch::nullopt) const override {
    auto cloned = std::make_shared<NeuralNetwork>(
      conv->options.in_channels(),
      conv->options.out_channels(),
      policy_fc->options.out_features(),
      residual_blocks.size()
    );
    
    // Copy all parameters
    cloned->conv->weight = conv->weight.clone();
    cloned->conv->bias = conv->bias.clone();
    cloned->bn->weight = bn->weight.clone();
    cloned->bn->bias = bn->bias.clone();
    cloned->bn->running_mean = bn->running_mean.clone();
    cloned->bn->running_var = bn->running_var.clone();
    
    // Update residual block cloning using getters
    for (size_t i = 0; i < residual_blocks.size(); ++i) {
      const auto& src_block = residual_blocks[i];
      auto& dst_block = cloned->residual_blocks[i];
      
      dst_block->GetConv1()->weight = src_block->GetConv1()->weight.clone();
      dst_block->GetConv1()->bias = src_block->GetConv1()->bias.clone();
      dst_block->GetBn1()->weight = src_block->GetBn1()->weight.clone();
      dst_block->GetBn1()->bias = src_block->GetBn1()->bias.clone();
      dst_block->GetBn1()->running_mean = src_block->GetBn1()->running_mean.clone();
      dst_block->GetBn1()->running_var = src_block->GetBn1()->running_var.clone();
      
      dst_block->GetConv2()->weight = src_block->GetConv2()->weight.clone();
      dst_block->GetConv2()->bias = src_block->GetConv2()->bias.clone();
      dst_block->GetBn2()->weight = src_block->GetBn2()->weight.clone();
      dst_block->GetBn2()->bias = src_block->GetBn2()->bias.clone();
      dst_block->GetBn2()->running_mean = src_block->GetBn2()->running_mean.clone();
      dst_block->GetBn2()->running_var = src_block->GetBn2()->running_var.clone();
    }
    
    // Clone policy head
    cloned->policy_conv->weight = policy_conv->weight.clone();
    cloned->policy_conv->bias = policy_conv->bias.clone();
    cloned->policy_bn->weight = policy_bn->weight.clone();
    cloned->policy_bn->bias = policy_bn->bias.clone();
    cloned->policy_bn->running_mean = policy_bn->running_mean.clone();
    cloned->policy_bn->running_var = policy_bn->running_var.clone();
    cloned->policy_fc->weight = policy_fc->weight.clone();
    cloned->policy_fc->bias = policy_fc->bias.clone();
    
    // Clone value head
    cloned->value_conv->weight = value_conv->weight.clone();
    cloned->value_conv->bias = value_conv->bias.clone();
    cloned->value_bn->weight = value_bn->weight.clone();
    cloned->value_bn->bias = value_bn->bias.clone();
    cloned->value_bn->running_mean = value_bn->running_mean.clone();
    cloned->value_bn->running_var = value_bn->running_var.clone();
    cloned->value_fc1->weight = value_fc1->weight.clone();
    cloned->value_fc1->bias = value_fc1->bias.clone();
    cloned->value_fc2->weight = value_fc2->weight.clone();
    cloned->value_fc2->bias = value_fc2->bias.clone();
    
    // Move to specified device if provided
    if (device.has_value()) {
      cloned->to(device.value());
    }
    
    return cloned;
  }

 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
  std::vector<std::shared_ptr<ResidualBlock>> residual_blocks;
  
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
  std::mutex forward_mutex_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_CORE_NEURAL_NETWORK_H_