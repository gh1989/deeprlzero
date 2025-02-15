#include "core/neural_network.h"

namespace alphazero {

ResidualBlock::ResidualBlock(int64_t num_filters) {
  conv1 = register_module("conv1", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3).padding(1)));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(num_filters));
  conv2 = register_module("conv2",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3).padding(1)));
  bn2 = register_module("bn2", torch::nn::BatchNorm2d(num_filters));
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
  auto identity = x;
  x = torch::relu(bn1(conv1(x)));
  x = bn2(conv2(x));
  x = x + identity;
  x = torch::relu(x);
  return x;
}

NeuralNetwork::NeuralNetwork(int64_t input_channels, 
                           int64_t num_filters,
                           int64_t num_actions,
                           int64_t num_residual_blocks)
    : conv(register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)))),
      bn(register_module("bn", torch::nn::BatchNorm2d(num_filters))) {
  // Residual blocks
  for (int i = 0; i < num_residual_blocks; ++i) {
    auto block = std::make_shared<ResidualBlock>(num_filters);
    residual_blocks.push_back(register_module("residual_block_" + std::to_string(i), block));
  }
  
  // Policy head - ensure dimensions match throughout
  policy_conv = register_module("policy_conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1)));  // Changed to 1 output channel
  policy_bn = register_module("policy_bn", 
      torch::nn::BatchNorm2d(1));
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(9, num_actions));  // 3x3 board flattened to 9
  
  // Value head
  value_conv = register_module("value_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1)));
  value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
  value_fc1 = register_module("value_fc1", torch::nn::Linear(9, 32));
  value_fc2 = register_module("value_fc2", torch::nn::Linear(32, 1));
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward(torch::Tensor x) {
  std::lock_guard<std::mutex> lock(forward_mutex_);
  if (cached_policy_.defined() && cached_policy_.numel() > 0) {
    cached_policy_.resize_(0);
  }

  // Process input through the network as usual.
  x = torch::relu(bn(conv(x)));
  for (const auto &block : residual_blocks) {
    x = block->forward(x);
  }
  
  // Policy head
  auto policy = policy_bn(policy_conv(x));
  policy = torch::relu(policy);
  policy = policy_fc(policy.flatten(1));
  policy = torch::softmax(policy, /*dim=*/1);
  
  // Value head
  auto value = value_bn(value_conv(x));
  value = torch::relu(value);
  value = torch::relu(value_fc1(value.flatten(1)));
  value = torch::tanh(value_fc2(value));
  
  return {policy, value};
}

}  // namespace alphazero 