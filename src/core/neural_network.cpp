#include "core/neural_network.h"

namespace alphazero {

// Simplified constructor - no residual blocks
NeuralNetwork::NeuralNetwork(
    int64_t input_channels,
    int64_t num_filters,
    int64_t num_actions,
    int64_t num_residual_blocks) {
  
  // Validate dimensions
  if (input_channels <= 0 || num_filters <= 0 || num_actions <= 0) {
    throw std::invalid_argument("Invalid neural network dimensions");
  }
  
  // Initial convolution layer
  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  // Policy head - just one conv + fc
  policy_conv = register_module("policy_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 2, 1)));
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(2 * 3 * 3, num_actions));
  
  // Value head - just one conv + fc
  value_conv = register_module("value_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1)));
  value_fc2 = register_module("value_fc2", 
      torch::nn::Linear(3 * 3, 1));
  
  // Initialize weights
  InitializeWeights();
}

// Constructor from config
NeuralNetwork::NeuralNetwork(const Config& config) :
    NeuralNetwork(1, config.num_filters, config.action_size, config.num_residual_blocks) {
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward(torch::Tensor x) {
  std::lock_guard<std::mutex> lock(*forward_mutex_);
  return forward_impl(x);
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward_impl(torch::Tensor x) {
  //std::lock_guard<std::mutex> lock(*forward_mutex_);
  
  // Initial convolution
  x = torch::relu(conv(x));
  
  // Policy head
  auto p = torch::relu(policy_conv(x));
  p = p.view({p.size(0), -1});  // Flatten
  auto policy = torch::log_softmax(policy_fc(p), 1);
  
  // Value head
  auto v = torch::relu(value_conv(x));
  v = v.view({v.size(0), -1});  // Flatten
  auto value = torch::tanh(value_fc2(v));
  
  // Cache results
  cached_policy_ = policy;
  cached_value_ = value;
  
  return {policy, value};
}

void NeuralNetwork::InitializeWeights() {
  // Use larger initialization to help gradient flow
  for (auto& p : parameters()) {
    if (p.dim() > 1) {
      // Use xavier_normal_ with higher gain for better gradient flow
      torch::nn::init::xavier_normal_(p, 1.5);
    } else if (p.dim() == 1) {
      // Use small positive bias to avoid dead neurons
      torch::nn::init::constant_(p, 0.1);
    }
  }
}

void NeuralNetwork::MoveToDevice(const torch::Device& device) {
  this->to(device);
}

void NeuralNetwork::reset() {
  // Simply recreate our minimal network
  const int board_size = 3 * 3;
  
  int64_t input_channels = 1; // Default
  int64_t num_actions = 9;    // Default for 3x3 board
  int64_t num_filters = 32;

  // If existing layers exist, get their dimensions
  if (policy_fc) {
    num_actions = policy_fc->options.out_features();
    input_channels = policy_fc->options.in_features() / board_size;
  }
  
 if (conv) {
    num_filters = conv->options.out_channels();
 }

  // Initial convolution layer
  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  policy_conv = register_module("policy_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 2, 1)));

  value_conv = register_module("value_conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1)));

  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(input_channels * board_size, num_actions));
  
  value_fc2 = register_module("value_fc2", 
      torch::nn::Linear(input_channels * board_size, 1));
  
  // Initialize weights
  InitializeWeights();
}

/*
void NeuralNetwork::save(const std::string& path) {
  torch::save(std::enable_shared_from_this<NeuralNetwork>::shared_from_this(), path);
}

void NeuralNetwork::load(const std::string& path) {
  torch::load(std::enable_shared_from_this<NeuralNetwork>::shared_from_this(), path);
}

std::shared_ptr<NeuralNetwork> NeuralNetwork::Clone(torch::Device device) const {
  auto cloned = std::make_shared<NeuralNetwork>(
    conv->options.in_channels(),
    conv->options.out_channels(),
    policy_fc->options.out_features()
  );
  
  // Copy all parameters
  for (auto& item : named_parameters()) {
    auto name = item.key();
    auto param = item.value();
    
    if (cloned->named_parameters().contains(name)) {
      cloned->named_parameters()[name].copy_(param);
    }
  }
  
  // Copy all buffers (running means/vars for batch norm)
  for (auto& item : named_buffers()) {
    auto name = item.key();
    auto buffer = item.value();
    
    if (cloned->named_buffers().contains(name)) {
      cloned->named_buffers()[name].copy_(buffer);
    }
  }
  
  // Move to specified device
  cloned->to(device);
  
  return cloned;
}
*/
}  // namespace alphazero 