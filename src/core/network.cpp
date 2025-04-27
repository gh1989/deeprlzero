#include "core/network.h"

namespace deeprlzero {

// Simplified constructor
NeuralNetwork::NeuralNetwork(
    int64_t input_channels,
    int64_t num_filters,
    int64_t num_actions,
    int64_t num_residual_blocks) {
  
  if (input_channels <= 0 || num_filters <= 0 || num_actions <= 0) {
    throw std::invalid_argument("Invalid neural network dimensions");
  }
  
  board_size_ = 3 * 3; // 3x3 for Tic-Tac-Toe
  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  batch_norm = register_module("batch_norm",
      torch::nn::BatchNorm2d(num_filters));
  
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(num_filters * board_size_, num_actions));
  
  value_fc = register_module("value_fc", 
      torch::nn::Linear(num_filters * board_size_, 1));

  InitializeWeights();
}

NeuralNetwork::NeuralNetwork(const Config& config) :
    NeuralNetwork(3, config.num_filters, config.action_size, config.num_residual_blocks) {
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward(torch::Tensor x) {
  std::lock_guard<std::mutex> lock(*forward_mutex_);
  return forward_impl(x);
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward_impl(torch::Tensor x) {
  x = conv(x);
  x = batch_norm(x);
  x = torch::relu(x);
  auto batch_size = x.size(0);
  x = x.reshape({batch_size, -1});
  auto policy_logits = policy_fc(x);
  auto value = torch::tanh(value_fc(x));
  cached_policy_ = policy_logits;
  cached_value_ = value;
  return std::make_pair(policy_logits, value);
}


void NeuralNetwork::InitializeWeights() {
  for (auto& p : parameters()) {
    if (p.dim() > 1) {
      torch::nn::init::kaiming_normal_(p);
      torch::nn::init::kaiming_normal_(p);
    } else if (p.dim() == 1) {
      if (p.size(0) == batch_norm->options.num_features()) {
        torch::nn::init::constant_(p, 0.1); 
      } else {
        torch::nn::init::zeros_(p);
      }
    }
  }
  
  batch_norm->train();
}

void NeuralNetwork::MoveToDevice(const torch::Device& device) {
  this->to(device);
}

void NeuralNetwork::reset() {
  int64_t input_channels = 3; // Default
  int64_t num_actions = 9;    // Default for 3x3 board
  int64_t num_filters = 32;   // Default
  
  if (conv) {
    input_channels = conv->options.in_channels();
    num_filters = conv->options.out_channels();
  }
  
  if (policy_fc) {
    num_actions = policy_fc->options.out_features();
  }

  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  batch_norm = register_module("batch_norm",
      torch::nn::BatchNorm2d(num_filters));
  
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(num_filters * board_size_, num_actions));
  
  value_fc = register_module("value_fc", 
      torch::nn::Linear(num_filters * board_size_, 1));
  
  InitializeWeights();
}

std::shared_ptr<torch::nn::Module> NeuralNetwork::clone(const torch::optional<torch::Device>& device) const {
    auto cloned = torch::nn::Cloneable<NeuralNetwork>::clone(device);
    auto typed_clone = std::dynamic_pointer_cast<NeuralNetwork>(cloned);
    
    typed_clone->forward_mutex_ = std::make_shared<std::mutex>();
    
    return cloned;
  }

}  // namespace deeprlzero 