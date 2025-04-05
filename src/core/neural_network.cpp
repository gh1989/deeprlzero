#include "core/neural_network.h"

namespace alphazero {

// Simplified constructor
NeuralNetwork::NeuralNetwork(
    int64_t input_channels,
    int64_t num_filters,
    int64_t num_actions,
    int64_t num_residual_blocks) {
  
  // Validate dimensions
  if (input_channels <= 0 || num_filters <= 0 || num_actions <= 0) {
    throw std::invalid_argument("Invalid neural network dimensions");
  }
  
  // Store the board size for convenience (assuming square board)
  board_size_ = 3 * 3; // 3x3 for Tic-Tac-Toe
  
  // Initial convolution layer
  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  // Add batch normalization for training stability
  batch_norm = register_module("batch_norm",
      torch::nn::BatchNorm2d(num_filters));
  
  // Policy head - no intermediate convolution, just flatten and linear
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(num_filters * board_size_, num_actions));
  
  // Value head - no intermediate convolution, just flatten and linear
  value_fc = register_module("value_fc", 
      torch::nn::Linear(num_filters * board_size_, 1));
  
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
  // First, let's print some diagnostic info
  //std::cout << "Input shape: " << x.sizes() << " Mean: " << x.mean().item<float>() << std::endl;
  //std::cout << "Input has non-zero values: " << (x.abs().sum().item<float>() > 0) << std::endl;
  
  // Apply convolutional layer
  x = conv(x);
  //std::cout << "After conv: " << x.sizes() << " Mean: " << x.mean().item<float>() << std::endl;
  
  // Apply batch norm and activation separately to check each step
  x = batch_norm(x);
  //std::cout << "After batch_norm: " << x.sizes() << " Mean: " << x.mean().item<float>() << std::endl;
  
  x = torch::relu(x);
  //std::cout << "After relu: " << x.sizes() << " Mean: " << x.mean().item<float>() << std::endl;
  
  // Store original shape for reshaping later
  auto batch_size = x.size(0);
  
  // Flatten the tensor for the fully connected layers - CAREFUL WITH THIS STEP!
  //std::cout << "Before flatten shape: " << x.sizes() << std::endl;
  x = x.reshape({batch_size, -1});
  //std::cout << "After flatten shape: " << x.sizes() << std::endl;
  
  // Policy head
  auto policy_logits = policy_fc(x);
  //std::cout << "Policy logits: " << policy_logits.sizes() << " Mean: " << policy_logits.mean().item<float>() << std::endl;
  
  // Value head
  auto value = torch::tanh(value_fc(x));
  //std::cout << "Value: " << value.sizes() << " Mean: " << value.mean().item<float>() << std::endl;
  
  // Store for caching WITHOUT breaking graph
  cached_policy_ = policy_logits;
  cached_value_ = value;
  
  return std::make_pair(policy_logits, value);
}


void NeuralNetwork::InitializeWeights() {
  // Use He initialization for ReLU networks
  for (auto& p : parameters()) {
    if (p.dim() > 1) {
      // Use kaiming_normal_ initialization which is better for ReLU
      torch::nn::init::kaiming_normal_(p);
    } else if (p.dim() == 1) {
      // Use small positive bias for ReLU activation
      if (p.size(0) == batch_norm->options.num_features()) {
        // For batch norm biases, use small positive values
        torch::nn::init::constant_(p, 0.1); 
      } else {
        // For other biases
        torch::nn::init::zeros_(p);
      }
    }
  }
  
  // Explicitly set batch norm to training mode
  batch_norm->train();
}

void NeuralNetwork::MoveToDevice(const torch::Device& device) {
  this->to(device);
}

void NeuralNetwork::reset() {
  // Get dimensions from existing layers
  int64_t input_channels = 1; // Default
  int64_t num_actions = 9;    // Default for 3x3 board
  int64_t num_filters = 32;   // Default

  // If existing layers exist, get their dimensions
  if (conv) {
    input_channels = conv->options.in_channels();
    num_filters = conv->options.out_channels();
  }
  
  if (policy_fc) {
    num_actions = policy_fc->options.out_features();
  }

  // Initial convolution layer
  conv = register_module("conv", 
      torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  
  // Add batch normalization
  batch_norm = register_module("batch_norm",
      torch::nn::BatchNorm2d(num_filters));
  
  // Policy head - direct from features to action logits
  policy_fc = register_module("policy_fc", 
      torch::nn::Linear(num_filters * board_size_, num_actions));
  
  // Value head - direct from features to value
  value_fc = register_module("value_fc", 
      torch::nn::Linear(num_filters * board_size_, 1));
  
  // Initialize weights
  InitializeWeights();
}

std::shared_ptr<torch::nn::Module> NeuralNetwork::clone(const torch::optional<torch::Device>& device) const {
    auto cloned = torch::nn::Cloneable<NeuralNetwork>::clone(device);
    auto typed_clone = std::dynamic_pointer_cast<NeuralNetwork>(cloned);
    
    // Set up a new mutex for the clone
    typed_clone->forward_mutex_ = std::make_shared<std::mutex>();
    
    return cloned;
  }

}  // namespace alphazero 