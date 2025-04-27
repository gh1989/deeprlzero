#include "core/network.h"
#include "core/logger.h"

#include <filesystem>
#include <cassert>

namespace deeprlzero {

NeuralNetwork::NeuralNetwork(const Config& config) {
  /// pack in the asserts
  assert(config.num_filters > 0 );
  assert(config.action_size > 0 );

  /// Specialized for tic-tac-toe again :@
  /// also input channels is always three, ours, theirs and a whole channel for the turn.
  const auto board_size = 3 * 3;  
  const auto input_channels = 3;

  /// input channels, output channels, kernel size
  conv = register_module( "conv", torch::nn::Conv2d( torch::nn::Conv2dOptions(input_channels, config.num_filters, 3).padding(1)));
  /// batch normalization
  batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(config.num_filters));

  /// output policy which is the distribution over the actions, in tic-tac-toe this is obviously 9
  policy_fc = register_module( "policy_fc", torch::nn::Linear(config.num_filters * board_size, config.action_size));
  /// output value which is just a float
  value_fc = register_module("value_fc",  torch::nn::Linear(config.num_filters * board_size, 1));

  InitializeWeights();
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward( torch::Tensor x) {
  std::lock_guard<std::mutex> lock(*forward_mutex_);
  return forward_impl(x);
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward_impl(
    torch::Tensor x) {
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
  int64_t input_channels = 3; 
  int64_t num_actions = 9;  // specific to tic tac toe
  int64_t num_filters = 32;    

  if (conv) {
    input_channels = conv->options.in_channels();
    num_filters = conv->options.out_channels();
  }

  if (policy_fc) {
    num_actions = policy_fc->options.out_features();
  }

  conv = register_module( "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)));
  batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(num_filters));
  policy_fc = register_module( "policy_fc", torch::nn::Linear(num_filters * board_size_, num_actions));
  value_fc = register_module("value_fc", torch::nn::Linear(num_filters * board_size_, 1));

  InitializeWeights();
}

std::shared_ptr<torch::nn::Module> NeuralNetwork::clone(
    const torch::optional<torch::Device>& device) const {
  auto cloned = torch::nn::Cloneable<NeuralNetwork>::clone(device);
  auto typed_clone = std::dynamic_pointer_cast<NeuralNetwork>(cloned);

  typed_clone->forward_mutex_ = std::make_shared<std::mutex>();

  return cloned;
}

std::shared_ptr<NeuralNetwork> NeuralNetwork::CreateInitialNetwork(const Config& config) {
    try {
        return std::make_shared<NeuralNetwork>(config);
    } catch (const std::exception& e) {
        std::cerr << "Error creating network: " << e.what() << std::endl;
        return nullptr;
    }
};

std::shared_ptr<NeuralNetwork> NeuralNetwork::LoadBestNetwork(const Config& config) {
    Logger &logger = Logger::GetInstance(config);
    try {
        if (std::filesystem::exists(config.model_path)) {
            logger.LogFormat("Loading model from: {}", config.model_path);
            auto network = std::make_shared<NeuralNetwork>(config);
            torch::load(network, config.model_path);
            logger.Log("Model loaded successfully");
            return network;
        }
    } catch (const std::exception& e) {
        logger.LogFormat("Error loading model: {}", e.what());
    }
    return nullptr;
};
  
void NeuralNetwork::SaveBestNetwork(std::shared_ptr<NeuralNetwork> network, const Config& config) {
    Logger &logger = Logger::GetInstance(config);
    try {
        logger.LogFormat("Saving model to: {}", config.model_path);
        torch::save(network, config.model_path);
        logger.Log("Model saved successfully");
    } catch (const std::exception& e) {
        logger.LogFormat("Error saving model: {}", e.what());
    }
}

/// pretty print of the validation around the gradient flow.
void NeuralNetwork::ValidateGradientFlow(const torch::Tensor& input,
                          const torch::Tensor& target_policy,
                          const torch::Tensor& target_value) {

  zero_grad();

  auto [policy_pred, value_pred] = forward(input);

  auto policy_loss =
      torch::nn::functional::cross_entropy(policy_pred, target_policy);
  auto value_loss = torch::nn::functional::mse_loss(value_pred, target_value);
  auto total_loss = policy_loss + value_loss;

  std::cout << "\n===== LOSS DETAILS =====\n";
  std::cout << "Policy Loss: " << policy_loss.item<float>() << std::endl;
  std::cout << "Value Loss: " << value_loss.item<float>() << std::endl;
  std::cout << "Total Loss: " << total_loss.item<float>() << std::endl;

  total_loss.backward();

  std::cout << "\n===== GRADIENT FLOW DETAILS =====\n";
  bool all_grads_ok = true;

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

float NeuralNetwork::CalculatePolicyEntropy(const std::vector<float>& policy) {
  float entropy = 0.0f;
  
  for (const float& prob : policy) {
    // Avoid log(0) by adding a small epsilon
    if (prob > 1e-10) {
      entropy -= prob * std::log(prob);
    }
  }
  
  return entropy;
}

} 