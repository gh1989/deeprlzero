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

  config_ = config;
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
  auto cloned_net = std::make_shared<NeuralNetwork>(config_);
  auto typed_device = device.has_value() ? device.value() : torch::kCPU;

  // Explicitly copy parameters to ensure a deep copy
  torch::NoGradGuard no_grad;
  auto new_params = cloned_net->named_parameters();
  auto this_params = this->named_parameters(true);
  
  for (const auto& param : this_params) {
    auto name = param.key();
    auto& tensor = param.value();
    auto new_tensor = new_params[name];
    new_tensor.copy_(tensor);
  }
  
  cloned_net->to(typed_device);
  return cloned_net;
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
  Logger &logger = Logger::GetInstance();
  auto [policy_pred, value_pred] = forward(input);

  auto policy_loss =
      torch::nn::functional::cross_entropy(policy_pred, target_policy);
  auto value_loss = torch::nn::functional::mse_loss(value_pred, target_value);
  auto total_loss = policy_loss + value_loss;

  logger.Log("\n===== LOSS DETAILS =====\n");
  logger.LogFormat("Policy Loss: {:.4f}", policy_loss.item<float>());
  logger.LogFormat("Value Loss: {:.4f}", value_loss.item<float>());
  logger.LogFormat("Total Loss: {:.4f}", total_loss.item<float>());

  total_loss.backward();

  logger.Log("\n===== GRADIENT FLOW DETAILS =====\n");
  bool all_grads_ok = true;

  auto named_parameters = this->named_parameters();
  for (const auto& pair : named_parameters) {
    const std::string& name = pair.key();
    const torch::Tensor& param = pair.value();

    if (!param.grad().defined()) {
      logger.LogFormat("Warning: No gradient defined for: {}", name);
      all_grads_ok = false;
    } else {
      float grad_norm = param.grad().abs().sum().item<float>();
      if (grad_norm == 0) {
        logger.LogFormat("Warning: Zero gradient for: {}", name);
        all_grads_ok = false;
      } else {
        logger.LogFormat("Parameter: {}, Gradient norm: {}", name, grad_norm);
      }
    }
  }

  if (all_grads_ok) {
    logger.Log("\nGradient flow validated successfully.");
  } else {
    logger.Log("\nGradient flow issues detected. This may impact training "
                  "effectiveness.");
    logger.Log("Common causes:\n");
    logger.Log("1. Loss function not connected to all parameters\n");
    logger.Log("2. Some layers not contributing to the network output\n");
    logger.Log("3. Incorrect input or target tensor shapes\n");
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