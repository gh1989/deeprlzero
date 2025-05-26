#include "network.h"
#include "logger.h"
#include "games/concepts.h"

#include <filesystem>
#include <cassert>

/*
Neural Network Architecture

    ┌───────────┐  ┌───────────┐
    │ Logits    │  │   Tanh    │
    │ (9 moves) │  │           │
    └───────────┘  └───────────┘

Residual Block
        ┌───────────────────┐
Input ──┤                   │
        │                   │
        ▼                   │
┌─────────────────┐         │
│  Conv2D (3×3)   │         │
└────────┬────────┘         │
         ▼                  │
┌─────────────────┐         │
│   BatchNorm2D   │         │
└────────┬────────┘         │
         ▼                  │
┌─────────────────┐         │
│      ReLU       │         │
└────────┬────────┘         │
         ▼                  │
┌─────────────────┐         │
│  Conv2D (3×3)   │         │
└────────┬────────┘         │
         ▼                  │
┌─────────────────┐         │
│   BatchNorm2D   │         │
└────────┬────────┘         │
         │                  │
         └────────┬─────────┘
                  ▼
                Output
*/

namespace deeprlzero {

ResidualBlock::ResidualBlock(int channels) {
  conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));
  conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
  bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
  
  // Initialize weights with Kaiming normal
  for (auto& p : parameters()) {
    if (p.dim() > 1) {
      // Conv weights
      torch::nn::init::kaiming_normal_(p);
    } else if (p.dim() == 1) {
      // BatchNorm weights/biases
      if (p.size(0) == channels) {
        torch::nn::init::constant_(p, 0.1);  // Batch norm weights
      } else {
        torch::nn::init::zeros_(p);  // Bias terms
      }
    }
  }
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
  auto out = torch::relu(bn1(conv1(x)));
  out = bn2(conv2(out));
  return x + out;
}


NeuralNetwork::NeuralNetwork(const Config& config, int board_size, int action_size) : config_(config) {
  assert(config.num_filters > 0);
  assert(board_size > 0);
  assert(action_size > 0); 

  const auto input_channels = 3;

  conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, config.num_filters, 3).padding(1)));
  batch_norm = register_module("batch_norm", torch::nn::BatchNorm2d(config.num_filters));

  for (int i = 0; i < config_.num_residual_blocks; i++) {
    res_blocks.push_back(register_module(
        "res_block_" + std::to_string(i), 
        std::make_shared<ResidualBlock>(config.num_filters)));
  }

  policy_fc = register_module("policy_fc", torch::nn::Linear(config.num_filters * board_size, action_size));
  value_fc = register_module("value_fc",  torch::nn::Linear(config.num_filters * board_size, 1));

  InitializeWeights();
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward( torch::Tensor x) {
  std::lock_guard<std::mutex> lock(*forward_mutex_); // :(
  return forward_impl(x);
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward_impl(
    torch::Tensor x) {
  x = conv(x);
  x = batch_norm(x);
  x = torch::relu(x);
  for (const auto& block : res_blocks) {
    x = block->forward(x);
  }
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

std::shared_ptr<NeuralNetwork> NeuralNetwork::NetworkClone(const torch::Device& device) const {
  auto cloned_net = std::make_shared<NeuralNetwork>(config_, board_size_, action_size_);
  
  torch::NoGradGuard no_grad;
  auto this_params = this->named_parameters(true);
  auto clone_params = cloned_net->named_parameters(true);
  
  for (const auto& param : this_params) {
    const auto& name = param.key();
    const auto& tensor = param.value();
    auto new_tensor = tensor.clone().detach();
    try {
      clone_params[name].copy_(new_tensor);
    } catch (const std::exception& e) {
      throw std::runtime_error("Parameter not found in cloned network: " + name + ": " + e.what());
    }
  }
  
  cloned_net->to(device);
  return cloned_net;
}

template <typename GameType>
requires GameConcept<GameType>
std::shared_ptr<NeuralNetwork> LoadBestNetwork(const Config& config) {
    Logger &logger = Logger::GetInstance(config);
    try {
        if (std::filesystem::exists(config.model_path)) {
            logger.LogFormat("Loading model from: {}", config.model_path);
            auto network = std::make_shared<NeuralNetwork>(config, GameType::GetBoardSize(), GameType::GetActionSize());
            torch::load(network, config.model_path);
            logger.Log("Model loaded successfully");
            return network;
        }
    } catch (const std::exception& e) {
        logger.LogFormat("Error loading model: {}", e.what());
    }
    return nullptr;
};
  
void SaveBestNetwork(std::shared_ptr<NeuralNetwork> network, const Config& config) {
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

float CalculatePolicyEntropy(const std::vector<float>& policy) {
  float entropy = 0.0f;
  
  for (const float& prob : policy) {
    if (prob > 1e-10) {
      entropy -= prob * std::log(prob);
    }
  }
  
  return entropy;
}

} 