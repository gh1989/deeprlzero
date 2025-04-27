#ifndef TRAINER_H_
#define TRAINER_H_

#include <torch/torch.h>

#include <deque>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "core/game.h"
#include "core/network.h"

namespace deeprlzero {

class Trainer {
 public:
  Trainer(std::shared_ptr<NeuralNetwork> network, const Config& config)
      : config_(config), network_(network) {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA is required for training");
    }

    // Move network to CUDA:0 before creating optimizer
    torch::Device cuda_device(torch::kCUDA, 0);
    network_->to(cuda_device);
    network_->train();

    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(), torch::optim::AdamOptions(config_.learning_rate)
                                    .weight_decay(config_.l2_reg));
  }
  torch::Tensor ComputePolicyLoss(const torch::Tensor& policy_preds,
                                  const torch::Tensor& policy_targets);
  torch::Tensor ComputeValueLoss(const torch::Tensor& value_preds,
                                 const torch::Tensor& value_targets);
  void Train(const std::vector<GameEpisode>& examples);

  // Add getters for training metrics
  float GetPolicyLoss() const { return last_policy_loss_; }
  float GetValueLoss() const { return last_value_loss_; }
  float GetTotalLoss() const { return last_total_loss_; }
  float GetParameterVariance() const { return last_param_variance_; }

 private:
  const Config& config_;
  std::shared_ptr<NeuralNetwork> network_;
  std::unique_ptr<torch::optim::Adam> optimizer_;

  // Track the latest losses
  float last_policy_loss_ = 0.0f;
  float last_value_loss_ = 0.0f;
  float last_total_loss_ = 0.0f;
  float last_param_variance_ = 0.0f;
};

// New structure to hold detailed evaluation outcomes.
struct EvaluationStats {
  float win_rate;
  float draw_rate;
  float loss_rate;

  std::string WinStats() const {
    return std::format("Win rate: {}%, Draw rate: {}%, Loss rate: {}%",
                       win_rate * 100, draw_rate * 100, loss_rate * 100);
  }

  bool IsBetterThan(const EvaluationStats& other) const {
    float score = win_rate + draw_rate * 0.5;
    float other_score = other.win_rate + other.draw_rate * 0.5;
    return score > other_score;
  }
};

class Evaluator {
 public:
  Evaluator(std::shared_ptr<NeuralNetwork> network, const Config& config);

  // Play against random player
  EvaluationStats EvaluateAgainstRandom();

  // Play two networks against each other
  EvaluationStats EvaluateNetworks(std::shared_ptr<NeuralNetwork> network1,
                                   std::shared_ptr<NeuralNetwork> network2,
                                   int num_games = 100);

  EvaluationStats EvaluateAgainstNetwork(
      std::shared_ptr<NeuralNetwork> opponent);

 private:
  bool IsIdenticalNetwork(std::shared_ptr<NeuralNetwork> network1,
                          std::shared_ptr<NeuralNetwork> network2);
  static constexpr int kNumEvaluationGames = 100;  // Fixed constant
  std::shared_ptr<NeuralNetwork> network_;
  const Config& config_;
};

}  // namespace deeprlzero

#endif