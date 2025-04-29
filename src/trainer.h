#ifndef TRAINER_H_
#define TRAINER_H_

#include <deque>
#include <format>
#include <iostream>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>
#include <torch/torch.h>

#include "game.h"
#include "network.h"
#include "logger.h"

namespace deeprlzero {

/// hold detailed evaluation outcomes.
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
  void Train(const GamePositions& examples);

  int IterationsSinceImprovement() const { return iterations_since_improvement_; }
  std::shared_ptr<NeuralNetwork> GetTrainedNetwork() const { return network_; }
  // Add getters for training metrics
  float GetPolicyLoss() const { return last_policy_loss_; }
  float GetValueLoss() const { return last_value_loss_; }
  float GetTotalLoss() const { return last_total_loss_; }
  float GetParameterVariance() const { return last_param_variance_; }
  
  float UpdateTemperature(float current_temperature);
  bool AcceptOrRejectNewNetwork( std::shared_ptr<NeuralNetwork> candidate_network, const EvaluationStats& stats );

  static std::shared_ptr<NeuralNetwork> CreateInitialNetwork(const Config& config);
  static std::shared_ptr<NeuralNetwork> LoadBestNetwork(const Config& config);
  
  EvaluationStats EvaluateAgainstRandom();
  EvaluationStats EvaluateNetworks(std::shared_ptr<NeuralNetwork> network1,
                                  std::shared_ptr<NeuralNetwork> network2,
                                  int num_games = 100);
  EvaluationStats EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent);

 private:
  const Config& config_;
  std::shared_ptr<NeuralNetwork> network_;
  std::unique_ptr<torch::optim::Adam> optimizer_;

  // Track the latest losses
  float last_policy_loss_ = 0.0f;
  float last_value_loss_ = 0.0f;
  float last_total_loss_ = 0.0f;
  float last_param_variance_ = 0.0f;
  int iterations_since_improvement_ = 0;
  bool IsIdenticalNetwork(std::shared_ptr<NeuralNetwork> network1,
                          std::shared_ptr<NeuralNetwork> network2);
  static constexpr int kNumEvaluationGames = 100;
};

}  

#endif