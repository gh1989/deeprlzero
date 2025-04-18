#ifndef ALPHAZERO_TRAINER_H_
#define ALPHAZERO_TRAINER_H_

#include "core/neural_network.h"
#include "core/self_play.h"
#include <vector>
#include <deque>
#include <torch/torch.h>
#include <iostream>
#include <numeric>

namespace alphazero {

class Trainer {
 public:
  Trainer(std::shared_ptr<NeuralNetwork> network, const Config& config) 
      : config_(config), network_(network) 
  {
      if (!torch::cuda::is_available()) {
          throw std::runtime_error("CUDA is required for training");
      }
      
      // Move network to CUDA:0 before creating optimizer
      torch::Device cuda_device(torch::kCUDA, 0);
      network_->to(cuda_device);
      network_->train();
      
      optimizer_ = std::make_unique<torch::optim::Adam>(
          network_->parameters(),
          torch::optim::AdamOptions(config_.learning_rate).weight_decay(config_.l2_reg)
      );
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

}  // namespace alphazero

#endif  // ALPHAZERO_TRAINER_H_ 