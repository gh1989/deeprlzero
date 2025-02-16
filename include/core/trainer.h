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
  Trainer(std::shared_ptr<NeuralNetwork> network, const Config& config) : config_(config), network_(network) 
  {
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(config_.learning_rate).weight_decay(config_.l2_reg)
    );
  };
  torch::Tensor ComputePolicyLoss(const torch::Tensor& policy_preds,
                                const torch::Tensor& policy_targets);
  torch::Tensor ComputeValueLoss(const torch::Tensor& value_preds,
                                const torch::Tensor& value_targets);
  void Train(std::shared_ptr<NeuralNetwork> network,
            const std::vector<GameEpisode>& examples);
  
 private:
  const Config& config_;
  std::shared_ptr<NeuralNetwork> network_;
  std::unique_ptr<torch::optim::Adam> optimizer_;
};

}  // namespace alphazero

#endif  // ALPHAZERO_TRAINER_H_ 