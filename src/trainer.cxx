#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "mcts.h"
#include "trainer.h"
#include "logger.h"

namespace deeprlzero {

bool AcceptOrRejectNewNetwork(
    std::shared_ptr<NeuralNetwork> network,
    std::shared_ptr<NeuralNetwork> candidate_network,
    const EvaluationStats& stats,
    const Config& config
) {
    Logger& logger = Logger::GetInstance(config);
    bool network_accepted = false;
    float win_loss_ratio = stats.WinLossRatio();
                          
    if (win_loss_ratio >= config.acceptance_threshold) {             
        // Save the network
        SaveBestNetwork(network, config);
        network_accepted = true;
    } else {
        network_accepted = false;
    }

    logger.LogFormat("Network acceptance decision: {}", network_accepted ? "ACCEPTED" : "REJECTED");
    logger.Log(stats.WinStats());
    return network_accepted;
}

torch::Tensor ComputePolicyLoss(const torch::Tensor& policy_preds, const torch::Tensor& policy_targets) {
  return torch::nn::functional::cross_entropy(policy_preds, policy_targets);
}

torch::Tensor ComputeValueLoss(const torch::Tensor& value_preds, const torch::Tensor& value_targets) {
  return torch::mse_loss(value_preds, value_targets);
}

} 