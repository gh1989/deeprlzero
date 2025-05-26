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

#include "network.h"
#include "logger.h"
#include "games/positions.h"
#include "games/concepts.h"
#include "eval_stats.h"

namespace deeprlzero {

void Train(
  std::shared_ptr<torch::optim::Optimizer> optimizer, 
  std::shared_ptr<NeuralNetwork> network, 
  const Config& config, 
  const GamePositions& positions);

// Evaluation functions
template <typename ExplicitVariant>
requires GameConcept<ExplicitVariant>
EvaluationStats EvaluateAgainstNetwork(
  std::shared_ptr<NeuralNetwork> network,
  std::shared_ptr<NeuralNetwork> opponent,
  const Config& config);

template <typename ExplicitVariant>
requires GameConcept<ExplicitVariant>
EvaluationStats EvaluateAgainstRandom(
  std::shared_ptr<NeuralNetwork> network,
  const Config& config);

// Helper functions
bool AcceptOrRejectNewNetwork(
  std::shared_ptr<NeuralNetwork> network,
  std::shared_ptr<NeuralNetwork> candidate_network,
  const EvaluationStats& stats,
  const Config& config);

bool IsIdenticalNetwork(
  std::shared_ptr<NeuralNetwork> network1,
  std::shared_ptr<NeuralNetwork> network2);

torch::Tensor ComputePolicyLoss(
  const torch::Tensor& policy_preds,
  const torch::Tensor& policy_targets);

torch::Tensor ComputeValueLoss(
  const torch::Tensor& value_preds,
  const torch::Tensor& value_targets);

std::shared_ptr<NeuralNetwork> CreateInitialNetwork(const Config& config);
std::shared_ptr<NeuralNetwork> LoadBestNetwork(const Config& config);

} 

/// Template implementations
#include "trainer.inl"

#endif