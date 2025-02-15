#include "core/trainer.h"
#include <torch/torch.h>
#include <random>

namespace alphazero {

torch::Tensor Trainer::ComputePolicyLoss(const torch::Tensor& policy_preds,
                                const torch::Tensor& policy_targets) {
  // Add a small constant to avoid log(0)
  constexpr float kEpsilon = 1e-8f;
  // Compute log(probabilities)
  auto log_policy = (policy_preds + kEpsilon).log();

  // Cross entropy loss: L_policy = -sum( pi * log(p) )
  // We sum over the action dimension and then take the mean over the batch.
  auto cross_entropy_loss = -torch::mean(torch::sum(policy_targets * log_policy, /*dim=*/1));

  // Entropy of the predictions: H(p) = -sum( p * log(p) )
  // Taking the mean over the batch.
  auto entropy = -torch::mean(torch::sum(policy_preds * log_policy, /*dim=*/1));

  // Entropy coefficient can be tuned (usually a small value such as 0.01 or so)
  constexpr float kEntropyCoef = 0.01f;

  // Total policy loss = cross entropy loss minus the entropy bonus.
  // (Subtracting entropy encourages the network to remain somewhat uncertain,
  //   promoting exploration.)
  auto loss = cross_entropy_loss - kEntropyCoef * entropy;

  return loss;
}

torch::Tensor Trainer::ComputeValueLoss(const torch::Tensor& value_preds,
                               const torch::Tensor& value_targets) {
  // Compute the Mean Squared Error loss between predicted and target values.
  // This loss measures the average squared difference between the predictions and targets.
  return torch::mse_loss(value_preds, value_targets);
}

void Trainer::Train(std::shared_ptr<NeuralNetwork> network,
                   const std::vector<GameExample>& examples) {
  if (examples.empty()) return;
    
  // Ensure CUDA is available for training
  if (!torch::cuda::is_available()) {
      throw std::runtime_error("CUDA is required for training");
  }
    
  // Move network to GPU for training
  torch::Device device(torch::kCUDA);
  network->to(device);
  network->train();

  std::vector<torch::Tensor> states;
  std::vector<torch::Tensor> policies;
  std::vector<torch::Tensor> values;
  states.reserve(examples.size());
  policies.reserve(examples.size());
  values.reserve(examples.size());

  // Convert examples to tensors directly on GPU
  for (const auto& example : examples) {
    states.push_back(example.board.to(device));
    policies.push_back(torch::from_blob((void*)example.policy.data(), 
                                          {1, (long)example.policy.size()}, 
                                          torch::kFloat).to(device));
    values.push_back(torch::tensor(example.value).to(device));
  }

  auto states_tensor = torch::stack(states);
  auto policies_tensor = torch::cat(policies);
  auto values_tensor = torch::stack(values);

  optimizer_->zero_grad();
  auto [policy_preds, value_preds] = network->forward(states_tensor);
  auto loss_policy = ComputePolicyLoss(policy_preds, policies_tensor);
  auto loss_value = ComputeValueLoss(value_preds, values_tensor);
  auto loss = loss_policy + loss_value;
  loss.backward();
  optimizer_->step();

  // After training, move back to CPU for self-play
  network->eval();
  network->to(torch::kCPU);
}

}  // namespace alphazero 