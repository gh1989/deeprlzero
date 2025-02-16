#include "core/trainer.h"
#include <torch/torch.h>
#include <random>

namespace alphazero {

torch::Tensor Trainer::ComputePolicyLoss(const torch::Tensor& pred_policy, 
                                       const torch::Tensor& target_policy) {
    // Cross entropy loss for policy
    return -torch::mean(torch::sum(target_policy * torch::log_softmax(pred_policy, 1), 1));
}

torch::Tensor Trainer::ComputeValueLoss(const torch::Tensor& pred_value, 
                                      const torch::Tensor& target_value) {
    // MSE loss for value
    return torch::mean(torch::pow(pred_value - target_value, 2));
}

void Trainer::Train(std::shared_ptr<NeuralNetwork> network,
                    const std::vector<GameEpisode>& episodes) {
  if (episodes.empty()) {
    throw std::runtime_error("No episodes to train on");
  }

  if (!torch::cuda::is_available()) {
    throw std::runtime_error("CUDA is required for training");
  }

  // Move the network to GPU for training.
  torch::Device device(torch::kCUDA);
  network->to(device);
  network->train();

  std::vector<torch::Tensor> states;
  std::vector<torch::Tensor> policies;
  std::vector<torch::Tensor> values;

  // Iterate over each full game and extract per-move training samples.
  for (const auto& episode : episodes) {
    // Optionally, you can check that the sizes of boards, policies, and values match.
    if (episode.boards.size() != episode.policies.size() ||
        episode.boards.size() != episode.values.size()) {
      throw std::runtime_error("Mismatch in boards, policies, and values sizes in a GameEpisode");
    }
    for (size_t i = 0; i < episode.boards.size(); ++i) {
      // The board tensor is moved to the appropriate device.
      states.push_back(episode.boards[i].to(device));
      
      // Convert the policy vector into a tensor.
      // 'torch::from_blob' does not own the memory so we clone the tensor.
      auto policy_tensor = torch::from_blob(
          const_cast<float*>(episode.policies[i].data()),
          {1, static_cast<long>(episode.policies[i].size())},
          torch::kFloat)
                                .clone()
                                .to(device);
      policies.push_back(policy_tensor);

      // Use the provided per-move target value. If you intended to use the final outcome for all moves,
      // you could replace 'episode.values[i]' with 'episode.outcome'.
      values.push_back(torch::tensor(
          episode.values[i],
          torch::TensorOptions().dtype(torch::kFloat)).to(device));
    }
  }

  // Aggregate all move-level samples into batch tensors.
  auto states_tensor = torch::stack(states);
  auto policies_tensor = torch::cat(policies);
  auto values_tensor = torch::stack(values);

  optimizer_->zero_grad();

  // Forward pass through the network.
  auto outputs = network->forward(states_tensor);
  // Assuming network->forward returns a tuple: (policy_preds, value_preds)
  auto policy_preds = std::get<0>(outputs);
  auto value_preds = std::get<1>(outputs);

  auto loss_policy = ComputePolicyLoss(policy_preds, policies_tensor);
  auto loss_value = ComputeValueLoss(value_preds, values_tensor);
  auto loss = loss_policy + loss_value;
  loss.backward();
  optimizer_->step();

  // After training, revert network back to CPU for self-play.
  network->eval();
  network->to(torch::kCPU);
}

}  // namespace alphazero 