#include "core/trainer.h"
#include <torch/torch.h>
#include <random>

namespace alphazero {

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
    
    torch::optim::Adam optimizer(
        network->parameters(),
        torch::optim::AdamOptions(config_.learning_rate)
            .weight_decay(config_.l2_reg)
    );

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

    // After training, move back to CPU for self-play
    network->eval();
    network->to(torch::kCPU);
}

}  // namespace alphazero 