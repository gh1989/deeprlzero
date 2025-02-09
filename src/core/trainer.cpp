#include "core/trainer.h"
#include <torch/torch.h>
#include <random>

namespace alphazero {

Trainer::Trainer(int batch_size, int num_epochs, float learning_rate)
    : batch_size_(batch_size), num_epochs_(num_epochs), learning_rate_(learning_rate) {}

void Trainer::Train(std::shared_ptr<NeuralNetwork> network,
                   const std::vector<GameExample>& examples) {
    if (examples.empty()) return;

    // Save CPU copy of network for later
    auto cpu_network = network->clone();
    
    // Move network and data to GPU
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    network->to(device);
    network->train();
    
    // Create optimizer with configured learning rate
    torch::optim::Adam optimizer(network->parameters(), learning_rate_);
    
    // More efficient GPU data transfer
    torch::TensorOptions options = torch::TensorOptions()
        .device(device)
        .pinned_memory(true);  // Enable pinned memory
    
    // Fill vectors before stacking
    std::vector<torch::Tensor> states, policies, values;
    states.reserve(examples.size());
    policies.reserve(examples.size());
    values.reserve(examples.size());

    // Convert examples to tensors
    for (const auto& example : examples) {
        states.push_back(example.board.to(device));
        policies.push_back(torch::from_blob((void*)example.policy.data(), 
                                          {1, (long)example.policy.size()}, 
                                          torch::kFloat).to(device));
        values.push_back(torch::tensor(example.value).to(device));
    }

    // Now stack the filled tensors
    auto states_tensor = torch::stack(states);
    auto policies_tensor = torch::cat(policies);
    auto values_tensor = torch::stack(values);

    // Training loop
    for (int epoch = 0; epoch < num_epochs_; ++epoch) {
        // Create random indices for shuffling
        std::vector<size_t> indices(examples.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        float epoch_loss = 0.0f;
        int num_batches = 0;

        // Process mini-batches
        for (size_t i = 0; i < examples.size(); i += batch_size_) {
            size_t batch_end = std::min(i + batch_size_, examples.size());
            auto state_slice = states_tensor.slice(0, i, batch_end);
            auto policy_slice = policies_tensor.slice(0, i, batch_end);
            auto value_slice = values_tensor.slice(0, i, batch_end);

            // Forward pass
            optimizer.zero_grad();
            auto [policy_pred, value_pred] = network->forward(state_slice);

            // Calculate policy loss (cross entropy with log_softmax)
            auto policy_loss = torch::nll_loss(policy_pred, policy_slice.argmax(1));
            
            // Calculate value loss
            auto value_loss = torch::mse_loss(value_pred.squeeze(), value_slice);
            
            auto total_loss = policy_loss + 0.5f * value_loss;

            // Backward pass
            total_loss.backward();
            optimizer.step();

            epoch_loss += total_loss.item<float>();
            num_batches++;

            // After each batch, ensure GPU operations are complete
            if (device.is_cuda()) {
                torch::cuda::synchronize();
            }
        }

        // Print epoch statistics
        if (num_batches > 0) {
            float avg_loss = epoch_loss / num_batches;
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs_ 
                     << " - Average Loss: " << avg_loss << std::endl;
        }
    }

    // After training is complete, copy trained weights back to CPU
    network->to(torch::kCPU);
}

}  // namespace alphazero 