#include "core/trainer.h"
#include <torch/torch.h>
#include <random>
#include <iostream>

namespace alphazero {

torch::Tensor Trainer::ComputePolicyLoss(const torch::Tensor& policy_preds,
                                         const torch::Tensor& policy_targets) {
    // Don't detach or clone tensors as it breaks the graph
    return torch::nn::functional::cross_entropy(policy_preds, policy_targets);
}

torch::Tensor Trainer::ComputeValueLoss(const torch::Tensor& value_preds,
                                        const torch::Tensor& value_targets) {
    // Don't detach or clone tensors
    return torch::mse_loss(value_preds, value_targets);
}

void Trainer::Train(const std::vector<GameEpisode>& episodes) {
    if (episodes.empty()) {
        throw std::runtime_error("No episodes to train on");
    }

    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for training");
    }   
    

    // Create optimizer after ensuring network is on correct device
    auto optimizer = torch::optim::Adam(
        network_->parameters(),
        torch::optim::AdamOptions(config_.learning_rate).weight_decay(config_.l2_reg)
    );
    
    // Standard device - explicitly use cuda:0 instead of just cuda
    torch::Device device(torch::kCUDA, 0);
    //std::cout << "Training on device: " << device << std::endl;
        
    network_->to(device);
    network_->train();

    //std::cout << "Learning rate: " << config_.learning_rate << std::endl;
    
    // Collect all data
    std::vector<torch::Tensor> all_boards;
    std::vector<float> all_policies;
    std::vector<float> all_values;

    for (const auto& episode : episodes) {
        all_boards.insert(all_boards.end(), episode.boards.begin(), episode.boards.end());
        for (const auto& policy : episode.policies) {
            all_policies.insert(all_policies.end(), policy.begin(), policy.end());
        }
        all_values.insert(all_values.end(), episode.values.begin(), episode.values.end());
    }

    // Convert to tensors - do not set requires_grad for input data
    auto states = torch::stack(torch::TensorList(all_boards)).to(device);
    auto policies = torch::tensor(all_policies).reshape({-1, config_.action_size}).to(device);
    auto values = torch::tensor(all_values).reshape({-1, 1}).to(device);

    /*
    std::cout << "Training data shapes:" << std::endl;
    std::cout << "States: " << states.sizes() << std::endl;
    std::cout << "Policies: " << policies.sizes() << std::endl;
    std::cout << "Values: " << values.sizes() << std::endl;
    */
    
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        // Save parameters before update for comparison
        std::vector<torch::Tensor> params_before;
        for (const auto& param : network_->parameters()) {
            params_before.push_back(param.clone().detach());
        }
        
        // Zero gradients at the start of each iteration
        optimizer.zero_grad();
        
        // Forward pass - explicitly use network's forward method
        auto outputs = network_->forward(states);
        auto policy_preds = outputs.first;
        auto value_preds = outputs.second;
        
        // Make sure all tensors are properly moved to the correct device
        policy_preds = policy_preds.to(device);
        value_preds = value_preds.to(device);
        
        // Compute loss - using torch's built-in losses for reliability
        auto loss_policy = -torch::mean(torch::sum(policies * torch::log_softmax(policy_preds, 1), 1));
        auto loss_value = torch::mse_loss(value_preds, values);
        auto total_loss = loss_policy + loss_value;
        
        // Print the device of the loss
        if (epoch == 0) {
            //std::cout << "Loss device: " << total_loss.device() << std::endl;
        }
        
        // Backward pass
        total_loss.backward();
        
        // Debug gradients after backward pass
        /*
        if (epoch == 0 || epoch == config_.num_epochs - 1) {
            std::cout << "\nGradient information (epoch " << epoch << "):" << std::endl;
            for (const auto& param : network_->parameters()) {
                if (param.grad().defined()) {
                    std::cout << "Grad norm: " << param.grad().norm().item<float>() 
                             << " - requires_grad: " << param.requires_grad() << std::endl;
                } else {
                    std::cout << "No gradient - requires_grad: " << param.requires_grad() << std::endl;
                }
            }
        }
        */
        
        // After backward pass, add this debug code:
        for (auto& param : network_->parameters()) {
            if (!param.grad().defined()) {
                // Create a zero gradient if none exists
                param.mutable_grad() = torch::zeros_like(param);
            }
        }
        
        // Perform parameter update
        optimizer.step();
        
        // Verify parameter updates
        auto params_after = network_->parameters();
        bool params_updated = false;
        for (size_t i = 0; i < params_after.size(); ++i) {
            if (!torch::allclose(params_after[i], params_before[i])) {
                params_updated = true;
                break;
            }
        }
        /*
        std::cout << "Epoch " << epoch + 1 << "/" << config_.num_epochs 
                  << " - Policy Loss: " << loss_policy.item<float>()
                  << " Value Loss: " << loss_value.item<float>()
                  << " Parameters updated: " << (params_updated ? "Yes" : "No") << std::endl;
                  */
    }

    // Add this at the end of the Train method
    {
        // Get parameter fingerprint to verify changes
        float param_sum = 0.0f;
        int param_count = 0;
        
        for (const auto& param : network_->parameters()) {
            // Use safer tensor operations instead of direct data_ptr access
            param_count += param.numel();
            // Sum absolute values using torch operations
            param_sum += param.abs().sum().item<float>();
        }
        
        std::cout << "\n=== NETWORK TRAINING COMPLETED ===" << std::endl;
        std::cout << "Parameter stats: sum=" << param_sum 
                  << ", avg=" << (param_count > 0 ? param_sum / param_count : 0.0f) 
                  << ", count=" << param_count << std::endl;
        std::cout << "================================\n" << std::endl;
    }
}

}  // namespace alphazero 