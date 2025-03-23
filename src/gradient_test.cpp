#include "core/neural_network.h"
#include "core/config.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

using namespace alphazero;

// Helper function to print parameter names and gradient status
void print_param_gradients(std::shared_ptr<NeuralNetwork> network) {
    std::cout << "\n===== PARAMETER GRADIENT DETAILS =====\n";
    
    // Get all named parameters
    auto named_parameters = network->named_parameters();
    int total_params = 0;
    int params_with_grad = 0;
    
    for (const auto& pair : named_parameters) {
        const std::string& name = pair.key();
        const torch::Tensor& param = pair.value();
        total_params++;
        
        bool has_grad = param.grad().defined() && param.grad().norm().item<float>() > 0;
        if (has_grad) {
            params_with_grad++;
        }
        
        std::cout << std::setw(30) << name << " : " 
                 << (has_grad ? "HAS GRADIENT" : "NO GRADIENT")
                 << ", shape=" << param.sizes()
                 << ", norm=" << (has_grad ? std::to_string(param.grad().norm().item<float>()) : "N/A")
                 << std::endl;
    }
    
    std::cout << "\nSummary: " << params_with_grad << "/" << total_params 
              << " parameters have gradients (" 
              << (100.0 * params_with_grad / total_params) << "%)" << std::endl;
}

int main() {
    // Check CUDA availability
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    // Choose device
    torch::Device device = cuda_available ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << device << std::endl;
    
    // Create config with small network for testing
    Config config;
    config.board_size = 3;
    config.action_size = 9;
    config.num_residual_blocks = 1;
    config.num_filters = 32;
    
    // Create network
    auto network = std::make_shared<NeuralNetwork>(
        1,  // input channels
        config.num_filters,
        config.action_size,
        config.num_residual_blocks
    );
    network->to(device);
    network->train();
    
    // Create dummy input data (batch of 4 board states)
    auto input = torch::randn({4, 1, 3, 3}, torch::TensorOptions().device(device));
    
    // Create dummy target outputs
    auto target_policy = torch::zeros({4, 9}, torch::TensorOptions().device(device));
    for (int i = 0; i < 4; i++) target_policy[i][i % 9] = 1.0; // One-hot encoding
    
    auto target_value = torch::ones({4, 1}, torch::TensorOptions().device(device));
    
    // Run the gradient flow validation
    network->ValidateGradientFlow(input, target_policy, target_value);
    
    return 0;
} 