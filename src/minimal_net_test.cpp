#include <torch/torch.h>
#include <iostream>

// Residual block implementation
class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int channels) 
        : conv1(register_module("conv1", torch::nn::Conv2d(
              torch::nn::Conv2dOptions(channels, channels, 3).padding(1)))),
          bn1(register_module("bn1", torch::nn::BatchNorm2d(channels))),
          conv2(register_module("conv2", torch::nn::Conv2d(
              torch::nn::Conv2dOptions(channels, channels, 3).padding(1)))),
          bn2(register_module("bn2", torch::nn::BatchNorm2d(channels))) {
        // Ensure batch norm layers are in training mode
        bn1->train();
        bn2->train();
    }
    
    torch::Tensor forward(torch::Tensor x, bool test_operations) {
        auto residual = x;
        
        // First conv block
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);
        
        // Second conv block
        out = conv2->forward(out);
        out = bn2->forward(out);
        
        // Test operations that might break gradient flow
        if (test_operations) {
            std::cout << "  Before clone: requires_grad = " << out.requires_grad() << std::endl;
            out = out.clone();
            std::cout << "  After clone: requires_grad = " << out.requires_grad() << std::endl;
            
            std::cout << "  Before to_cpu: requires_grad = " << out.requires_grad() << std::endl;
            out = out.cpu();
            std::cout << "  After to_cpu: requires_grad = " << out.requires_grad() << std::endl;
            
            std::cout << "  Before to_device: requires_grad = " << out.requires_grad() << std::endl;
            out = out.to(x.device());
            std::cout << "  After to_device: requires_grad = " << out.requires_grad() << std::endl;
        }
        
        // Skip connection and ReLU
        out = out + residual;
        out = torch::relu(out);
        
        return out;
    }
    
private:
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
};

// Network with residual blocks
class ResidualNetwork : public torch::nn::Module {
public:
    ResidualNetwork(int input_channels = 1, int num_filters = 32, int num_blocks = 1) 
        : conv(register_module("conv", torch::nn::Conv2d(
              torch::nn::Conv2dOptions(input_channels, num_filters, 3).padding(1)))),
          bn(register_module("bn", torch::nn::BatchNorm2d(num_filters))),
          policy_head(register_module("policy_head", torch::nn::Linear(num_filters*9, 9))),
          value_head1(register_module("value_head1", torch::nn::Linear(num_filters*9, 32))),
          value_head2(register_module("value_head2", torch::nn::Linear(32, 1))) {
        
        // Create and register residual blocks
        for (int i = 0; i < num_blocks; i++) {
            // Name format: res_block{i}
            std::string block_name = "res_block" + std::to_string(i);
            auto block = register_module(block_name, std::make_shared<ResidualBlock>(num_filters));
            res_blocks.push_back(block);
        }
        
        // Ensure batch norm is in training mode
        bn->train();
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, bool test_operations) {
        // Initial convolution
        x = conv->forward(x);
        x = bn->forward(x);
        x = torch::relu(x);
        
        // Test operations after initial conv
        if (test_operations) {
            std::cout << "After initial conv:" << std::endl;
            std::cout << "  Before clone: requires_grad = " << x.requires_grad() << std::endl;
            x = x.clone();
            std::cout << "  After clone: requires_grad = " << x.requires_grad() << std::endl;
            
            std::cout << "  Before to_cpu: requires_grad = " << x.requires_grad() << std::endl;
            x = x.cpu();
            std::cout << "  After to_cpu: requires_grad = " << x.requires_grad() << std::endl;
            
            std::cout << "  Before to_device: requires_grad = " << x.requires_grad() << std::endl;
            x = x.to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
            std::cout << "  After to_device: requires_grad = " << x.requires_grad() << std::endl;
        }
        
        // Residual blocks
        for (auto& block : res_blocks) {
            x = block->forward(x, test_operations);
        }
        
        // Flatten for fully connected layers
        auto flattened = x.reshape({x.size(0), -1});
        
        // Policy head
        auto policy = policy_head->forward(flattened);
        
        // Value head
        auto value = torch::relu(value_head1->forward(flattened));
        value = torch::tanh(value_head2->forward(value));
        
        return {policy, value};
    }
    
private:
    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
    torch::nn::Linear policy_head, value_head1, value_head2;
    std::vector<std::shared_ptr<ResidualBlock>> res_blocks;
};

void print_gradients(torch::nn::Module& module) {
    int total = 0, with_grad = 0;
    for (const auto& pair : module.named_parameters()) {
        const std::string& name = pair.key();
        const torch::Tensor& param = pair.value();
        total++;
        
        bool has_grad = param.grad().defined() && param.grad().norm().item<float>() > 0;
        if (has_grad) with_grad++;
        
        std::cout << name << ": " << (has_grad ? "HAS GRADIENT" : "NO GRADIENT");
        if (has_grad) std::cout << ", norm = " << param.grad().norm().item<float>();
        std::cout << std::endl;
    }
    
    std::cout << "\nSummary: " << with_grad << "/" << total 
              << " parameters have gradients (" 
              << (100.0 * with_grad / total) << "%)" << std::endl;
}

int main() {
    // Check CUDA availability
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    std::cout << "Using device: " << device << std::endl;
    
    // Run two tests: one without operations, one with operations
    for (int test = 0; test < 2; test++) {
        bool test_operations = (test == 1);
        std::cout << "\n========================================\n";
        std::cout << "TEST " << (test_operations ? "WITH" : "WITHOUT") << " TENSOR OPERATIONS\n";
        std::cout << "========================================\n";
        
        // Create network with residual blocks
        ResidualNetwork network(1, 32, 1);  // 1 input channel, 32 filters, 1 residual block
        network.to(device);
        network.train();
        
        // Create dummy input
        auto input = torch::ones({4, 1, 3, 3}, device);
        
        // Create dummy targets
        auto target_policy = torch::zeros({4, 9}, device);
        for (int i = 0; i < 4; i++) target_policy[i][i % 9] = 1.0;
        auto target_value = torch::ones({4, 1}, device);
        
        // Create optimizer
        torch::optim::Adam optimizer(network.parameters(), 0.001);
        
        // One training step
        optimizer.zero_grad();
        
        // Forward pass
        auto [policy, value] = network.forward(input, test_operations);
        
        // Compute loss function
        auto policy_loss = -torch::mean(torch::sum(target_policy * torch::log_softmax(policy, 1), 1));
        auto value_loss = torch::mse_loss(value, target_value);
        auto total_loss = policy_loss + value_loss;
        
        std::cout << "\nForward results:" << std::endl;
        std::cout << "Policy loss: " << policy_loss.item<float>() << std::endl;
        std::cout << "Value loss: " << value_loss.item<float>() << std::endl;
        std::cout << "Total loss: " << total_loss.item<float>() << std::endl;
        
        // Backward pass
        total_loss.backward();
        
        // Print which parameters have gradients
        std::cout << "\nGradient information:" << std::endl;
        print_gradients(network);
        
        // Update parameters
        optimizer.step();
    }
    
    return 0;
} 