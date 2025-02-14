#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    if (torch::cuda::is_available()) {
        auto cuda_device = torch::Device(torch::kCUDA);
        torch::Tensor t = torch::ones({2, 2}, cuda_device);
        std::cout << "Tensor on CUDA: " << t << std::endl;
    }
    
    return 0;
} 