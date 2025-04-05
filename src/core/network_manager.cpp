#include "core/network_manager.h"
#include <torch/serialize.h>
#include <iostream>
#include <fstream>

namespace alphazero {

NetworkManager::NetworkManager(const Config& config)
    : config_(config), 
      current_temperature_(config.initial_temperature)
      {
        std::cout << "NetworkManager initialized with temperature: " << current_temperature_ << std::endl;
      }

bool NetworkManager::AcceptOrRejectNewNetwork(
    std::shared_ptr<NeuralNetwork> network,
    EvaluationStats evaluation_stats) {
  
  // For first-time acceptance (no prior best model)
  if (!best_network_) {
    std::cout << "No existing best model found. Accepting initial network." << std::endl;
    best_network_ = network;
    best_evaluation_stats_ = evaluation_stats;
    best_iteration_ = current_iteration_;
    SaveBestNetwork();
    return true;
  }

  // Calculate score using standard scoring: win=1, draw=0.5, loss=0
  float score = evaluation_stats.win_rate + (0.5f * evaluation_stats.draw_rate);
  
  // Accept if score exceeds threshold
  if (score > config_.acceptance_threshold) {
    std::cout << "\nNetwork ACCEPTED: Score (" << score << ") > Threshold (" 
              << config_.acceptance_threshold << ")" << std::endl;
    best_network_ = network;
    best_evaluation_stats_ = evaluation_stats;
    best_iteration_ = current_iteration_;
    SaveBestNetwork();
    return true;
  } else {
    std::cout << "\nNetwork REJECTED: Score (" << score << ") <= Threshold (" 
              << config_.acceptance_threshold << ")" << std::endl;
    return false;
  }
}

void NetworkManager::UpdateTemperature() {
    current_temperature_ = std::max(
        config_.min_temperature,
        current_temperature_ * config_.temperature_decay
    );
}

void NetworkManager::SaveBestNetwork() const {
    torch::serialize::OutputArchive archive;
    best_network_->save(archive);
    archive.save_to(config_.model_path + ".best");
}

void NetworkManager::SetBestNetwork(std::shared_ptr<NeuralNetwork> network) {
    // Get the device of the source network
    auto device = network->parameters().begin()->device();
    // Create deep copy and cast back to NeuralNetwork
    best_network_ = std::dynamic_pointer_cast<NeuralNetwork>(network->clone(device));
    
    if (!best_network_) {
        throw std::runtime_error("Failed to create deep copy of network in SetBestNetwork");
    }
}

std::shared_ptr<NeuralNetwork> NetworkManager::CreateInitialNetwork() {
   
    auto network = std::make_shared<NeuralNetwork>(
        3,  // input channels
        config_.num_filters,
        config_.action_size,
        config_.num_residual_blocks
    );
    return network;
}

bool NetworkManager::LoadBestNetwork() {
    try {
        // Check if the model file exists
        std::ifstream file_check(config_.model_path);
        if (!file_check.good()) {
            std::cout << "Model file not found at: " << config_.model_path << std::endl;
            return false;
        }
        file_check.close();
        
        // Create a new network instance
        auto network = std::make_shared<NeuralNetwork>(
            config_.action_size,
            config_.num_filters,
            config_.num_residual_blocks,
            config_.dropout_rate
        );
        
        // Load the model from disk
        torch::serialize::InputArchive archive;
        archive.load_from(config_.model_path);
        network->load(archive);  // Load into the new network, not best_network_
        
        // Set as best network AFTER loading
        best_network_ = network;
        
        std::cout << "Successfully loaded model from: " << config_.model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::shared_ptr<NeuralNetwork> NetworkManager::GetBestNetwork() {
    // Create a fresh clone on the desired device and ensure all tensors are there
    auto device = torch::Device(torch::kCPU);
    auto cloned_network = std::dynamic_pointer_cast<NeuralNetwork>(best_network_->clone(device));
    
    // Explicitly ensure all tensors are on the same device
    cloned_network->to(device);
    
    return cloned_network;
}

}  // namespace alphazero   
  // Keep these methods for file path-based serialization
  void save(const std::string& path);
  void load(const std::string& path);