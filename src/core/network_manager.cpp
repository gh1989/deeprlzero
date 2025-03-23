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
        1,  // input channels
        config_.num_filters,
        config_.action_size,
        config_.num_residual_blocks
    );
    return network;
}

bool NetworkManager::LoadBestNetwork() {
    std::string best_model_path = config_.model_path + ".best";
    std::ifstream best_model_file(best_model_path);
    
    if (!best_model_file.good()) {
        std::cout << "No existing best model found. Starting with fresh network." << std::endl;
        return false;
    }
    
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(best_model_path);
        best_network_->load(archive);
        std::cout << "Loaded best model from: " << best_model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading best model: " << e.what() << std::endl;
        std::cerr << "Starting with fresh network" << std::endl;
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