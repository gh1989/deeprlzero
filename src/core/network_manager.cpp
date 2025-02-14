#include "core/network_manager.h"
#include <torch/serialize.h>
#include <iostream>
#include <fstream>

namespace alphazero {

NetworkManager::NetworkManager(const Config& config)
    : config_(config), 
      current_temperature_(config.initial_temperature) {}

bool NetworkManager::AcceptNewNetwork(std::shared_ptr<NeuralNetwork> network, float win_rate) {
    if (win_rate > config_.acceptance_threshold) {
        std::cout << "\n  ✓ New network accepted!" << std::endl;
        best_network_ = network;
        current_iteration_++;
        
        if (win_rate > best_win_rate_) {
            best_win_rate_ = win_rate;
            best_iteration_ = current_iteration_;
            SaveBestNetwork();
        }
        return true;
    }
    
    std::cout << "\n  × Network rejected" << std::endl;
    return false;
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
    best_network_ = network;
}

std::shared_ptr<NeuralNetwork> NetworkManager::CreateInitialNetwork() {
    auto network = std::make_shared<NeuralNetwork>(
        1,  // input channels
        config_.num_filters,
        9,  // num_actions for TicTacToe
        config_.num_residual_blocks
    );
    best_network_ = network;
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

}  // namespace alphazero 