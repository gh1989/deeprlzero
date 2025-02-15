#include "core/network_manager.h"
#include <torch/serialize.h>
#include <iostream>
#include <fstream>

namespace alphazero {

NetworkManager::NetworkManager(const Config& config)
    : config_(config), 
      current_temperature_(config.initial_temperature) {}

bool NetworkManager::AcceptOrRejectNewNetwork(std::shared_ptr<NeuralNetwork> network, EvaluationStats evaluation_stats) {
    if (evaluation_stats.IsBetterThan(best_evaluation_stats_)) {
        best_network_ = network;
        best_evaluation_stats_ = evaluation_stats;
        best_iteration_ = current_iteration_;
        SaveBestNetwork();
        logger_.LogFormat("  ✓ New network accepted!");
        return true;
    } else {
        logger_.LogFormat("  × Network rejected");
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
    best_network_ = network;
}

std::shared_ptr<NeuralNetwork> NetworkManager::CreateInitialNetwork() {
    auto network = std::make_shared<NeuralNetwork>(
        1,  // input channels
        config_.num_filters,
        9,  // num_actions for TicTacToe
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

}  // namespace alphazero 