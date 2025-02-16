#include "core/network_manager.h"
#include <torch/serialize.h>
#include <iostream>
#include <fstream>

namespace alphazero {

NetworkManager::NetworkManager(const Config& config)
    : config_(config), 
      current_temperature_(config.initial_temperature)
      {
      }

bool NetworkManager::AcceptOrRejectNewNetwork(
    std::shared_ptr<NeuralNetwork> network,
    EvaluationStats evaluation_stats) {
  // Calculate the sum over all evaluation outcomes.
  int sum = evaluation_stats.win_rate + evaluation_stats.draw_rate +
            evaluation_stats.loss_rate;

  // Reject the network if its loss rate is too high.
  // loss_threshold is now provided by the config.
  if (evaluation_stats.loss_rate > config_.loss_threshold) {
    logger_.LogFormat("  × Network rejected due to high loss rate: {}",
                      evaluation_stats.loss_rate);
    return false;
  }

  // For first-time acceptance, when no best model exists yet.
  if ((best_evaluation_stats_.win_rate == 0) &&
      (best_evaluation_stats_.draw_rate == 0) &&
      (best_evaluation_stats_.loss_rate == 0)) {
    std::cout << "No existing best model found. Starting with fresh network."
              << std::endl;
    best_network_ = network;
    best_evaluation_stats_ = evaluation_stats;
    best_iteration_ = current_iteration_;
    SaveBestNetwork();
    logger_.LogFormat("  ✓ New network accepted (initial acceptance)!");
    return true;
  }

  float new_performance = (evaluation_stats.win_rate + evaluation_stats.draw_rate);
  if (new_performance >= config_.acceptance_threshold && 
        evaluation_stats.loss_rate <= config_.loss_threshold) {
    best_network_ = network;
    best_evaluation_stats_ = evaluation_stats;
    best_iteration_ = current_iteration_;
    SaveBestNetwork();
    logger_.LogFormat("  ✓ New network accepted (meets config threshold)!");
    return true;
  } else {
    logger_.LogFormat("  × Network rejected (performance below config threshold).");
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