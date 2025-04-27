#include "core/network.h"
#include "core/trainer.h"
#include "core/config.h"
#include "core/logger.h"
#include "core/thread.h"
#include "core/game.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <format>
#include <filesystem>

namespace deeprlzero {

float CalculatePolicyEntropy(const std::vector<float>& policy) {
  float entropy = 0.0f;
  for (float p : policy) {
    // Handle zero probabilities (log(0) is undefined)
    if (p > 1e-10) {
      entropy -= p * std::log(p);
    }
  }
  return entropy;
}

float CalculateAverageExplorationMetric(const std::vector<GameEpisode>& episodes) {
  if (episodes.empty()) {
    return 0.0f;
  }
  
  float total_entropy = 0.0f;
  int total_moves = 0;
  
  for (const auto& episode : episodes) {
    for (const auto& policy : episode.policies) {
      total_entropy += CalculatePolicyEntropy(policy);
      total_moves++;
    }
  }
  
  return total_moves > 0 ? total_entropy / total_moves : 0.0f;
}

} 

int main(int argc, char** argv) {
    auto config = deeprlzero::Config::ParseCommandLine(argc, argv);
    auto& logger = deeprlzero::Logger::GetInstance(config);
    
    if (!torch::cuda::is_available()) {
        logger.Log("ERROR: CUDA is not available. This program requires a CUDA-capable GPU for training.");
        return 1;
    }

    std::shared_ptr<deeprlzero::NeuralNetwork> best_network = nullptr;
    float current_temperature = config.initial_temperature;
    int iterations_since_improvement = 0;

    auto createInitialNetwork = [&config]() -> std::shared_ptr<deeprlzero::NeuralNetwork> {
        try {
            return std::make_shared<deeprlzero::NeuralNetwork>(config);
        } catch (const std::exception& e) {
            std::cerr << "Error creating network: " << e.what() << std::endl;
            return nullptr;
        }
    };

    auto loadBestNetwork = [&config, &logger]() -> std::shared_ptr<deeprlzero::NeuralNetwork> {
        try {
            if (std::filesystem::exists(config.model_path)) {
                logger.LogFormat("Loading model from: {}", config.model_path);
                auto network = std::make_shared<deeprlzero::NeuralNetwork>(config);
                torch::load(network, config.model_path);
                logger.Log("Model loaded successfully");
                return network;
            }
        } catch (const std::exception& e) {
            logger.LogFormat("Error loading model: {}", e.what());
        }
        return nullptr;
    };
    
    auto saveBestNetwork = [&config, &logger](std::shared_ptr<deeprlzero::NeuralNetwork> network) {
        try {
            logger.LogFormat("Saving model to: {}", config.model_path);
            torch::save(network, config.model_path);
            logger.Log("Model saved successfully");
        } catch (const std::exception& e) {
            logger.LogFormat("Error saving model: {}", e.what());
        }
    };

    auto updateTemperature = [&current_temperature, &config]() {
        current_temperature = std::max(
            config.min_temperature, 
            current_temperature * config.temperature_decay
        );
        return current_temperature;
    };

    auto acceptOrRejectNewNetwork = [&](
        std::shared_ptr<deeprlzero::NeuralNetwork> candidate_network,
        const deeprlzero::EvaluationStats& stats) -> bool {
        
        float win_loss_ratio = (stats.win_rate + 0.5f * stats.draw_rate) / 
                              (stats.loss_rate + 0.5f * stats.draw_rate);
                              
        if (win_loss_ratio >= config.acceptance_threshold) {
            // Accept the new network
            best_network = candidate_network;
            saveBestNetwork(best_network);
            iterations_since_improvement = 0;
            return true;
        } else {
            // Reject the new network
            iterations_since_improvement++;
            return false;
        }
    };
    
    // Initialize best network
    best_network = loadBestNetwork();
    if (!best_network) {
        logger.Log("No existing model found. Creating a new one.");
        best_network = createInitialNetwork();
        saveBestNetwork(best_network);
    }

    for (int iter = 0; iter < config.num_iterations; ++iter) {
        auto network_to_train = std::dynamic_pointer_cast<deeprlzero::NeuralNetwork>(
            best_network->clone(torch::kCPU));

        deeprlzero::SelfPlay<deeprlzero::TicTacToe> self_play(
            best_network, 
            config,
            current_temperature
        );
        deeprlzero::Trainer trainer(network_to_train, config);
        deeprlzero::Evaluator evaluator(best_network, config);

        logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations);
        std::vector<deeprlzero::GameEpisode> episodes;
        if (config.num_threads > 1) {
            episodes = self_play.ExecuteEpisodesParallel();
        } else {
            episodes.reserve(config.episodes_per_iteration);
            for (int i = 0; i < config.episodes_per_iteration; ++i) {
                episodes.push_back(self_play.ExecuteEpisode());
            }
        }
        
        logger.Log("Completed self-play episodes generation.");

        float exploration_metric = deeprlzero::CalculateAverageExplorationMetric(episodes);
        logger.LogFormat("Exploration metric: {:.4f}", exploration_metric);
        trainer.Train(episodes);
        deeprlzero::EvaluationStats evaluation_stats = 
            evaluator.EvaluateAgainstNetwork(network_to_train);
        bool network_accepted = acceptOrRejectNewNetwork(network_to_train, evaluation_stats);
        
        logger.LogFormat("Network acceptance decision: {}", 
                       network_accepted ? "ACCEPTED" : "REJECTED");
        updateTemperature();
        logger.LogFormat("Temperature updated to: {:.4f}", current_temperature);
        if (iter % 5 == 0) {
            deeprlzero::EvaluationStats random_eval_stats = 
                evaluator.EvaluateAgainstRandom();
            
            logger.LogFormat("Performance vs Random: Win: {:.1f}%, Draw: {:.1f}%, Loss: {:.1f}%",
                           random_eval_stats.win_rate * 100,
                           random_eval_stats.draw_rate * 100,
                           random_eval_stats.loss_rate * 100);
        }
    }
    
    logger.Log("Training complete!");
    return 0;
} 