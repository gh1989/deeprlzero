#include "core/neural_network.h"
#include "core/network_manager.h"
#include "core/self_play.h"
#include "core/trainer.h"
#include "core/evaluator.h"
#include "core/config.h"
#include "core/logger.h"
#include "core/thread.h"
#include "core/mcts_stats.h"
#include "core/tictactoe.h"
#include "core/logger.h"
#include <iostream>
#include <fstream>
#include <omp.h>

int main(int argc, char** argv) {
    auto config = alphazero::Config::ParseCommandLine(argc, argv);
    auto& logger = alphazero::Logger::GetInstance(config);
    alphazero::NetworkManager network_manager(config);
    
    if (!torch::cuda::is_available()) {
        if (auto result = logger.Log("ERROR: CUDA is not available. This program requires a CUDA-capable GPU for training."); 
            !result) {
            std::cerr << "Failed to log error message" << std::endl;
            return 1;
        }
        return 1;
    }

/*
    if(!network_manager.LoadBestNetwork()) {
        auto network = network_manager.CreateInitialNetwork();
        network_manager.SetBestNetwork(network);
    }
*/

    auto network = network_manager.CreateInitialNetwork();
    network_manager.SetBestNetwork(network);

    alphazero::MCTSStats aggregated_stats;

    for (int iter = 0; iter < config.num_iterations; ++iter) {
        auto current_best_network = network_manager.GetBestNetwork();
        auto network_to_train = std::dynamic_pointer_cast<alphazero::NeuralNetwork>(current_best_network->clone(torch::kCPU));

        // Create the necessary objects with proper template parameter
        alphazero::SelfPlay<alphazero::TicTacToe> self_play(
            network_manager.GetBestNetwork(), 
            config,
            network_manager.GetCurrentTemperature()
        );
        alphazero::Trainer trainer(network_to_train, config);
        alphazero::Evaluator evaluator(network_manager.GetBestNetwork(), config);

        if (auto log_result = logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations); !log_result) {
            std::cerr << "Failed to log iteration start" << std::endl;
        }

        //Generate the episodes with the best network.
        auto episodes = std::vector<alphazero::GameEpisode>();
        for (int i = 0; i < config.episodes_per_iteration; ++i) {
            episodes.push_back(self_play.ExecuteEpisode());
        }
        
        if (auto result = logger.Log("Completed self-play episodes generation."); !result) {
            std::cerr << "Failed to log completion" << std::endl;
        }
        //Train the clone of the best network with these episodes
        trainer.Train(episodes);
        //Now which network should be instantiated for the evaluator? and which one is the argument?
        alphazero::EvaluationStats evaluation_stats = evaluator.EvaluateAgainstNetwork(network_to_train);
        
        if (auto result = logger.LogFormat("Iteration {} Summary:\n  Win Rate vs Best: {}\n  Temperature: {}", 
            iter + 1, evaluation_stats.WinStats(), network_manager.GetCurrentTemperature()); !result) {
            std::cerr << "Failed to log iteration summary" << std::endl;
        }
                
        network_manager.AcceptOrRejectNewNetwork(network_to_train, evaluation_stats);
        network_manager.UpdateTemperature();
    }
    
    // Save final model
    torch::serialize::OutputArchive final_archive;
    network_manager.GetBestNetwork()->save(final_archive);
    //final_archive.save_to(config.model_path);
    
    // Final evaluation
    alphazero::Evaluator final_evaluator(network_manager.GetBestNetwork(), config);
    alphazero::EvaluationStats final_win_rate_best = final_evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
    alphazero::EvaluationStats final_win_rate_random = final_evaluator.EvaluateAgainstRandom();
    
    if (auto result = logger.LogFormat("Final Evaluation Results:\n  Win rate vs best: {}%\n  Win rate vs random: {}%",
        final_win_rate_best.WinStats(), final_win_rate_random.WinStats()); !result) {
        std::cerr << "Failed to log final results" << std::endl;
    }
    
    aggregated_stats.LogStatistics();
    return 0;
} 