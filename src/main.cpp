#include "core/neural_network.h"
#include "core/network_manager.h"
#include "core/self_play.h"
#include "core/trainer.h"
#include "core/evaluator.h"
#include "core/config.h"
#include "core/logger.h"
#include <iostream>
#include <fstream>
#include <omp.h>

int main(int argc, char** argv) {
    auto config = alphazero::Config::ParseCommandLine(argc, argv);
    
    // Initialize logger
    auto& logger = alphazero::Logger::GetInstance(config);
    
    // Initialize network manager and create network
    alphazero::NetworkManager network_manager(config);
    
    // Check CUDA availability before proceeding
    if (!torch::cuda::is_available()) {
        if (auto result = logger.Log("ERROR: CUDA is not available. This program requires a CUDA-capable GPU for training."); 
            !result) {
            std::cerr << "Failed to log error message" << std::endl;
            return 1;
        }
        return 1;
    }
    
    auto network = network_manager.CreateInitialNetwork();
    
    // Try to load existing best network
    network_manager.LoadBestNetwork();
    
    // After loading or creating the network
    network->to(torch::kCPU);  // Start on CPU for self-play
    network->eval();

    // Initialize self-play and trainer
    alphazero::SelfPlay self_play(network, config);
    alphazero::Trainer trainer(config);
    
    for (int iter = 0; iter < config.num_iterations; ++iter) {
        if (auto log_result = logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations); !log_result) {
            std::cerr << "Failed to log iteration start" << std::endl;
        }
        
        // Set OpenMP parameters
        omp_set_num_threads(config.num_threads);
        
        // Generate self-play games in parallel
        std::vector<alphazero::GameExample> examples;
        if (auto result = logger.LogFormat("Generating {} self-play games using {} threads...", 
            config.episodes_per_iteration, config.num_threads); !result) {
            std::cerr << "Failed to log game generation start" << std::endl;
        }
        
        #pragma omp parallel for schedule(dynamic, 1) proc_bind(spread)
        for (int episode = 0; episode < config.episodes_per_iteration; ++episode) {
            if (episode % 10 == 0) {
                #pragma omp critical
                {
                    if (auto result = logger.LogFormat("[THREAD]{} Game {}/{}", 
                        omp_get_thread_num(), episode, config.episodes_per_iteration); !result) {
                        std::cerr << "Failed to log thread progress" << std::endl;
                    }
                }
            }
            auto episode_examples = self_play.ExecuteEpisode();
            #pragma omp critical
            {
                examples.insert(examples.end(), episode_examples.begin(), episode_examples.end());
            }
        }
        if (auto result = logger.Log("Completed self-play games generation."); !result) {
            std::cerr << "Failed to log completion" << std::endl;
        }
        
        // After self-play games are complete, log the MCTS statistics
        if (auto log_result = logger.LogFormat("\nMCTS Statistics for Iteration {}:", iter + 1); !log_result) {
            std::cerr << "Failed to log MCTS stats header" << std::endl;
        }
        self_play.GetStats().LogStatistics();
        self_play.ClearStats();
        
        // Training phase (on GPU)
        trainer.Train(network, examples);
        
        // Evaluation phase (on CPU)
        network->to(torch::kCPU);
        network->eval();
        alphazero::Evaluator evaluator(network, config);
        float win_rate = evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
        
        if (auto result = logger.LogFormat("Iteration {} Summary:\n  Win Rate vs Best: {}%\n  Temperature: {}", 
            iter + 1, win_rate * 100, network_manager.GetCurrentTemperature()); !result) {
            std::cerr << "Failed to log iteration summary" << std::endl;
        }
        
        // Accept or reject new network
        if (!network_manager.AcceptNewNetwork(network, win_rate)) {
            network = network_manager.GetBestNetwork();
        }
        
        // Update temperature for next iteration
        network_manager.UpdateTemperature();
    }
    
    // Save final model
    torch::serialize::OutputArchive final_archive;
    network->save(final_archive);
    final_archive.save_to(config.model_path);
    
    // Final evaluation
    network->to(torch::kCPU);
    network->eval();
    alphazero::Evaluator final_evaluator(network, config);
    float final_win_rate_best = final_evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
    float final_win_rate_random = final_evaluator.EvaluateAgainstRandom();
    
    if (auto result = logger.LogFormat("Final Evaluation Results:\n  Win rate vs best: {}%\n  Win rate vs random: {}%",
        final_win_rate_best * 100, final_win_rate_random * 100); !result) {
        std::cerr << "Failed to log final results" << std::endl;
    }
    
    return 0;
} 