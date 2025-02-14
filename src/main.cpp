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
    alphazero::Logger logger(config);
    
    // Initialize network manager and create network
    alphazero::NetworkManager network_manager(config);
    
    // Check CUDA availability before proceeding
    if (!torch::cuda::is_available()) {
        logger.Log("ERROR: CUDA is not available. This program requires a CUDA-capable GPU for training.");
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
        logger.Log("Starting iteration " + std::to_string(iter + 1) + "/" + std::to_string(config.num_iterations));
        
        // Set OpenMP parameters
        omp_set_num_threads(config.num_threads);
        
        // Generate self-play games in parallel
        std::vector<alphazero::GameExample> examples;
        logger.Log("Generating " + std::to_string(config.episodes_per_iteration) + " self-play games using " + std::to_string(config.num_threads) + " threads...");
        
        #pragma omp parallel for schedule(dynamic, 1) proc_bind(spread)
        for (int episode = 0; episode < config.episodes_per_iteration; ++episode) {
            if (episode % 10 == 0) {
                #pragma omp critical
                {
                    logger.Log("[THREAD]" + std::to_string(omp_get_thread_num()) + " Game " +
                        std::to_string(episode) + "/" + std::to_string(config.episodes_per_iteration) + "\r");
                }
            }
            auto episode_examples = self_play.ExecuteEpisode();
            #pragma omp critical
            {
                examples.insert(examples.end(), episode_examples.begin(), episode_examples.end());
            }
        }
        logger.Log("Completed self-play games generation.");
        
        // Training phase (on GPU)
        trainer.Train(network, examples);
        
        // Evaluation phase (on CPU)
        network->to(torch::kCPU);
        network->eval();
        alphazero::Evaluator evaluator(network, config);
        float win_rate = evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
        
        logger.Log("Iteration " + std::to_string(iter + 1) + " Summary:"
                  + "\n  Win Rate vs Best: " + std::to_string(win_rate * 100) + "%"
                  + "\n  Temperature: " + std::to_string(network_manager.GetCurrentTemperature()));
        
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
    
    logger.Log("Final Evaluation Results:\n  Win rate vs best: " + 
               std::to_string(final_win_rate_best * 100) + "%\n  Win rate vs random: " + 
               std::to_string(final_win_rate_random * 100) + "%");
    
    return 0;
} 