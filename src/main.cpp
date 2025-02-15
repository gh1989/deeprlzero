#include "core/neural_network.h"
#include "core/network_manager.h"
#include "core/self_play.h"
#include "core/trainer.h"
#include "core/evaluator.h"
#include "core/config.h"
#include "core/logger.h"
#include "core/thread.h"
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

    // Create an initial network then load the best if possible.    
    auto network = network_manager.CreateInitialNetwork();
    network_manager.SetBestNetwork(network);
    network_manager.LoadBestNetwork();
    
    // After loading or creating the network
    network->to(torch::kCPU);  // Start on CPU for self-play
    network->eval();

    // Initialize self-play and trainer
    alphazero::SelfPlay self_play(network, config);
    alphazero::Trainer trainer(network, config);

  // Aggregated statistics from all episodes.
  alphazero::MCTSStats aggregated_stats;

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
        


  alphazero::ParallelFor(config.episodes_per_iteration, [&](int episode) {
    // Get a thread-local SelfPlay instance using the factory function.
    auto &local_self_play = alphazero::GetThreadLocalInstance<alphazero::SelfPlay>([&]() {
      return new alphazero::SelfPlay(network, config);
    });

    // Execute a self-play episode.
    auto episode_examples = local_self_play.ExecuteEpisode();

    // In the critical section, merge the thread's stats into the aggregator.
    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
      examples.insert(examples.end(),
                      episode_examples.begin(),
                      episode_examples.end());
      // Get a local copy of the current stats from the self-play instance.
      alphazero::MCTSStats local_stats = local_self_play.GetStats();
      // Merge the local stats into the aggregated stats.
      aggregated_stats.MergeStats(local_stats);
      // Clear the local stats so that subsequent episodes start fresh.
      local_self_play.ClearStats();
    }
  });

        if (auto result = logger.Log("Completed self-play games generation."); !result) {
            std::cerr << "Failed to log completion" << std::endl;
        }
        
        // After self-play games are complete, log the MCTS statistics
        if (auto log_result = logger.LogFormat("\nMCTS Statistics for Iteration {}:", iter + 1); !log_result) {
            std::cerr << "Failed to log MCTS stats header" << std::endl;
        }
        self_play.ClearStats();
        
        // Training phase (on GPU)
        trainer.Train(network, examples);
        
        // Evaluation phase (on CPU)
        network->to(torch::kCPU);
        network->eval();
        alphazero::Evaluator evaluator(network, config);
        alphazero::EvaluationStats win_rate = evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
        
        if (auto result = logger.LogFormat("Iteration {} Summary:\n  Win Rate vs Best: {}\n  Temperature: {}", 
            iter + 1, win_rate.WinStats(), network_manager.GetCurrentTemperature()); !result) {
            std::cerr << "Failed to log iteration summary" << std::endl;
        }
        
        // Accept or reject new network
        if (!network_manager.AcceptOrRejectNewNetwork(network, win_rate)) {
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
    alphazero::EvaluationStats final_win_rate_best = final_evaluator.EvaluateAgainstNetwork(network_manager.GetBestNetwork());
    alphazero::EvaluationStats final_win_rate_random = final_evaluator.EvaluateAgainstRandom();
    
    if (auto result = logger.LogFormat("Final Evaluation Results:\n  Win rate vs best: {}%\n  Win rate vs random: {}%",
        final_win_rate_best.WinStats(), final_win_rate_random.WinStats()); !result) {
        std::cerr << "Failed to log final results" << std::endl;
    }
    
    aggregated_stats.LogStatistics();

    return 0;
} 