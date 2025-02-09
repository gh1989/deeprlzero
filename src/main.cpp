#include "core/neural_network.h"
#include "core/self_play.h"
#include "core/trainer.h"
#include "core/evaluator.h"
#include "core/config.h"
#include <iostream>
#include <fstream>
#include <omp.h>

int main(int argc, char** argv) {
    auto config = alphazero::Config::ParseCommandLine(argc, argv);
    
    // Initialize neural network
    auto network = std::make_shared<alphazero::NeuralNetwork>(
        1,  // input channels
        config.num_filters,
        9,  // num_actions for TicTacToe
        config.num_residual_blocks
    );
    
    // Try to load the best network if it exists
    std::string best_model_path = config.model_path + ".best";
    std::ifstream best_model_file(best_model_path);
    if (best_model_file.good()) {
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(best_model_path);
            network->load(archive);
            std::cout << "Loaded best model from: " << best_model_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading best model: " << e.what() << std::endl;
            std::cerr << "Starting with fresh network" << std::endl;
        }
    } else {
        std::cout << "No existing best model found. Starting with fresh network." << std::endl;
    }
    
    // After loading or creating the network
    network->to(torch::kCPU);
    network->eval();

    // Initialize self-play with CPU network
    float current_temperature = config.temperature;
    alphazero::SelfPlay self_play(network, 
                                 config.num_simulations, 
                                 config.c_puct, 
                                 current_temperature);
    
    // Initialize trainer
    alphazero::Trainer trainer(config.batch_size, 
                             config.num_epochs, 
                             config.learning_rate);
    
    std::shared_ptr<alphazero::NeuralNetwork> best_network = network;  // Start with loaded/initial network
    float best_win_rate = 0.0f;
    int best_iteration = 0;
    
    for (int iter = 0; iter < config.num_iterations; ++iter) {
        std::cout << "Starting iteration " << iter + 1 << "/" << config.num_iterations << std::endl;
        
        // Set OpenMP parameters based on config
        omp_set_num_threads(config.num_threads);
        
        // Generate self-play games in parallel
        std::vector<alphazero::GameExample> examples;
        std::cout << "\nGenerating " << config.episodes_per_iteration 
                 << " self-play games using " << config.num_threads << " threads..." << std::endl;
        
        #pragma omp parallel for schedule(dynamic, 1) proc_bind(spread)
        for (int episode = 0; episode < config.episodes_per_iteration; ++episode) {
            if (episode % 10 == 0) {
                #pragma omp critical
                {
                    std::cout << "Game " << episode << "/" << config.episodes_per_iteration << "\r" << std::flush;
                }
            }
            auto episode_examples = self_play.ExecuteEpisode();
            #pragma omp critical
            {
                examples.insert(examples.end(), episode_examples.begin(), episode_examples.end());
            }
        }
        std::cout << "\nCompleted self-play games generation." << std::endl;
        
        // Train network
        trainer.Train(network, examples);
        
        // Evaluate against best network
        alphazero::Evaluator evaluator(network, config.c_puct, config.num_simulations);
        float win_rate = evaluator.EvaluateAgainstNetwork(best_network);
        
        std::cout << "\nIteration " << (iter + 1) << " Summary:"
                  << "\n  Win Rate vs Best: " << (win_rate * 100) << "%"
                  << "\n  Current Best from Iteration: " << best_iteration + 1
                  << "\n  Best Historical Win Rate vs Best: " << (best_win_rate * 100) << "%";
        
        // Accept if win rate exceeds threshold
        float acceptance_threshold = 0.52f;  // Lower from 0.55f
        if (win_rate > acceptance_threshold) {
            std::cout << "\n  ✓ New network accepted!" << std::endl;
            best_network = network;
            best_iteration = iter;
            
            // Update best win rate if current win rate is better
            if (win_rate > best_win_rate) {
                best_win_rate = win_rate;
                
                // Save best model
                torch::serialize::OutputArchive archive;
                network->save(archive);
                archive.save_to(config.model_path + ".best");
            }
        } else {
            std::cout << "\n  × Network rejected" << std::endl;
            // Optionally revert network to best
            network = best_network;
        }
        
        current_temperature = std::max(0.1f, current_temperature * 0.95f);  // Decay temperature
        
        std::cout << "Completed iteration " << iter + 1 << std::endl;
    }
    
    // Save final model
    torch::serialize::OutputArchive final_archive;
    network->save(final_archive);
    final_archive.save_to(config.model_path);
    
    // Final evaluation
    alphazero::Evaluator evaluator(network, 
                                  config.c_puct, 
                                  config.num_simulations * 4);
    float final_win_rate_best = evaluator.EvaluateAgainstNetwork(best_network);
    float final_win_rate_random = evaluator.EvaluateAgainstRandom();
    
    std::cout << "\nFinal Evaluation Results:"
              << "\n  Win rate vs best: " << final_win_rate_best * 100 << "%"
              << "\n  Win rate vs random: " << final_win_rate_random * 100 << "%" << std::endl;
    
    return 0;
} 