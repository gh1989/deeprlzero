#include "network.h"
#include "trainer.h"
#include "config.h"
#include "logger.h"
#include "thread.h"
#include "games/chess.h"
#include "selfplay.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <format>

using namespace deeprlzero;
using Game = Chess;

// A run using the tic tac toe
int main(int argc, char** argv) {
  auto config = Config::ParseCommandLine(argc, argv);
  auto& logger = Logger::GetInstance(config);

  if (!torch::cuda::is_available()) {
    logger.Log("ERROR: CUDA is not available."); // dont even want to run if this is the case.
    return 1;
  }

  // Track training metrics across iterations
  float last_policy_loss = 0.0f;
  float last_value_loss = 0.0f;
  float last_total_loss = 0.0f;
  float last_param_variance = 0.0f;
  int iterations_since_improvement = 0;

  /// get the best network from the file or create a new one and just save that (static operations pass in the config.)
  std::shared_ptr<NeuralNetwork> best_network = CreateNetwork<Game>(config); // config twice... here and below. not good
  best_network = LoadBestNetwork(best_network, config); // the network arch depends on the game now... we find out if this design choice is going to cause issues...
  if (!best_network) {
      logger.Log("No existing model found. Creating a new one.");
      best_network = CreateNetwork<Game>(config);
      SaveBestNetwork(best_network, config);
  }
    
  auto optimizer = std::make_shared<torch::optim::Adam>(best_network->parameters(), config.learning_rate);

  for (int iter = 0; iter < config.num_iterations; ++iter) {
      /// for the self play, we need to clone the network to the cpu - potential source of error.
      auto best_clone = best_network->NetworkClone(torch::kCPU);
      auto network_to_train = std::dynamic_pointer_cast<NeuralNetwork>(best_clone);

      /// start the self play and collect the episodes.
      logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations);
      GamePositions positions;
      if (config.exhaustive_self_play) {
        positions = AllEpisodes<Game>();
      }
      else if (config.num_threads > 1) {
        positions = ExecuteEpisodesParallel<Game>(best_network, config);
      } else {
        positions = ExecuteEpisode<Game>(best_network, config);
      }
      logger.LogFormat("Collected {} game positions from self-play", positions.boards.size());

      /// was the self play good?
      float exploration_metric = 0.0f;
      for (const auto& policy : positions.policies) {
        exploration_metric += CalculatePolicyEntropy(policy);
      }
      if (!positions.policies.empty()) {
        exploration_metric /= positions.policies.size();
        logger.LogFormat("Average exploration metric: {:.4f}", exploration_metric);
      }

      /// train - then accept/reject
      Train(optimizer, network_to_train, config, positions);
                      
      EvaluationStats evaluation_stats = 
          EvaluateAgainstNetwork<Game>(network_to_train, best_network, config);
          
      bool network_accepted = 
          AcceptOrRejectNewNetwork(network_to_train, best_network, evaluation_stats, config );
                                  
      if (network_accepted) {
        best_network = std::dynamic_pointer_cast<NeuralNetwork>(
            network_to_train->NetworkClone(torch::kCPU));
        SaveBestNetwork(best_network, config);
        iterations_since_improvement = 0;
      }
      else iterations_since_improvement++;

      /// every 5 iterations evaluation against random and 
      /// log the results
      logger.Log("Evaluating against random network ...");
      EvaluateAgainstRandom<Game>(best_network, config);
  }
  
  logger.Log("Training complete!");
  return 0;
} 