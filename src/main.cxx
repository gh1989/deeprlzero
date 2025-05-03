#include "network.h"
#include "trainer.h"
#include "config.h"
#include "logger.h"
#include "thread.h"
#include "game.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <format>

using namespace deeprlzero;

int main(int argc, char** argv) {
  auto config = Config::ParseCommandLine(argc, argv);
  auto& logger = Logger::GetInstance(config);

  if (!torch::cuda::is_available()) {
    logger.Log("ERROR: CUDA is not available."); // dont even want to run if this is the case.
    return 1;
  }

  /// get the best network from the file or create a new one and just save that (static operations pass in the config.)
  std::shared_ptr<NeuralNetwork> best_network = nullptr;
  best_network = NeuralNetwork::LoadBestNetwork(config);
  if (!best_network) {
      logger.Log("No existing model found. Creating a new one.");
      best_network = NeuralNetwork::CreateInitialNetwork(config);
      NeuralNetwork::SaveBestNetwork(best_network, config);
  }
    
  for (int iter = 0; iter < config.num_iterations; ++iter) {
      /// for the self play, we need to clone the network to the cpu - potential source of error.
      auto best_clone = best_network->NetworkClone(torch::kCPU);
      auto network_to_train = std::dynamic_pointer_cast<NeuralNetwork>(best_clone);
      SelfPlay<TicTacToe> self_play(best_network, config);
      Trainer trainer(network_to_train, config);
 
      /// start the self play and collect the episodes.
      logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations);
      GamePositions positions;
      if (config.exhaustive_self_play) {
        positions = self_play.AllEpisodes();
      }
      else if (config.num_threads > 1) {
        positions = self_play.ExecuteEpisodesParallel();
      } else {
        positions = self_play.ExecuteEpisode();
      }
      logger.LogFormat("Collected {} game positions from self-play", positions.boards.size());

      /// was the self play good?
      float exploration_metric = 0.0f;
      for (const auto& policy : positions.policies) {
        exploration_metric += NeuralNetwork::CalculatePolicyEntropy(policy);
      }
      if (!positions.policies.empty()) {
        exploration_metric /= positions.policies.size();
        logger.LogFormat("Average exploration metric: {:.4f}", exploration_metric);
      }

      /// train - then accept/reject
      trainer.Train(positions);
      EvaluationStats evaluation_stats = trainer.EvaluateAgainstNetwork(best_network);
      bool network_accepted = trainer.AcceptOrRejectNewNetwork(best_network, evaluation_stats);
      if (network_accepted) {
        best_network = std::dynamic_pointer_cast<NeuralNetwork>(trainer.GetTrainedNetwork()->NetworkClone(torch::kCPU));
        NeuralNetwork::SaveBestNetwork(best_network, config);
      }

      /// every 5 iterations evaluation against random and 
      /// log the results
      logger.Log("Evaluating against random network ...");
      trainer.EvaluateAgainstRandom();
  }
  
  logger.Log("Training complete!");
  return 0;
} 