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
    
  float current_temperature = config.initial_temperature;
  for (int iter = 0; iter < config.num_iterations; ++iter) {
      /// for the self play, we need to clone the network to the cpu - potential source of error.
      auto best_clone = best_network->clone(torch::kCPU);
      auto network_to_train = std::dynamic_pointer_cast<NeuralNetwork>(best_clone);
      SelfPlay<TicTacToe> self_play( best_network, config, current_temperature );
      Trainer trainer(network_to_train, config);
 
      /// start the self play and collect the episodes.
      logger.LogFormat("Starting iteration {}/{}", iter + 1, config.num_iterations);
      std::vector<GameEpisode> episodes;
      if (config.num_threads > 1) {
        episodes = self_play.ExecuteEpisodesParallel();
      } else {
        episodes.reserve(config.episodes_per_iteration);
        for (int i = 0; i < config.episodes_per_iteration; ++i) {
          episodes.push_back(self_play.ExecuteEpisode());
        }
      }
      logger.Log("Completed self-play episodes generation.");

      /// was the self play good?
      float exploration_metric = CalculateAverageExplorationMetric(episodes);
      /// train - then accept/reject
      trainer.Train(episodes);
      EvaluationStats evaluation_stats = trainer.EvaluateAgainstNetwork(best_network);
      bool network_accepted = trainer.AcceptOrRejectNewNetwork(best_network, evaluation_stats);

      ///there's no temperature update in the trainer.
      ///trainer.UpdateTemperature(current_temperature);

      /// every 5 iterations evaluation against random and 
      /// log the results
      trainer.EvaluateAgainstRandom();
  }
  
  logger.Log("Training complete!");
  return 0;
} 