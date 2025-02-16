#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <memory>
#include <format>

// Include project headers.
#include "core/config.h"
#include "core/network_manager.h"
#include "core/self_play.h"
#include "core/trainer.h"
#include "core/evaluator.h"
#include "core/neural_network.h"

using namespace alphazero;

// Structure to hold the experiment result.
struct ExperimentResult {
  Config config;
  EvaluationStats eval_stats;
};

// Compute a simple performance score that rewards high win/draw rates and penalizes losses.
// For TicTacToe, ideal play converges to draws (loss_rate should be near 0).
float ComputePerformanceScore(const EvaluationStats& stats) {
  return stats.win_rate + stats.draw_rate - stats.loss_rate;
}

int main() {
  // Seed a random engine.
  std::random_device rd;
  std::mt19937 gen(rd());

  // Define discrete sets for hyperparameters.
  std::vector<float> cpuct_values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> temperature_values = {0.5f, 1.0f, 1.5f, 2.0f};
  std::vector<int> games_values = {10, 25, 50};
  std::vector<int> filters_values = {16, 32, 64};
  std::vector<float> prior_alpha_values = {0.5f, 0.75f, 1.0f};
  std::vector<float> gamma_beta_values = {0.5f, 1.0f, 1.5f};
  std::vector<float> temperature_decay_values = {0.95f, 1.0f, 1.05f};
  std::vector<float> learning_rate_values = {1e-4f, 1e-3f, 1e-2f};

  int num_experiments = 10;  // Number of experiments to run.
  std::vector<ExperimentResult> experiment_results;

  for (int exp_i = 0; exp_i < num_experiments; ++exp_i) {
    // Start from a default config then override with random choices.
    Config config;
    std::uniform_int_distribution<> dist_cpuct(0, cpuct_values.size() - 1);
    std::uniform_int_distribution<> dist_temp(0, temperature_values.size() - 1);
    std::uniform_int_distribution<> dist_games(0, games_values.size() - 1);
    std::uniform_int_distribution<> dist_filters(0, filters_values.size() - 1);
    std::uniform_int_distribution<> dist_prior_alpha(0, prior_alpha_values.size() - 1);
    std::uniform_int_distribution<> dist_gamma_beta(0, gamma_beta_values.size() - 1);
    std::uniform_int_distribution<> dist_temp_decay(0, temperature_decay_values.size() - 1);
    std::uniform_int_distribution<> dist_learning_rate(0, learning_rate_values.size() - 1);

    config.c_puct = cpuct_values[dist_cpuct(gen)];
    config.temperature = temperature_values[dist_temp(gen)];
    config.initial_temperature = config.temperature;
    config.episodes_per_iteration = games_values[dist_games(gen)];
    config.num_filters = filters_values[dist_filters(gen)];
    config.prior_alpha = prior_alpha_values[dist_prior_alpha(gen)];
    config.gamma_beta = gamma_beta_values[dist_gamma_beta(gen)];
    config.temperature_decay = temperature_decay_values[dist_temp_decay(gen)];
    config.learning_rate = learning_rate_values[dist_learning_rate(gen)];

    // Log the chosen hyperparameters.
    std::cout << "Experiment " << exp_i + 1 << " configuration:\n";
    std::cout << "  cpuct: " << config.c_puct << "\n";
    std::cout << "  temperature: " << config.temperature << "\n";
    std::cout << "  games per iteration: " << config.episodes_per_iteration << "\n";
    std::cout << "  num filters: " << config.num_filters << "\n";
    std::cout << "  prior alpha: " << config.prior_alpha << "\n";
    std::cout << "  gamma beta: " << config.gamma_beta << "\n";
    std::cout << "  temperature decay: " << config.temperature_decay << "\n";
    std::cout << "  learning rate: " << config.learning_rate << "\n";

    // Initialize the network and network manager with this configuration.
    NetworkManager network_manager(config);
    std::shared_ptr<NeuralNetwork> network = network_manager.CreateInitialNetwork();
    network_manager.SetBestNetwork(network);

    // Run a few training iterations to simulate convergence.
    int training_iterations = 5;
    for (int iter = 0; iter < training_iterations; ++iter) {
      SelfPlay self_play(network, config);
      std::vector<GameEpisode> episodes;
      for (int i = 0; i < config.episodes_per_iteration; ++i) {
        episodes.push_back(self_play.ExecuteEpisode());
      }

      Trainer trainer(network, config);
      trainer.Train(network, episodes);
      network_manager.UpdateTemperature();
    }

    // Evaluate network performance against a random opponent.
    network->to(torch::kCPU);
    network->eval();
    Evaluator evaluator(network, config);
    EvaluationStats eval_stats = evaluator.EvaluateAgainstRandom();

    std::cout << "Evaluation results for experiment " << exp_i + 1 << ":\n";
    std::cout << "  " << eval_stats.WinStats() << "\n";

    ExperimentResult result { config, eval_stats };
    experiment_results.push_back(result);
  }

  // Determine the best hyperparameter configuration (highest performance score).
  float best_score = -std::numeric_limits<float>::infinity();
  int best_index = -1;
  for (size_t i = 0; i < experiment_results.size(); ++i) {
    float score = ComputePerformanceScore(experiment_results[i].eval_stats);
    std::cout << "Experiment " << i + 1 << " performance score: " << score << "\n";
    if (score > best_score) {
      best_score = score;
      best_index = static_cast<int>(i);
    }
  }

  if (best_index != -1) {
    std::cout << "\nBest hyperparameter configuration is from experiment " << best_index + 1
              << " with a performance score of " << best_score << ":\n";
    const Config &best_config = experiment_results[best_index].config;
    std::cout << "  cpuct: " << best_config.c_puct << "\n";
    std::cout << "  temperature: " << best_config.temperature << "\n";
    std::cout << "  games per iteration: " << best_config.episodes_per_iteration << "\n";
    std::cout << "  num filters: " << best_config.num_filters << "\n";
    std::cout << "  prior alpha: " << best_config.prior_alpha << "\n";
    std::cout << "  gamma beta: " << best_config.gamma_beta << "\n";
    std::cout << "  temperature decay: " << best_config.temperature_decay << "\n";
    std::cout << "  learning rate: " << best_config.learning_rate << "\n";
  }

  return 0;
}
