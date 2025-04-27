#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "core/mcts.h"
#include "core/trainer.h"
#include "core/logger.h"

namespace deeprlzero {

void Trainer::Train(const std::vector<GameEpisode>& episodes) {
  if (episodes.empty()) {
    throw std::runtime_error("No episodes to train on");
  }

  auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

  network_->to(device);
  network_->train();

  // Prepare training data
  std::vector<torch::Tensor> all_boards;
  std::vector<float> all_policies;
  std::vector<float> all_values;

  for (const auto& episode : episodes) {
    all_boards.insert(all_boards.end(), episode.boards.begin(),
                      episode.boards.end());
    for (const auto& policy : episode.policies) {
      all_policies.insert(all_policies.end(), policy.begin(), policy.end());
    }

    // Use the final outcome for all positions in this episode
    for (size_t i = 0; i < episode.boards.size(); ++i) {
      // Flip sign for player 2's perspective
      float value = (i % 2 == 0) ? episode.outcome : -episode.outcome;
      all_values.push_back(value);
    }
  }

  auto states = torch::stack(torch::TensorList(all_boards)).to(device);
  auto policies =
      torch::tensor(all_policies).reshape({-1, config_.action_size}).to(device);
  auto values = torch::tensor(all_values).reshape({-1, 1}).to(device);

  // Training loop
  for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
    optimizer_->zero_grad();

    auto outputs = network_->forward(states);
    auto policy_preds = outputs.first;
    auto value_preds = outputs.second;

    auto loss_policy = -torch::mean(
        torch::sum(policies * torch::log_softmax(policy_preds, 1), 1));
    auto loss_value = torch::mse_loss(value_preds, values);
    auto total_loss = loss_policy + loss_value;

    total_loss.backward();
    optimizer_->step();
  }
}

float Trainer::UpdateTemperature(float current_temperature) {
    current_temperature = std::max(
        config_.min_temperature, 
        current_temperature * config_.temperature_decay
    );
    return current_temperature;
};

bool Trainer::AcceptOrRejectNewNetwork(
    std::shared_ptr<NeuralNetwork> candidate_network,
    const EvaluationStats& stats
) {
    float win_loss_ratio = (stats.win_rate + 0.5f * stats.draw_rate) / 
                          (stats.loss_rate + 0.5f * stats.draw_rate);
                          
    if (win_loss_ratio >= config_ .acceptance_threshold) {
        network_ = candidate_network;
        NeuralNetwork::SaveBestNetwork(network_, config_);
        iterations_since_improvement_= 0;
        return true;
    } else {
        iterations_since_improvement_++;
        return false;
    }
}

bool Trainer::IsIdenticalNetwork(std::shared_ptr<NeuralNetwork> network1,
                                   std::shared_ptr<NeuralNetwork> network2) {
  // Check if networks are identical by comparing their parameters
  bool networks_identical = true;
  auto main_params = network1->parameters();
  auto opp_params = network2->parameters();

  if (main_params.size() != opp_params.size()) {
    networks_identical = false;
  } else {
    for (size_t i = 0; i < main_params.size(); i++) {
      if (!torch::equal(main_params[i], opp_params[i])) {
        networks_identical = false;
        break;
      }
    }
  }
  return networks_identical;
}

EvaluationStats Trainer::EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent) {
  network_->to(torch::kCPU);
  network_->eval();
  opponent->to(torch::kCPU);
  opponent->eval();

  float main_param_sum = 0.0f;
  int main_param_count = 0;
  for (const auto& param : network_->parameters()) {
    auto flat_param = param.flatten();
    main_param_count += param.numel();
    main_param_sum += flat_param.abs().sum().item<float>();
  }
  float main_avg = main_param_sum / main_param_count;
  std::cout << "Best network - Parameter stats: sum=" << main_param_sum
            << ", avg=" << main_avg << ", count=" << main_param_count
            << std::endl;

  float opponent_param_sum = 0.0f;
  int opponent_param_count = 0;
  for (const auto& param : opponent->parameters()) {
    auto flat_param = param.flatten();
    opponent_param_count += param.numel();
    opponent_param_sum += flat_param.abs().sum().item<float>();
  }
  float opponent_avg = opponent_param_sum / opponent_param_count;
  std::cout << "Opponent network - Parameter stats: sum=" << opponent_param_sum
            << ", avg=" << opponent_avg << ", count=" << opponent_param_count
            << std::endl;

  if (IsIdenticalNetwork(network_, opponent)) {
    throw std::runtime_error(
        "Evaluator: Cannot evaluate a network against an identical network!");
  }

  int wins = 0, draws = 0, losses = 0;
  const int total_games = config_.num_evaluation_games;

  MCTS mcts_main(network_, config_);
  MCTS mcts_opponent(opponent, config_);

  for (int i = 0; i < total_games; ++i) {
    auto game = std::make_unique<TicTacToe>();  // :(
    bool network_plays_first = (i % 2 == 0);

    int move_count = 0;

    while (!game->IsTerminal()) {
      bool is_network_turn =
          ((game->GetCurrentPlayer() == 1) == network_plays_first);

      if (is_network_turn) {
        mcts_main.ResetRoot();
        for (int sim = 0; sim < config_.num_simulations * 4; ++sim) {
          mcts_main.Search(game.get(), mcts_main.GetRoot());
        }
        int action = mcts_main.SelectMove(game.get(), 0.6f);
        game->MakeMove(action);
      } else {
        mcts_opponent.ResetRoot();
        for (int sim = 0; sim < config_.num_simulations * 4; ++sim) {
          mcts_opponent.Search(game.get(), mcts_opponent.GetRoot());
        }
        int action = mcts_opponent.SelectMove(game.get(), 0.6f);
        game->MakeMove(action);
      }
      move_count++;
    }

    float result = game->GetGameResult();
    float perspective_result = network_plays_first ? result : -result;

    if (perspective_result > 0) {
      losses++;  // Opponent lost
      std::cout << "L";
    } else if (perspective_result == 0) {
      draws++;
      std::cout << "D";
    } else {
      wins++;  // Opponent won
      std::cout << "W";
    }
  }

  EvaluationStats stats;
  stats.win_rate = static_cast<float>(wins) / total_games;
  stats.draw_rate = static_cast<float>(draws) / total_games;
  stats.loss_rate = static_cast<float>(losses) / total_games;
  return stats;
}

EvaluationStats Trainer::EvaluateAgainstRandom() {
  int wins = 0;
  int draws = 0;
  int losses = 0;
  const int total_games = config_.num_evaluation_games;

  std::random_device rd;
  std::mt19937 gen(rd());

  MCTS mcts(network_, config_);

  for (int i = 0; i < total_games; ++i) {
    auto game = std::make_unique<TicTacToe>();
    bool network_plays_first = (i % 2 == 0);

    while (!game->IsTerminal()) {
      bool is_network_turn =
          ((game->GetCurrentPlayer() == 1) == network_plays_first);

      if (is_network_turn) {
        mcts.ResetRoot();
        for (int sim = 0; sim < config_.num_simulations * 4; ++sim) {
          mcts.Search(game.get(), mcts.GetRoot());
        }
        int action = mcts.SelectMove(game.get(), 0.0f);
        game->MakeMove(action);
      } else {
        auto valid_moves = game->GetValidMoves();
        std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
        int action = valid_moves[dis(gen)];
        game->MakeMove(action);
      }
    }

    float result = game->GetGameResult();
    float perspective_result = network_plays_first ? result : -result;

    if (perspective_result > 0) {
      wins++;
    } else if (perspective_result == 0) {
      draws++;
    } else {
      losses++;
    }
  }

  EvaluationStats stats;
  stats.win_rate = static_cast<float>(wins) / total_games;
  stats.draw_rate = static_cast<float>(draws) / total_games;
  stats.loss_rate = static_cast<float>(losses) / total_games;
        
  Logger &logger = Logger::GetInstance(config_);          
  auto log_string = logger.LogFormat("Performance vs Random: Win: {:.1f}%, Draw: {:.1f}%, Loss: {:.1f}%",
      stats.win_rate * 100,
      stats.draw_rate * 100,
      stats.loss_rate * 100);
  return stats;
}

torch::Tensor Trainer::ComputePolicyLoss(const torch::Tensor& policy_preds, const torch::Tensor& policy_targets) {
  return torch::nn::functional::cross_entropy(policy_preds, policy_targets);
}

torch::Tensor Trainer::ComputeValueLoss(const torch::Tensor& value_preds, const torch::Tensor& value_targets) {
  return torch::mse_loss(value_preds, value_targets);
}

} 