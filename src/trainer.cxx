#include <algorithm>
#include <iostream>
#include <random>
#include <torch/torch.h>

#include "mcts.h"
#include "trainer.h"
#include "logger.h"

#include "games/tictactoe.h" //Fix

namespace deeprlzero {

void Trainer::Train(const GamePositions& positions) {
  if (positions.boards.empty()) {
    throw std::runtime_error("No positions to train on");
  }

  // Save initial parameters for comparison
  std::vector<torch::Tensor> initial_params;
  for (const auto& param : network_->parameters()) {
    initial_params.push_back(param.clone());
  }

  // Move to GPU
  torch::Device device(torch::kCUDA);
  network_->to(device);
  network_->train();

  // Convert policies to tensor
  std::vector<torch::Tensor> policy_tensors;
  for (const auto& policy : positions.policies) {
    policy_tensors.push_back(torch::from_blob(
        const_cast<float*>(policy.data()), 
        {static_cast<int64_t>(policy.size())},
        torch::kFloat32).clone());
  }

  // Stack tensors and move to GPU
  auto states = torch::stack(positions.boards).to(device);
  auto policies = torch::stack(policy_tensors).to(device);
  auto values = torch::tensor(positions.values).reshape({-1, 1}).to(device);

  ///network_->ValidateGradientFlow(states, policies, values);

  // Training loop
  for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
    optimizer_->zero_grad();

    auto outputs = network_->forward(states);
    auto policy_preds = outputs.first;
    auto value_preds = outputs.second;
    
    auto loss_policy = -torch::mean(
        torch::sum(policies * torch::log_softmax(policy_preds, 1), 1));
    auto loss_value = torch::mse_loss(value_preds, values);
    auto total_loss = loss_policy + loss_value;  // Weight policy loss higher

    total_loss.backward();
    optimizer_->step();
  }

  // After training loop, check if parameters changed
  Logger& logger = Logger::GetInstance(config_);
  bool params_changed = false;
  float total_diff = 0.0f;
  
  for (size_t i = 0; i < network_->parameters().size(); ++i) {
    auto diff = (network_->parameters()[i] - initial_params[i]).abs().sum().item<float>();
    total_diff += diff;
    if (diff > 1e-6) {
      params_changed = true;
    }
  }
  
  logger.LogFormat("Parameter changes after training: total_diff={:.6f}, changed={}", 
                  total_diff, params_changed ? "YES" : "NO");
  
  // Move back to CPU for compatibility with other parts of the code
  network_->to(torch::kCPU);
}

bool Trainer::AcceptOrRejectNewNetwork(
    std::shared_ptr<NeuralNetwork> candidate_network,
    const EvaluationStats& stats
) {
    Logger& logger = Logger::GetInstance(config_);
    bool network_accepted = false;
    float win_loss_ratio = stats.WinLossRatio();
                          
    if (win_loss_ratio >= config_.acceptance_threshold) {             
        // Save the network
        NeuralNetwork::SaveBestNetwork(network_, config_);
        iterations_since_improvement_ = 0;
        network_accepted = true;
    } else {
        iterations_since_improvement_++;
        network_accepted = false;
    }

    logger.LogFormat("Network acceptance decision: {}", network_accepted ? "ACCEPTED" : "REJECTED");
    logger.Log(stats.WinStats());
    return network_accepted;
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

  Logger &logger = Logger::GetInstance(config_);
  float main_param_sum = 0.0f;
  int main_param_count = 0;
  for (const auto& param : network_->parameters()) {
    auto flat_param = param.flatten();
    main_param_count += param.numel();
    main_param_sum += flat_param.abs().sum().item<float>();
  }
  float main_avg = main_param_sum / main_param_count;
  logger.LogFormat("OUR TRAINING NETWORK- Parameter stats: sum={:.4f}, avg={:.4f}, count={}", main_param_sum, main_avg, main_param_count);

  float opponent_param_sum = 0.0f;
  int opponent_param_count = 0;
  for (const auto& param : opponent->parameters()) {
    auto flat_param = param.flatten();
    opponent_param_count += param.numel();
    opponent_param_sum += flat_param.abs().sum().item<float>();
  }
  float opponent_avg = opponent_param_sum / opponent_param_count;
  logger.LogFormat("LATEST ACCEPTED BENCHMARK - Parameter stats: sum={:.4f}, avg={:.4f}, count={}", opponent_param_sum, opponent_avg, opponent_param_count);

  if (IsIdenticalNetwork(network_, opponent)) {
    throw std::runtime_error(
        "Evaluator: Cannot evaluate a network against an identical network!");
  }

  int wins_first = 0, draws_first = 0, losses_first = 0;
  int wins_second = 0, draws_second = 0, losses_second = 0;
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
        int action = mcts_main.SelectMove(game.get(), config_.eval_temperature);
        game->MakeMove(action);
      } else {
        mcts_opponent.ResetRoot();
        for (int sim = 0; sim < config_.num_simulations * 4; ++sim) {
          mcts_opponent.Search(game.get(), mcts_opponent.GetRoot());
        }
        int action = mcts_opponent.SelectMove(game.get(), config_.eval_temperature);
        game->MakeMove(action);
      }
      move_count++;
    }

    float result = game->GetGameResult();
    float perspective_result = network_plays_first ? result : -result;

    if (perspective_result > 0) {
      if (network_plays_first) {
        wins_first++;
      } else {
        wins_second++;
      }
    } else if (perspective_result == 0) {
      if (network_plays_first) {
        draws_first++;
      } else {
        draws_second++;
      }
    } else {
      if (network_plays_first) {
        losses_first++;
      } else {
        losses_second++;
      }
    }
  }

  EvaluationStats stats(wins_first, draws_first, losses_first, wins_second, draws_second, losses_second, total_games);  
  return stats;
}

EvaluationStats Trainer::EvaluateAgainstRandom() {
  int wins_first = 0;
  int draws_first = 0;
  int losses_first = 0;
  int wins_second = 0;
  int draws_second = 0;
  int losses_second = 0;

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
        int action = mcts.SelectMove(game.get(), config_.eval_temperature);
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
      if (network_plays_first) {
        wins_first++;
      } else {
        wins_second++;
      }
    } else if (perspective_result == 0) {
      if (network_plays_first) {
        draws_first++;
      } else {
        draws_second++;
      }
    } else {
      if (network_plays_first) {
        losses_first++;
      } else {
        losses_second++;
      }
    }
  }

  EvaluationStats stats(wins_first, draws_first, losses_first, wins_second, draws_second, losses_second, total_games);
  Logger &logger = Logger::GetInstance(config_);          
  auto log_string = logger.Log(stats.WinStats());
  return stats;
}

torch::Tensor Trainer::ComputePolicyLoss(const torch::Tensor& policy_preds, const torch::Tensor& policy_targets) {
  return torch::nn::functional::cross_entropy(policy_preds, policy_targets);
}

torch::Tensor Trainer::ComputeValueLoss(const torch::Tensor& value_preds, const torch::Tensor& value_targets) {
  return torch::mse_loss(value_preds, value_targets);
}

} 