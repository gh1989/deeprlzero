#ifndef TRAINER_INL_
#define TRAINER_INL_

#include "games/concepts.h"
#include "network.h"
#include "config.h"
#include "games/positions.h"
#include "eval_stats.h"
#include "mcts.h"
namespace deeprlzero {

template <typename GameType>
requires GameConcept<GameType>
void Train(std::shared_ptr<torch::optim::Optimizer> optimizer, 
           std::shared_ptr<NeuralNetwork> network, 
           const Config& config, 
           const GamePositions& positions) {
  if (positions.boards.empty()) {
    throw std::runtime_error("No positions to train on");
  }

  torch::Device device(torch::kCUDA);
  network->to(device);
  network->train();

  std::vector<torch::Tensor> policy_tensors;
  for (const auto& policy : positions.policies) {
    policy_tensors.push_back(torch::from_blob(
        const_cast<float*>(policy.data()), 
        {static_cast<int64_t>(policy.size())},
        torch::kFloat32).clone());
  }

  auto states = torch::stack(positions.boards).to(device);
  auto policies = torch::stack(policy_tensors).to(device);
  auto values = torch::tensor(positions.values).reshape({-1, 1}).to(device);
  
  // TODO: For longer games, the value targets would need temporal adjustment
  // such as discounted returns or TD(λ) approaches to properly handle the 
  // delayed nature of rewards in games like chess

  for (int epoch = 0; epoch < config.num_epochs; ++epoch) {
    optimizer->zero_grad();

    auto outputs = network->forward(states);
    auto policy_preds = outputs.first;
    auto value_preds = outputs.second;
    
    auto loss_policy = -torch::mean(
        torch::sum(policies * torch::log_softmax(policy_preds, 1), 1));
    auto loss_value = torch::mse_loss(value_preds, values);
    auto total_loss = loss_policy + loss_value;

    total_loss.backward();
    optimizer->step();
  }

  network->to(torch::kCPU);
}

template <typename GameType>
requires GameConcept<GameType>
EvaluationStats EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> network,
                                     std::shared_ptr<NeuralNetwork> opponent,
                                     const Config& config) {
  network->to(torch::kCPU);
  network->eval();
  opponent->to(torch::kCPU);
  opponent->eval();

  int wins_first = 0, draws_first = 0, losses_first = 0;
  int wins_second = 0, draws_second = 0, losses_second = 0;
  const int total_games = config.num_evaluation_games;

  MCTS mcts_main(network, config);
  MCTS mcts_opponent(opponent, config);

  for (int i = 0; i < total_games; ++i) {
    GameType game;
    bool network_plays_first = (i % 2 == 0);

    while (!game.IsTerminal()) {
      bool is_network_turn =
          ((game.GetCurrentPlayer() == 1) == network_plays_first);

      if (is_network_turn) {
        mcts_main.ResetRoot();
        for (int sim = 0; sim < config.num_simulations * 4; ++sim) {
          mcts_main.Search(game, mcts_main.GetRoot());
        }
        int action = mcts_main.SelectMove(game, config.eval_temperature);
        game.MakeMove(action);
      } else {
        mcts_opponent.ResetRoot();
        for (int sim = 0; sim < config.num_simulations * 4; ++sim) {
          mcts_opponent.Search(game, mcts_opponent.GetRoot());
        }
        int action = mcts_opponent.SelectMove(game, config.eval_temperature);
        game.MakeMove(action);
      }
    }

    float result = game.GetGameResult();
    
    if (network_plays_first) {
      if (result > 0) wins_first++;
      else if (result < 0) losses_first++;
      else draws_first++;
    } else {
      if (result < 0) wins_second++;
      else if (result > 0) losses_second++;
      else draws_second++;
    }
  }

  return EvaluationStats(wins_first, draws_first, losses_first, 
                        wins_second, draws_second, losses_second, total_games);
}

template <typename GameType>
requires GameConcept<GameType>
EvaluationStats EvaluateAgainstRandom(std::shared_ptr<NeuralNetwork> network,
                                    const Config& config) {
  network->to(torch::kCPU);
  network->eval();
  
  int wins_first = 0, draws_first = 0, losses_first = 0;
  int wins_second = 0, draws_second = 0, losses_second = 0;
  const int total_games = config.num_evaluation_games;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  MCTS mcts(network, config);
  
  for (int i = 0; i < total_games; ++i) {
    GameType game;
    bool network_plays_first = (i % 2 == 0);
    
    while (!game.IsTerminal()) {
      bool is_network_turn =
          ((game.GetCurrentPlayer() == 1) == network_plays_first);
      
      if (is_network_turn) {
        mcts.ResetRoot();
        for (int sim = 0; sim < config.num_simulations * 4; ++sim) {
          mcts.Search(game, mcts.GetRoot());
        }
        int action = mcts.SelectMove(game, config.eval_temperature);
        game.MakeMove(action);
      } else {
        auto valid_moves = game.GetValidMoves();
        std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
        int action = valid_moves[dis(gen)];
        game.MakeMove(action);
      }
    }
    
    float result = game.GetGameResult();
    
    if (network_plays_first) {
      if (result > 0) wins_first++;
      else if (result < 0) losses_first++;
      else draws_first++;
    } else {
      if (result < 0) wins_second++;
      else if (result > 0) losses_second++;
      else draws_second++;
    }
  }
  
  return EvaluationStats(wins_first, draws_first, losses_first, 
                        wins_second, draws_second, losses_second, total_games);
}

}  // namespace deeprlzero

#endif  // TRAINER_INL_