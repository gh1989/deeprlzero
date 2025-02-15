#include "core/evaluator.h"
#include "core/tictactoe.h"
#include "core/mcts.h"
#include <random>
#include <iostream>
#include <algorithm>

namespace alphazero {

Evaluator::Evaluator(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {}

EvaluationStats Evaluator::EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent) {
    int wins = 0;
    int draws = 0;
    int losses = 0;
    const int total_games = config_.num_evaluation_games;
    
    MCTS mcts_main(network_, config_);
    MCTS mcts_opponent(opponent, config_);
    
    for (int i = 0; i < total_games; ++i) {
        auto game = std::make_unique<TicTacToe>();
        bool network_plays_first = (i % 2 == 0);
        
        while (!game->IsTerminal()) {
            bool is_network_turn = ((game->GetCurrentPlayer() == 1) == network_plays_first);
            
            if (is_network_turn) {
                int action = mcts_main.SelectMove(*game, 0.0f);
                game->MakeMove(action);
            } else {
                int action = mcts_opponent.SelectMove(*game, 0.0f);
                game->MakeMove(action);
            }
        }
        
        float result = game->GetGameResult();
        if ((network_plays_first && result == 1.0f) ||
            (!network_plays_first && result == -1.0f)) {
            wins++;
        } else if (result == 0.0f) {
            draws++;
        } else {
            losses++;
        }
    }
    
    EvaluationStats stats;
    stats.win_rate = static_cast<float>(wins) / total_games;
    stats.draw_rate = static_cast<float>(draws) / total_games;
    stats.loss_rate = static_cast<float>(losses) / total_games;
    return stats;
}

EvaluationStats Evaluator::EvaluateAgainstRandom() {
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
            bool is_network_turn = ((game->GetCurrentPlayer() == 1) == network_plays_first);
            
            if (is_network_turn) {
                int action = mcts.SelectMove(*game, 0.0f);
                // Optionally, you can validate the action here.
                game->MakeMove(action);
            } else {
                auto valid_moves = game->GetValidMoves();
                std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
                int action = valid_moves[dis(gen)];
                game->MakeMove(action);
            }
        }
        
        float result = game->GetGameResult();
        if ((network_plays_first && result == 1.0f) ||
            (!network_plays_first && result == -1.0f)) {
            wins++;
        } else if (result == 0.0f) {
            draws++;
        } else {
            losses++;
        }
    }
    
    EvaluationStats stats;
    stats.win_rate = static_cast<float>(wins) / total_games;
    stats.draw_rate = static_cast<float>(draws) / total_games;
    stats.loss_rate = static_cast<float>(losses) / total_games;
    return stats;
}

// New method: detailed evaluation that counts wins, draws, and losses.
EvaluationStats Evaluator::EvaluateAgainstNetworkDetailed(
    std::shared_ptr<NeuralNetwork> opponent) {
    int wins = 0;
    int draws = 0;
    int losses = 0;
    const int total_games = config_.num_evaluation_games;
    
    MCTS mcts_main(network_, config_);
    MCTS mcts_opponent(opponent, config_);
    
    for (int i = 0; i < total_games; ++i) {
        auto game = std::make_unique<TicTacToe>();
        bool network_plays_first = (i % 2 == 0);
        
        while (!game->IsTerminal()) {
            bool is_network_turn = (game->GetCurrentPlayer() == 1) == network_plays_first;
            
            if (is_network_turn) {
                int action = mcts_main.SelectMove(*game, 0.0f);
                game->MakeMove(action);
            } else {
                int action = mcts_opponent.SelectMove(*game, 0.0f);
                game->MakeMove(action);
            }
        }
        
        float result = game->GetGameResult();
        if ((network_plays_first && result == 1.0f) ||
            (!network_plays_first && result == -1.0f)) {
            wins++;
        } else if (result == 0.0f) {
            draws++;
        } else {
            losses++;
        }
    }
    
    EvaluationStats stats;
    stats.win_rate = static_cast<float>(wins) / total_games;
    stats.draw_rate = static_cast<float>(draws) / total_games;
    stats.loss_rate = static_cast<float>(losses) / total_games;
    return stats;
}

} // namespace alphazero