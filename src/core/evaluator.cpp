#include "core/evaluator.h"
#include "core/tictactoe.h"
#include <random>
#include <iostream>

namespace alphazero {

Evaluator::Evaluator(std::shared_ptr<NeuralNetwork> network, 
                     float c_puct, 
                     int num_simulations)
    : network_(network),
      c_puct_(c_puct),
      num_simulations_(num_simulations) {
    // Constructor implementation
}

float Evaluator::EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent) {
    float score = 0.0f;
    
    // Create MCTS instances for both networks
    MCTS mcts_main(network_, c_puct_, num_simulations_);
    MCTS mcts_opponent(opponent, c_puct_, num_simulations_);
    
    static constexpr int kNumEvaluationGames = 40;  // Reduced from 100 for faster evaluation
    for (int i = 0; i < kNumEvaluationGames; ++i) {
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
            score += 1.0f;
        } else if (result == 0.0f) {  // Draw
            score += 0.5f;  // Count draws as half a win
        }
    }
    
    return score / kNumEvaluationGames;
}

float Evaluator::EvaluateAgainstRandom() {
    int wins = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    MCTS mcts(network_, c_puct_, num_simulations_);
    
    static constexpr int kNumEvaluationGames = 100;  // Fixed constant
    for (int i = 0; i < kNumEvaluationGames; ++i) {
        auto game = std::make_unique<TicTacToe>();
        bool network_plays_first = (i % 2 == 0);
        
        while (!game->IsTerminal()) {
            bool is_network_turn = (game->GetCurrentPlayer() == 1) == network_plays_first;
            
            if (is_network_turn) {
                int action = mcts.SelectMove(*game, 0.0f);  // Temperature 0 for deterministic play
                
                // Validate move
                auto valid_moves = game->GetValidMoves();
                if (std::find(valid_moves.begin(), valid_moves.end(), action) == valid_moves.end()) {
                    std::cerr << "Warning: Network selected invalid move " << action << std::endl;
                }
                
                game->MakeMove(action);
            } else {
                auto valid_moves = game->GetValidMoves();
                std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
                int random_idx = dis(gen);
                game->MakeMove(valid_moves[random_idx]);
            }
        }
        
        float result = game->GetGameResult();
        if ((network_plays_first && result == 1.0f) || 
            (!network_plays_first && result == -1.0f)) {
            wins++;
        }
    }
    
    return static_cast<float>(wins) / kNumEvaluationGames;
}

} // namespace alphazero