#include "core/self_play.h"
#include "core/tictactoe.h"
#include <random>
#include <cassert>
#include <numeric>
#include <iostream>

namespace alphazero {

SelfPlay::SelfPlay(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {
    network_->to(torch::kCPU);
    network_->eval();
}

std::vector<GameExample> SelfPlay::ExecuteEpisode() {
    // Ensure we're on CPU for parallel execution
    network_->to(torch::kCPU);
    std::vector<GameExample> examples;
    auto game = std::make_unique<TicTacToe>();
    MCTS mcts(network_, config_);
    
    // Game state initialization checks
    assert(game != nullptr);
    assert(game->GetActionSize() == 9 && "TicTacToe should have 9 possible actions");
    
    while (!game->IsTerminal()) {
        // Get and validate current game state
        auto current_board = game->GetCanonicalBoard();
        assert(current_board.defined() && "Board tensor must be defined");
        assert(current_board.numel() > 0 && "Board tensor cannot be empty");
        
        // Validate valid moves
        auto valid_moves = game->GetValidMoves();
        assert(!valid_moves.empty() && "Must have at least one valid move in non-terminal state");
        for (int move : valid_moves) {
            assert(move >= 0 && move < game->GetActionSize() && "Valid moves must be within bounds");
        }
        
        // Get and validate MCTS probabilities
        auto action_probs = mcts.GetActionProbabilities(*game, config_.temperature);
        assert(!action_probs.empty() && "MCTS must return non-empty probabilities");
        assert(action_probs.size() == game->GetActionSize() && "MCTS must return probabilities for all actions");
        
        // More detailed probability validation
        for (size_t i = 0; i < action_probs.size(); ++i) {
            if (action_probs[i] < 0.0f || action_probs[i] > 1.0f) {
                throw std::runtime_error("Invalid probability at index " + std::to_string(i) + 
                                       ": " + std::to_string(action_probs[i]));
            }
        }
        
        float prob_sum = std::accumulate(action_probs.begin(), action_probs.end(), 0.0f);
        if (std::abs(prob_sum - 1.0f) >= 1e-6f) {
            std::string probs_str;
            for (float p : action_probs) {
                probs_str += std::to_string(p) + " ";
            }
            throw std::runtime_error("Probabilities sum to " + std::to_string(prob_sum) + 
                                   " instead of 1.0. Probabilities: " + probs_str);
        }
        
        // Store example
        GameExample example;
        example.board = current_board;
        example.policy = action_probs;
        examples.push_back(example);
        
        // Filter and validate moves
        std::vector<float> filtered_probs = action_probs;
        std::vector<bool> is_valid(game->GetActionSize(), false);
        for (int move : valid_moves) {
            is_valid[move] = true;
        }
        
        for (size_t i = 0; i < filtered_probs.size(); ++i) {
            if (!is_valid[i]) {
                filtered_probs[i] = 0.0f;
            }
        }
        
        // Validate filtered probabilities
        float sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        assert(sum > 0.0f && "Must have at least one valid move with non-zero probability");
        
        // Normalize and validate
        for (float& prob : filtered_probs) {
            prob /= sum;
        }
        
        float normalized_sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        assert(std::abs(normalized_sum - 1.0f) < 1e-6f && "Normalized probabilities must sum to 1");
        
        // Move selection
        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        int move = dist(rng_);  // Using class member RNG
        
        // Validate selected move
        assert(move >= 0 && move < game->GetActionSize() && "Selected move must be within bounds");
        assert(is_valid[move] && "Selected move must be valid");
        
        // Make move and validate game state
        game->MakeMove(move);
        assert(std::find(valid_moves.begin(), valid_moves.end(), move) != valid_moves.end() && 
               "Selected move must have been in valid moves list");
        
        // After each move, accumulate the stats
        const MCTSStats& move_stats = mcts.GetStats();
        mcts_stats_ = move_stats;  // This will aggregate the stats
        mcts.ClearStats();  // Clear for next move
    }
    
    // Validate final game state and examples
    float final_value = game->GetGameResult();
    assert(final_value >= -1.0f && final_value <= 1.0f && "Game result must be in [-1, 1]");
    assert(!examples.empty() && "Must have collected at least one example");
    
    for (auto& example : examples) {
        example.value = final_value;
        final_value = -final_value;
    }
    
    return examples;
}

}  // namespace alphazero 