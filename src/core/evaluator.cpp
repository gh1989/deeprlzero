#include "core/evaluator.h"
#include "core/tictactoe.h"
#include "core/mcts.h"
#include <random>
#include <iostream>
#include <algorithm>

namespace alphazero {

Evaluator::Evaluator(std::shared_ptr<NeuralNetwork> network, const Config& config)
    : network_(network), config_(config) {}

bool Evaluator::IsIdenticalNetwork(std::shared_ptr<NeuralNetwork> network1, 
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

EvaluationStats Evaluator::EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent) {    
    network_->to(torch::kCPU);
    network_->eval();
    opponent->to(torch::kCPU);
    opponent->eval();
    
{
        // Debug main network parameters
        float main_param_sum = 0.0f;
        int main_param_count = 0;
        for (const auto& param : network_->parameters()) {
            auto data = param.data_ptr<float>();
            for (int64_t i = 0; i < param.numel(); ++i) {
                main_param_sum += std::abs(data[i]);
                main_param_count++;
            }
        }
        
        // Debug opponent network parameters
        float opp_param_sum = 0.0f;
        int opp_param_count = 0;
        for (const auto& param : opponent->parameters()) {
            auto data = param.data_ptr<float>();
            for (int64_t i = 0; i < param.numel(); ++i) {
                opp_param_sum += std::abs(data[i]);
                opp_param_count++;
            }
        }
        
        /*
        std::cout << "\n=== EVALUATION NETWORKS COMPARISON ===" << std::endl;
        std::cout << "Main network: sum=" << main_param_sum 
                  << ", avg=" << (main_param_sum / main_param_count) << std::endl;
        std::cout << "Opponent network: sum=" << opp_param_sum 
                  << ", avg=" << (opp_param_sum / opp_param_count) << std::endl;
        std::cout << "Different?: " << (std::abs(main_param_sum - opp_param_sum) > 1e-6 ? "YES" : "NO") << std::endl;
        std::cout << "=====================================\n" << std::endl;
        */
    }

    if (IsIdenticalNetwork(network_, opponent)) {
        throw std::runtime_error("Evaluator: Cannot evaluate a network against an identical network!");
    }
    
    int wins = 0, draws = 0, losses = 0;
    const int total_games = config_.num_evaluation_games;
    
    MCTS mcts_main(network_, config_);
    MCTS mcts_opponent(opponent, config_);
    
    for (int i = 0; i < total_games; ++i) {
        auto game = std::make_unique<TicTacToe>();
        bool network_plays_first = (i % 2 == 0);
        
        //std::cout << "\nGame " << i << ": Main network plays " 
        //          << (network_plays_first ? "first" : "second") << "\n";
        int move_count = 0;
        
        while (!game->IsTerminal()) {
            bool is_network_turn = ((game->GetCurrentPlayer() == 1) == network_plays_first);
            
            if (is_network_turn) {
                mcts_main.ResetRoot();
                for (int sim = 0; sim < config_.num_simulations; ++sim) {
                    mcts_main.Search(game.get(), mcts_main.GetRoot());
                }
                int action = mcts_main.SelectMove(game.get(), 0.6f);
                //std::cout << "Main network move " << move_count << ": " << action << "\n";
                game->MakeMove(action);
            } else {
                mcts_opponent.ResetRoot();
                for (int sim = 0; sim < config_.num_simulations; ++sim) {
                    mcts_opponent.Search(game.get(), mcts_opponent.GetRoot());
                }
                int action = mcts_opponent.SelectMove(game.get(), 0.6f);
                //std::cout << "Opponent move " << move_count << ": " << action << "\n";
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
                int action = mcts.SelectMove(game.get(), 0.0f);
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
    return stats;
}

} // namespace alphazero