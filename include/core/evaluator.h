#ifndef ALPHAZERO_EVALUATOR_H
#define ALPHAZERO_EVALUATOR_H

#include "game.h"
#include "mcts.h"
#include "config.h"
#include <memory>
#include <format>

namespace alphazero {

// New structure to hold detailed evaluation outcomes.
struct EvaluationStats {
    float win_rate;
    float draw_rate;
    float loss_rate;

    std::string WinStats() const {
        return std::format("Win rate: {}%, Draw rate: {}%, Loss rate: {}%",
            win_rate * 100, draw_rate * 100, loss_rate * 100);
    }

    bool IsBetterThan(const EvaluationStats& other) const {
        float score = win_rate + draw_rate * 0.5;
        float other_score = other.win_rate + other.draw_rate * 0.5;
        return score > other_score;
    }
};

class Evaluator {
public:
    Evaluator(std::shared_ptr<NeuralNetwork> network, const Config& config);
    
    // Play against random player
    EvaluationStats EvaluateAgainstRandom();
    
    // Play two networks against each other
    EvaluationStats EvaluateNetworks(std::shared_ptr<NeuralNetwork> network1, 
                          std::shared_ptr<NeuralNetwork> network2,
                          int num_games = 100);

    EvaluationStats EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent);

    // New detailed evaluation method.
    EvaluationStats EvaluateAgainstNetworkDetailed(
        std::shared_ptr<NeuralNetwork> opponent);

private:
    static constexpr int kNumEvaluationGames = 100;  // Fixed constant
    std::shared_ptr<NeuralNetwork> network_;
    const Config& config_;
};

} // namespace alphazero

#endif // ALPHAZERO_EVALUATOR_H 