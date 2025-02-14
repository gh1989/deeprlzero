#ifndef ALPHAZERO_EVALUATOR_H
#define ALPHAZERO_EVALUATOR_H

#include "game.h"
#include "mcts.h"
#include "config.h"
#include <memory>

namespace alphazero {

class Evaluator {
public:
    Evaluator(std::shared_ptr<NeuralNetwork> network, const Config& config);
    
    // Play against random player
    float EvaluateAgainstRandom();
    
    // Play two networks against each other
    float EvaluateNetworks(std::shared_ptr<NeuralNetwork> network1, 
                          std::shared_ptr<NeuralNetwork> network2,
                          int num_games = 100);

    float EvaluateAgainstNetwork(std::shared_ptr<NeuralNetwork> opponent);

private:
    static constexpr int kNumEvaluationGames = 100;  // Fixed constant
    std::shared_ptr<NeuralNetwork> network_;
    const Config& config_;
};

} // namespace alphazero

#endif // ALPHAZERO_EVALUATOR_H 