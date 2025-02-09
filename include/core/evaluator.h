#ifndef ALPHAZERO_EVALUATOR_H
#define ALPHAZERO_EVALUATOR_H

#include "game.h"
#include "mcts.h"
#include <memory>

namespace alphazero {

class Evaluator {
public:
    Evaluator(std::shared_ptr<NeuralNetwork> network, 
              float c_puct = 1.0f,
              int num_simulations = 100);
    
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
    float c_puct_;
    int num_simulations_;
};

} // namespace alphazero

#endif // ALPHAZERO_EVALUATOR_H 